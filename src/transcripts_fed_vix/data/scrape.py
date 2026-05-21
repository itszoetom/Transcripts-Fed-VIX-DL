"""Scrapers for Federal Reserve text sources.

Two document sources are pulled:

    1. FOMC minutes (1993-present, ~264 docs):
       index pages at
         https://www.federalreserve.gov/monetarypolicy/fomchistorical{YYYY}.htm
       (1993 through ~2018, archived layout) and
         https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
       (current layout, holds the most recent two years).

    2. Humphrey-Hawkins / Semiannual Monetary Policy Report testimony
       (1997-present, ~58 docs):
       index page at
         https://www.federalreserve.gov/newsevents/testimony.htm
       plus per-year archive pages for older testimony.

Why we cache to data/raw/ aggressively:
    The Federal Reserve site is courteous-rate-limited and occasionally
    reshuffles URLs. Re-running the scraper from scratch every time is wasteful
    and fragile. So we save every fetched HTML/PDF byte-for-byte to
    data/raw/<source>/<doc_id>.{html,pdf} and the extracted plain text to
    data/raw/<source>/<doc_id>.txt. Re-runs skip anything already on disk.

Why we extract plain text here rather than at modeling time:
    Sentence-level encoding requires clean text input. Doing the
    HTML-/PDF-stripping once and caching the result keeps the rest of the
    pipeline deterministic and lets us iterate on segmentation without
    re-downloading.

This module deliberately uses only `requests` + `beautifulsoup4` + `pypdf` (no
selenium / no headless browser) because every page we need is server-rendered
HTML or a static PDF.
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Base host. Kept as a constant so a future mirror can be substituted easily.
FED_HOST = "https://www.federalreserve.gov"

# Identify ourselves politely. The Fed serves all this material publicly, but a
# UA string with contact context is good citizenship and avoids occasional 403s
# from default Python/requests UA.
USER_AGENT = (
    "transcripts-fed-vix research scraper / academic use "
    "(contact: ztomlins@uoregon.edu)"
)

# Delay between HTTP requests, in seconds. Conservative — the bottleneck is not
# our scraper.
REQUEST_DELAY_SEC = 0.5

# Earliest years we attempt to scrape for each source. FOMC minutes start in
# 1993 (the year the Fed began releasing them); the Humphrey-Hawkins Act
# semiannual testimony has Fed-website coverage from 1997 onward.
FOMC_START_YEAR = 1993
HH_START_YEAR = 1997


@dataclass
class ScrapedDoc:
    """One scraped document, post text extraction.

    Attributes:
        doc_id:       Stable identifier of the form "{source}_{YYYYMMDD}".
        source:       Either "fomc" or "humphrey_hawkins".
        release_date: ISO date the document was first publicly released. For
                      FOMC minutes this is the *release date* (~3 weeks after
                      the meeting), not the meeting date itself, because the
                      release date is when markets could plausibly react.
        url:          Source URL the text was extracted from.
        text:         Plain-text body of the document.
    """

    doc_id: str
    source: str
    release_date: date
    url: str
    text: str


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------


def _session() -> requests.Session:
    """A `requests.Session` preconfigured with our user-agent."""
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def _fetch_bytes(sess: requests.Session, url: str, *, retries: int = 3) -> bytes:
    """GET `url` returning raw bytes, with retries and a polite delay."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = sess.get(url, timeout=30)
            resp.raise_for_status()
            time.sleep(REQUEST_DELAY_SEC)
            return resp.content
        except Exception as exc:  # noqa: BLE001 — broad on purpose, we retry
            last_exc = exc
            logger.warning("fetch failed (%s/%s) %s: %s", attempt + 1, retries, url, exc)
            time.sleep(2 ** attempt)  # exponential backoff
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _extract_html_text(html_bytes: bytes) -> str:
    """Strip HTML chrome and return body text.

    The Fed's minutes/testimony pages have a consistent <div id="article"> or
    <div id="content"> container in the modern layout, and a less-consistent
    layout in archived pages. We try a couple of selectors and fall back to
    full-body text extraction; downstream sentence segmentation tolerates
    trailing navigation cruft far better than missing body content.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    for selector in ("#article", "#content", "div.col-xs-12.col-sm-8.col-md-8"):
        node = soup.select_one(selector)
        if node is not None:
            return _normalize_whitespace(node.get_text(separator=" "))
    # Fallback: whole document.
    return _normalize_whitespace(soup.get_text(separator=" "))


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF byte string using pypdf.

    pypdf occasionally returns empty strings on scanned PDFs. The FOMC
    historical minutes from the 1990s are *text* PDFs (not scans), so this is
    rarely a problem in practice, but we log empty extractions so the user can
    spot-check them.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _normalize_whitespace(" ".join(pages))
    if not text.strip():
        logger.warning("PDF extracted to empty text — likely scanned image PDF")
    return text


_WS_RE = re.compile(r"\s+")


def _normalize_whitespace(s: str) -> str:
    """Collapse runs of whitespace into single spaces and strip."""
    return _WS_RE.sub(" ", s).strip()


# ---------------------------------------------------------------------------
# FOMC minutes
# ---------------------------------------------------------------------------

# Filename pattern used in modern Fed URLs: fomcminutesYYYYMMDD.<ext>
_FOMC_MINUTES_URL_RE = re.compile(
    r"/(?:monetarypolicy|fomc)/fomcminutes(?P<date>\d{8})\.(?P<ext>htm|pdf)",
    re.IGNORECASE,
)


def _parse_fomc_index_page(html_bytes: bytes) -> list[tuple[str, str]]:
    """Return list of (url, release_date_iso) from one FOMC index page.

    The Fed historical pages list each meeting with a "Minutes" link. The link
    URL always contains the meeting/release date in YYYYMMDD form, so we use
    that as our date stamp. We prefer HTML over PDF when both are linked.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    found: dict[str, tuple[str, str]] = {}  # date_iso -> (url, ext)

    for a in soup.find_all("a", href=True):
        m = _FOMC_MINUTES_URL_RE.search(a["href"])
        if not m:
            continue
        d = m.group("date")
        date_iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        ext = m.group("ext").lower()

        # Prefer HTML if we've already seen a PDF for this date.
        existing = found.get(date_iso)
        if existing is None or (existing[1] == "pdf" and ext == "htm"):
            url = a["href"]
            if url.startswith("/"):
                url = FED_HOST + url
            found[date_iso] = (url, ext)

    return [(url, date_iso) for date_iso, (url, _ext) in found.items()]


def scrape_fomc_minutes(
    raw_dir: Path,
    *,
    start_year: int = FOMC_START_YEAR,
    end_year: int | None = None,
) -> list[ScrapedDoc]:
    """Scrape FOMC minutes from start_year through end_year (inclusive).

    Returns a list of ScrapedDoc with text already extracted. Caches all
    intermediate artifacts under raw_dir / "fomc" /. Re-running is idempotent:
    anything cached on disk is reused rather than re-downloaded.

    Args:
        raw_dir:    Root data/raw/ directory.
        start_year: First calendar year to scrape (default 1993).
        end_year:   Last calendar year to scrape (default = current year).

    Returns:
        List of ScrapedDoc sorted by release_date ascending.
    """
    end_year = end_year or date.today().year
    fomc_dir = raw_dir / "fomc"
    fomc_dir.mkdir(parents=True, exist_ok=True)

    sess = _session()
    docs: list[ScrapedDoc] = []
    discovered: list[tuple[str, str]] = []  # (url, date_iso)

    for year in range(start_year, end_year + 1):
        # The historical-layout URL covers archived years. Newer years are also
        # mirrored on the historical pages once the calendar year ends, so this
        # one URL family works for the whole range we want — but in case it's
        # missing we silently move on (most-recent-year case).
        index_url = f"{FED_HOST}/monetarypolicy/fomchistorical{year}.htm"
        index_cache = fomc_dir / f"index_{year}.html"
        try:
            if index_cache.exists():
                html = index_cache.read_bytes()
            else:
                html = _fetch_bytes(sess, index_url)
                index_cache.write_bytes(html)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not fetch FOMC index for %s (%s): %s", year, index_url, exc)
            continue
        discovered.extend(_parse_fomc_index_page(html))

    # The most-recent-year index sometimes isn't on the historical layout yet.
    # Also scrape the current calendar page; duplicates are deduped by URL+date.
    calendars_url = f"{FED_HOST}/monetarypolicy/fomccalendars.htm"
    calendars_cache = fomc_dir / "fomccalendars.html"
    try:
        if calendars_cache.exists():
            html = calendars_cache.read_bytes()
        else:
            html = _fetch_bytes(sess, calendars_url)
            calendars_cache.write_bytes(html)
        discovered.extend(_parse_fomc_index_page(html))
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not fetch FOMC calendars: %s", exc)

    # Deduplicate by date_iso (prefer HTML; the prefer-HTML logic is already in
    # _parse_fomc_index_page within a single page, but across pages we still
    # need to break ties — keep the first one seen).
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for url, date_iso in discovered:
        if date_iso in seen:
            continue
        seen.add(date_iso)
        unique.append((url, date_iso))

    unique.sort(key=lambda t: t[1])
    logger.info("FOMC minutes: discovered %d unique documents", len(unique))

    for url, date_iso in unique:
        doc_id = f"fomc_{date_iso.replace('-', '')}"
        text = _download_and_extract_doc(sess, url, fomc_dir, doc_id)
        if not text:
            continue
        docs.append(
            ScrapedDoc(
                doc_id=doc_id,
                source="fomc",
                release_date=date.fromisoformat(date_iso),
                url=url,
                text=text,
            )
        )

    # Persist a discovery manifest for reproducibility.
    (fomc_dir / "discovered.json").write_text(
        json.dumps([{"url": u, "date": d} for u, d in unique], indent=2)
    )
    return docs


# ---------------------------------------------------------------------------
# Humphrey-Hawkins / Semiannual Monetary Policy Report testimony
# ---------------------------------------------------------------------------

# Match titles that identify the semiannual report testimony across the era
# during which the Fed has called it different things ("Humphrey-Hawkins",
# "Monetary Policy Report", "Semiannual Monetary Policy Report").
_HH_TITLE_RE = re.compile(
    r"(humphrey[-\s]hawkins|"
    r"semiannual\s+monetary\s+policy\s+report|"
    r"monetary\s+policy\s+report\s+to\s+the\s+congress)",
    re.IGNORECASE,
)

# Year-by-year testimony archive URLs.
_HH_INDEX_URLS = [f"{FED_HOST}/newsevents/testimony/{year}-testimony.htm"
                  for year in range(2006, date.today().year + 1)]
# Plus the current-year landing page.
_HH_INDEX_URLS.append(f"{FED_HOST}/newsevents/testimony.htm")
# Pre-2006 testimony lives at /boarddocs/hh/{YYYY}/(february|july)/testimony.htm etc.
# Coverage is patchy; we attempt a known pattern.
for year in range(HH_START_YEAR, 2006):
    for half in ("february", "july"):
        _HH_INDEX_URLS.append(f"{FED_HOST}/boarddocs/hh/{year}/{half}/testimony.htm")


def _parse_hh_index_page(base_url: str, html_bytes: bytes) -> list[tuple[str, str]]:
    """Return list of (url, release_date_iso) for HH-style testimony entries.

    The newer testimony layout lists each testimony as a row containing the
    title and a date. We pick rows whose title matches the HH/Monetary Policy
    Report regex and parse the date from the surrounding text (or from the
    URL if dated).
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    out: list[tuple[str, str]] = []

    # Modern layout: each testimony is an <a> inside a row also containing a
    # <time> element with the release date.
    for row in soup.select("div.row.eventlist-time, .eventlist__item, div.row"):
        title_node = row.find("a")
        if title_node is None:
            continue
        title = title_node.get_text(" ", strip=True)
        if not _HH_TITLE_RE.search(title):
            continue
        # Date may be a <time datetime="...">, otherwise look for a sibling
        # text like "February 14, 2023".
        d = _find_date_near(row)
        if d is None:
            continue
        url = title_node.get("href", "")
        if url.startswith("/"):
            url = FED_HOST + url
        elif not url.startswith("http"):
            # Relative to base_url's directory.
            url = base_url.rsplit("/", 1)[0] + "/" + url
        out.append((url, d.isoformat()))

    # Pre-2006 boarddocs layout: testimony page IS the testimony, so the
    # base_url's URL itself names the date in path; we don't iterate links
    # there — we treat the page itself as a doc if its title matches.
    if "/boarddocs/hh/" in base_url:
        page_title = soup.title.get_text(" ", strip=True) if soup.title else ""
        if _HH_TITLE_RE.search(page_title) or _HH_TITLE_RE.search(html_bytes.decode("utf-8", errors="ignore")):
            d = _find_date_in_text(soup.get_text(" "))
            if d is not None:
                out.append((base_url, d.isoformat()))

    return out


_DATE_TEXT_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s+(\d{4})"
)


def _find_date_near(node) -> date | None:
    """Best-effort date extraction from a list-row HTML node."""
    t = node.find("time")
    if t is not None and t.get("datetime"):
        try:
            return datetime.fromisoformat(t["datetime"][:10]).date()
        except Exception:  # noqa: BLE001
            pass
    return _find_date_in_text(node.get_text(" "))


def _find_date_in_text(text: str) -> date | None:
    """Find the first "Month D, YYYY" date string and parse it."""
    m = _DATE_TEXT_RE.search(text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(0), "%B %d, %Y").date()
    except ValueError:
        return None


def scrape_humphrey_hawkins(raw_dir: Path) -> list[ScrapedDoc]:
    """Scrape Humphrey-Hawkins / Monetary Policy Report testimony.

    Strategy: walk a list of testimony index pages (one per year for 2006+
    plus older "boarddocs" pages), filter rows whose titles match the
    Humphrey-Hawkins/Monetary Policy Report family, and download each match.

    Returns ScrapedDocs sorted by release_date.
    """
    hh_dir = raw_dir / "humphrey_hawkins"
    hh_dir.mkdir(parents=True, exist_ok=True)

    sess = _session()
    discovered: list[tuple[str, str]] = []

    for idx_url in _HH_INDEX_URLS:
        slug = idx_url.replace(FED_HOST, "").strip("/").replace("/", "_") or "root"
        cache = hh_dir / f"index_{slug}.html"
        try:
            if cache.exists():
                html = cache.read_bytes()
            else:
                html = _fetch_bytes(sess, idx_url)
                cache.write_bytes(html)
        except Exception as exc:  # noqa: BLE001
            logger.debug("HH index miss %s: %s", idx_url, exc)
            continue
        discovered.extend(_parse_hh_index_page(idx_url, html))

    # Deduplicate.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for url, date_iso in discovered:
        key = f"{date_iso}|{url}"
        if key in seen:
            continue
        seen.add(key)
        unique.append((url, date_iso))
    unique.sort(key=lambda t: t[1])
    logger.info("Humphrey-Hawkins: discovered %d unique documents", len(unique))

    docs: list[ScrapedDoc] = []
    for url, date_iso in unique:
        doc_id = f"hh_{date_iso.replace('-', '')}"
        text = _download_and_extract_doc(sess, url, hh_dir, doc_id)
        if not text:
            continue
        docs.append(
            ScrapedDoc(
                doc_id=doc_id,
                source="humphrey_hawkins",
                release_date=date.fromisoformat(date_iso),
                url=url,
                text=text,
            )
        )

    (hh_dir / "discovered.json").write_text(
        json.dumps([{"url": u, "date": d} for u, d in unique], indent=2)
    )
    return docs


# ---------------------------------------------------------------------------
# Per-document download + extraction (shared)
# ---------------------------------------------------------------------------


def _download_and_extract_doc(
    sess: requests.Session,
    url: str,
    out_dir: Path,
    doc_id: str,
) -> str:
    """Download a single document URL, cache it, extract text, cache that too.

    Returns the extracted plain text (possibly empty if extraction failed).
    """
    text_cache = out_dir / f"{doc_id}.txt"
    if text_cache.exists():
        return text_cache.read_text(encoding="utf-8")

    is_pdf = url.lower().endswith(".pdf")
    raw_cache = out_dir / f"{doc_id}.{'pdf' if is_pdf else 'html'}"

    try:
        if raw_cache.exists():
            raw_bytes = raw_cache.read_bytes()
        else:
            raw_bytes = _fetch_bytes(sess, url)
            raw_cache.write_bytes(raw_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to download %s for %s: %s", url, doc_id, exc)
        return ""

    try:
        text = _extract_pdf_text(raw_bytes) if is_pdf else _extract_html_text(raw_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.error("failed to extract text from %s (%s): %s", doc_id, url, exc)
        return ""

    text_cache.write_text(text, encoding="utf-8")
    return text


# ---------------------------------------------------------------------------
# Serialization for downstream use
# ---------------------------------------------------------------------------


def scraped_docs_to_records(docs: Iterable[ScrapedDoc]) -> list[dict]:
    """Convert ScrapedDoc instances to plain-dict records for parquet / JSON."""
    out = []
    for d in docs:
        rec = asdict(d)
        rec["release_date"] = d.release_date.isoformat()
        out.append(rec)
    return out
