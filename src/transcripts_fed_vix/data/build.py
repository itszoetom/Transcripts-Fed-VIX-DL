"""End-to-end builder for the processed dataset.

Orchestrates:

    scrape FOMC + Humphrey-Hawkins
        -> segment each document into the first 80 sentences (NLTK punkt)
            -> fetch VIX closes (FRED VIXCLS) and align to next trading day
                -> compute 3-day forward VIX change as the regression target
                    -> persist a single parquet under data/processed/

Output columns (one row per document):

    doc_id                str    e.g. "fomc_20180131"
    source                str    "fomc" | "humphrey_hawkins"
    release_date          date   document's public release date
    aligned_trading_date  date   first VIX trading day >= release_date
    vix_t                 float  VIX close on aligned_trading_date
    vix_t_plus_3          float  VIX close 3 trading days later
    target                float  vix_t_plus_3 - vix_t  (regression target)
    sentences             list[str]   first 80 sentences of the document
    url                   str    URL the text was scraped from

The function is idempotent: re-running with everything already on disk just
returns the cached parquet. Pass `force=True` to rebuild from scraped raw
files (without re-downloading).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

from .scrape import scrape_fomc_minutes, scrape_humphrey_hawkins, scraped_docs_to_records
from .segment import segment_document, SENTENCE_CAP
from .vix import fetch_vix, compute_forward_change

logger = logging.getLogger(__name__)


def build_processed_dataset(
    raw_dir: Path,
    processed_dir: Path,
    *,
    processed_filename: str = "documents.parquet",
    vix_cache_filename: str = "vix.csv",
    force: bool = False,
) -> pd.DataFrame:
    """Build (or load) the processed dataset.

    Args:
        raw_dir:            data/raw/ directory; will be created if missing.
        processed_dir:      data/processed/ directory; will be created if missing.
        processed_filename: Name of the parquet file under processed_dir.
        vix_cache_filename: Name of the VIX cache CSV under raw_dir.
        force:              If True, rebuild the parquet even if it already
                            exists. Does not force re-scraping (scrape cache
                            files under raw_dir are still respected).

    Returns:
        DataFrame with the schema described in the module docstring.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / processed_filename

    if processed_path.exists() and not force:
        logger.info("loading cached processed dataset from %s", processed_path)
        return pd.read_parquet(processed_path)

    # 1) Scrape both sources. Scrape functions cache raw HTML/PDF + extracted
    #    text under raw_dir/<source>/, so re-runs are cheap.
    logger.info("scraping FOMC minutes …")
    fomc_docs = scrape_fomc_minutes(raw_dir)
    logger.info("scraping Humphrey-Hawkins testimony …")
    hh_docs = scrape_humphrey_hawkins(raw_dir)
    all_docs = list(fomc_docs) + list(hh_docs)
    if not all_docs:
        raise RuntimeError(
            "No documents scraped. Check network access and that "
            "data/raw/fomc/index_*.html / humphrey_hawkins/index_*.html were fetched."
        )
    logger.info("total scraped documents: %d", len(all_docs))

    # 2) Sentence-segment + 80-sentence truncation.
    sentence_lists: list[list[str]] = []
    for d in all_docs:
        sents = segment_document(d.text, cap=SENTENCE_CAP)
        sentence_lists.append(sents)

    # 3) Fetch VIX (cached) and align each release to next trading day + target.
    vix = fetch_vix(cache_path=raw_dir / vix_cache_filename)

    rows: list[dict] = []
    dropped = 0
    for doc, sents in zip(all_docs, sentence_lists):
        if not sents:
            dropped += 1
            logger.warning("dropping %s: empty after segmentation", doc.doc_id)
            continue
        align = compute_forward_change(doc.release_date, vix)
        if align is None:
            dropped += 1
            logger.warning("dropping %s: no VIX target available (too recent or too old)", doc.doc_id)
            continue
        rows.append(
            {
                "doc_id": doc.doc_id,
                "source": doc.source,
                "release_date": doc.release_date,
                "aligned_trading_date": align.aligned_trading_date,
                "vix_t": align.vix_t,
                "vix_t_plus_3": align.vix_t_plus_h,
                "target": align.target,
                "sentences": sents,
                "url": doc.url,
                "n_sentences": len(sents),
            }
        )

    if dropped:
        logger.info("dropped %d documents during build (no sentences or no VIX target)", dropped)

    df = pd.DataFrame(rows)
    # Sort chronologically, critical for downstream temporal splits.
    df = df.sort_values("release_date").reset_index(drop=True)

    # Pandas stores `date` as object; convert release/aligned to pyarrow
    # timestamp-friendly form before writing parquet.
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["aligned_trading_date"] = pd.to_datetime(df["aligned_trading_date"])

    df.to_parquet(processed_path, index=False)
    logger.info("wrote %d rows to %s", len(df), processed_path)

    # Persist the discovery manifest summary alongside the dataset for the
    # methodology section.
    summary = {
        "n_rows": int(len(df)),
        "n_fomc": int((df["source"] == "fomc").sum()),
        "n_humphrey_hawkins": int((df["source"] == "humphrey_hawkins").sum()),
        "first_release": str(df["release_date"].min().date()),
        "last_release": str(df["release_date"].max().date()),
        "sentence_cap": SENTENCE_CAP,
    }
    (processed_dir / "build_summary.json").write_text(json.dumps(summary, indent=2))
    return df
