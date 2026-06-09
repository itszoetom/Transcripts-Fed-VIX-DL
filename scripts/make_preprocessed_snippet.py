"""Render a styled snippet of preprocessed Fed-text data.

Shows the first ~6 numbered sentences from a real Humphrey-Hawkins testimony
document. Goes on the Dataset slide to make "sentence-segmented, capped at
80" tangible.

Run from the repo root:
    python scripts/make_preprocessed_snippet.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path("outputs/figures/preprocessed_snippet.png")
ACCENT_COLOR = "#1F4E79"
BODY_DARK = "#1A1A1A"
PANEL_BG = "#F8F8F8"


def _wrap(text: str, width: int = 78) -> str:
    """Soft-wrap a sentence to a max line width by word."""
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for w in words:
        if sum(len(x) for x in cur) + len(cur) + len(w) > width:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    chosen = next(d for d in docs if d["doc_id"] == "hh_20040720")
    indices = [1, 2, 3, 4, 5]
    # Limit long sentences (e.g., sentence 1 is header boilerplate) to ~210
    # display chars so the snippet stays compact. The model itself sees the
    # full sentences; this truncation is for the slide only.
    def _trim(s: str, limit: int = 210) -> str:
        return s if len(s) <= limit else s[: limit - 1].rstrip() + "..."
    # Clean up a known PDF-extraction quirk: a mangled em-dash that came
    # through as the Polish "S-acute" character. Replace with a real dash for
    # display readability.
    def _clean(s: str) -> str:
        return s.replace("Ś", " - ").replace("ś", " - ")
    sentences = [_clean(_trim(chosen["sentences"][i - 1])) for i in indices]

    fig = plt.figure(figsize=(8, 5.0))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.axis("off")

    # Header strip
    ax.text(0.02, 0.94,
            "Preprocessed input  -  first 80 sentences of one document",
            color=ACCENT_COLOR, fontsize=13, fontweight="bold",
            transform=ax.transAxes, ha="left", va="top")
    ax.text(0.02, 0.87,
            "Source: Humphrey-Hawkins testimony, Chairman Greenspan, 2004-07-20",
            color="#555555", fontsize=9, style="italic",
            transform=ax.transAxes, ha="left", va="top")

    # Bullet block
    cur_y = 0.78
    for sent_num, sentence in zip(indices, sentences):
        wrapped = _wrap(sentence, width=78)
        n_lines = wrapped.count("\n") + 1
        # Sentence number badge on the left
        ax.text(0.03, cur_y, f"[{sent_num}]",
                color=ACCENT_COLOR, fontsize=10, fontweight="bold",
                transform=ax.transAxes, ha="left", va="top", family="monospace")
        # Sentence text
        ax.text(0.09, cur_y, wrapped,
                color=BODY_DARK, fontsize=10,
                transform=ax.transAxes, ha="left", va="top")
        cur_y -= 0.038 * n_lines + 0.04

    # Footer note
    ax.text(0.02, 0.04,
            "(sentences 6 through 80 truncated for display)",
            color="#777777", fontsize=9, style="italic",
            transform=ax.transAxes, ha="left", va="bottom")

    # Subtle panel background
    from matplotlib.patches import FancyBboxPatch
    bg = FancyBboxPatch((0.005, 0.005), 0.99, 0.99,
                        boxstyle="round,pad=0.005,rounding_size=0.02",
                        facecolor=PANEL_BG, edgecolor=ACCENT_COLOR, linewidth=1.0,
                        transform=ax.transAxes, zorder=-1)
    ax.add_patch(bg)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
