"""Timeline figure showing the train, val, and three test regimes.

For the Data slide. Renders a horizontal timeline from 1993 to 2026 with
colored bands for each split, annotated with U.S. presidents and the
inauguration-day boundaries that anchor the splits. The figure makes it
visually clear that:

  - We train ONE model on pre-2017 data.
  - The val set is the last 15 percent of the pre-2017 range.
  - The three test regimes are not used during training; they are held out
    so we can test whether the relationship the model learned generalizes
    forward in time.

Run from the repo root:
    python scripts/make_timeline_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUTPUT = Path("outputs/figures/timeline_splits.png")

# Color palette
ACCENT_COLOR = "#1F4E79"
TRAIN_COLOR = "#A7C7E7"  # light blue for the training pool
VAL_COLOR = "#FFD27F"    # gold for the val carve-out
TEST1_COLOR = "#E89090"  # light red for Trump 1 era test
TEST2_COLOR = "#90C290"  # light green for Biden era test
TEST3_COLOR = "#C89BC8"  # purple for Trump 2 era test
BODY_TEXT = "#1A1A1A"


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))

    year_start = 1993
    year_end = 2026.5

    band_y = 1.2
    band_h = 0.9

    # Train pool: 1993 to ~2014 (the part that is NOT carved off into val).
    train_pool_end = 2014.0
    train_pool = Rectangle((year_start, band_y), train_pool_end - year_start,
                           band_h, facecolor=TRAIN_COLOR,
                           edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.add_patch(train_pool)
    ax.text((year_start + train_pool_end) / 2, band_y + band_h / 2,
            "TRAIN  (1993 to ~2014)\n233 documents",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=BODY_TEXT)

    # Val carve-out: last 15 percent of pre-2017 chronologically.
    val_start = train_pool_end
    val_end = 2017.054  # 2017-01-20 inauguration day
    val = Rectangle((val_start, band_y), val_end - val_start, band_h,
                    facecolor=VAL_COLOR, edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.add_patch(val)
    ax.text((val_start + val_end) / 2, band_y + band_h / 2,
            "VAL\n35 docs",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=BODY_TEXT)

    # Regime 1: Trump 1 era test set
    r1_start = 2017.054
    r1_end = 2021.054
    r1 = Rectangle((r1_start, band_y), r1_end - r1_start, band_h,
                   facecolor=TEST1_COLOR, edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.add_patch(r1)
    ax.text((r1_start + r1_end) / 2, band_y + band_h / 2,
            "TEST regime 1\n40 docs",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=BODY_TEXT)

    # Regime 2: Biden era test set
    r2_start = 2021.054
    r2_end = 2025.054
    r2 = Rectangle((r2_start, band_y), r2_end - r2_start, band_h,
                   facecolor=TEST2_COLOR, edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.add_patch(r2)
    ax.text((r2_start + r2_end) / 2, band_y + band_h / 2,
            "TEST regime 2\n40 docs",
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=BODY_TEXT)

    # Regime 3: Trump 2 era test set
    r3_start = 2025.054
    r3_end = year_end
    r3 = Rectangle((r3_start, band_y), r3_end - r3_start, band_h,
                   facecolor=TEST3_COLOR, edgecolor=ACCENT_COLOR, linewidth=1.2)
    ax.add_patch(r3)
    ax.text((r3_start + r3_end) / 2, band_y + band_h / 2,
            "regime 3\n13 docs",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color=BODY_TEXT)

    # President strip BELOW the data bands
    pres_y = 0.6
    pres_h = 0.42
    presidents = [
        ("Clinton", 1993, 2001.054),
        ("G. W. Bush", 2001.054, 2009.054),
        ("Obama", 2009.054, 2017.054),
        ("Trump 1", 2017.054, 2021.054),
        ("Biden", 2021.054, 2025.054),
        ("Trump 2", 2025.054, year_end),
    ]
    for name, p_start, p_end in presidents:
        ax.add_patch(Rectangle((p_start, pres_y), p_end - p_start, pres_h,
                               facecolor="white", edgecolor="#777777", linewidth=0.8))
        size = 9 if (p_end - p_start) > 1.5 else 8
        ax.text((p_start + p_end) / 2, pres_y + pres_h / 2, name,
                ha="center", va="center", fontsize=size, color=BODY_TEXT)

    # Fed Chair strip BELOW the president strip (presidentially appointed,
    # tenures cross multiple administrations -> makes the political-pressure
    # angle visible at a glance).
    chair_y = 0.15
    chair_h = 0.36
    chairs = [
        ("Greenspan (1987 to 2006)", 1993, 2006.06),
        ("Bernanke (2006 to 2014)", 2006.06, 2014.07),
        ("Yellen (2014 to 2018)", 2014.07, 2018.10),
        ("Powell (2018 to present)", 2018.10, year_end),
    ]
    for name, c_start, c_end in chairs:
        ax.add_patch(Rectangle((c_start, chair_y), c_end - c_start, chair_h,
                               facecolor="#F2F2F2", edgecolor="#777777", linewidth=0.8))
        if c_end - c_start > 4.0:
            ax.text((c_start + c_end) / 2, chair_y + chair_h / 2, name,
                    ha="center", va="center", fontsize=8, color=BODY_TEXT)
        else:
            # Shorter name for narrow boxes
            short = name.split(" (")[0]
            ax.text((c_start + c_end) / 2, chair_y + chair_h / 2, short,
                    ha="center", va="center", fontsize=8, color=BODY_TEXT)

    # Row labels on the left
    ax.text(year_start - 0.6, band_y + band_h / 2, "splits",
            ha="right", va="center", fontsize=9, color=BODY_TEXT, style="italic")
    ax.text(year_start - 0.6, pres_y + pres_h / 2, "President",
            ha="right", va="center", fontsize=9, color=BODY_TEXT, style="italic")
    ax.text(year_start - 0.6, chair_y + chair_h / 2, "Fed Chair",
            ha="right", va="center", fontsize=9, color=BODY_TEXT, style="italic")

    # Vertical inauguration-day boundary lines (the actual split anchors).
    # Labels sit slightly above the data bands and below the title so they
    # don't collide with either.
    for x, label in [(2017.054, "2017-01-20\nTrump 1 inaug."),
                     (2021.054, "2021-01-20\nBiden inaug."),
                     (2025.054, "2025-01-20\nTrump 2 inaug.")]:
        ax.axvline(x, ymin=0.10, ymax=0.62, color=ACCENT_COLOR, linewidth=1.3,
                   linestyle="--", alpha=0.85)
        ax.text(x, 2.65, label, ha="center", va="bottom", fontsize=8.5,
                color=ACCENT_COLOR, fontweight="bold")

    # Year tick marks across the bottom
    ax.set_xticks([1993, 2001, 2009, 2017, 2021, 2025])
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlim(year_start - 1.5, year_end + 0.3)
    ax.set_ylim(0.1, 4.3)
    ax.set_yticks([])

    # Hide top/right/left spines
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)

    # Title and subtitle sit above the inauguration-date labels so nothing overlaps.
    ax.text(year_start, 4.15,
            "Train on pre-2017 data, test on three later time periods",
            ha="left", va="bottom", fontsize=14, fontweight="bold",
            color=ACCENT_COLOR)
    ax.text(year_start, 3.65,
            "Split boundaries: U.S. presidential inauguration days. The model never sees post-2017 data during training.",
            ha="left", va="bottom", fontsize=10, color=BODY_TEXT, style="italic")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
