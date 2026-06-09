"""Render an architecture diagram for the model slide.

Boxes-and-arrows showing the full forward pass: sentences in, scalar prediction
out, with frozen vs trained components color-coded. Saved as a 300 DPI PNG.

Run from the repo root:
    python scripts/make_architecture_diagram.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT = Path("outputs/figures/architecture.png")

# Color palette. Frozen components are gray; trained components are blue.
FROZEN_COLOR = "#D9D9D9"
TRAINED_COLOR = "#4C78A8"
ACCENT_COLOR = "#1F4E79"
TEXT_DARK = "#1A1A1A"


def _box(ax, xy, wh, text, color, text_color="white", fontsize=11) -> None:
    """Draw a rounded rectangle with centered text."""
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.15",
        linewidth=1.2, edgecolor=ACCENT_COLOR, facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold")


def _arrow(ax, xy_from, xy_to) -> None:
    """Draw a directed arrow between two coordinates."""
    arrow = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color=ACCENT_COLOR,
    )
    ax.add_patch(arrow)


def _label(ax, xy, text, color=TEXT_DARK, fontsize=9) -> None:
    """Add a small italic annotation."""
    ax.text(xy[0], xy[1], text, color=color, fontsize=fontsize,
            ha="center", va="center", style="italic")


def main() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4.5)
    ax.axis("off")

    # Single horizontal flow, top to bottom title and legend.
    # All boxes sit on the row y in [1.4, 3.0] with annotations below.

    # 1. Input sentences
    _box(ax, (0.1, 1.6), (1.7, 1.4),
         "FOMC minutes\n+ HH testimony\n(first 80 sentences)",
         color="white", text_color=TEXT_DARK, fontsize=10)

    # 2. Frozen FinBERT
    _box(ax, (2.1, 1.6), (1.8, 1.4),
         "FinBERT\n(frozen)\n110M params",
         color=FROZEN_COLOR, text_color=TEXT_DARK, fontsize=11)
    _label(ax, (3.0, 1.35), "(no gradients)")

    # 3. Per-sentence embeddings
    _box(ax, (4.2, 1.8), (1.6, 1.0),
         "80 x 768\nsentence\nembeddings",
         color="white", text_color=TEXT_DARK, fontsize=10)

    # 4. Additive attention (trained)
    _box(ax, (6.1, 1.6), (1.8, 1.4),
         "Additive\nattention\n(~99k params)",
         color=TRAINED_COLOR, text_color="white", fontsize=11)
    _label(ax, (7.0, 1.35), "(trained)", color=ACCENT_COLOR)

    # 5. Document vector
    _box(ax, (8.2, 1.8), (1.3, 1.0),
         "doc vec\n(768)",
         color="white", text_color=TEXT_DARK, fontsize=10)

    # 6. Linear head (trained)
    _box(ax, (9.8, 1.6), (1.5, 1.4),
         "Linear\nhead\n(769 params)",
         color=TRAINED_COLOR, text_color="white", fontsize=11)
    _label(ax, (10.55, 1.35), "(trained)", color=ACCENT_COLOR)

    # 7. Output
    _box(ax, (11.5, 1.8), (1.4, 1.0),
         "predicted\n10-day\nVIX change",
         color=ACCENT_COLOR, text_color="white", fontsize=10)

    # Arrows (left to right)
    _arrow(ax, (1.8, 2.3), (2.1, 2.3))
    _arrow(ax, (3.9, 2.3), (4.2, 2.3))
    _arrow(ax, (5.8, 2.3), (6.1, 2.3))
    _arrow(ax, (7.9, 2.3), (8.2, 2.3))
    _arrow(ax, (9.5, 2.3), (9.8, 2.3))
    _arrow(ax, (11.3, 2.3), (11.5, 2.3))

    # Title at top
    ax.text(6.5, 4.1, "Frozen FinBERT + additive sentence attention + linear regression head",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=ACCENT_COLOR)
    ax.text(6.5, 3.6, "About 99,000 trained parameters; 110M frozen FinBERT parameters",
            ha="center", va="center", fontsize=10, color=TEXT_DARK, style="italic")

    # Color legend, bottom left
    legend_y = 0.45
    ax.add_patch(FancyBboxPatch((0.3, legend_y - 0.15), 0.35, 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=FROZEN_COLOR, edgecolor=ACCENT_COLOR))
    ax.text(0.75, legend_y, "frozen (no gradients)", fontsize=10, va="center")
    ax.add_patch(FancyBboxPatch((3.3, legend_y - 0.15), 0.35, 0.30,
                                boxstyle="round,pad=0.02,rounding_size=0.05",
                                facecolor=TRAINED_COLOR, edgecolor=ACCENT_COLOR))
    ax.text(3.75, legend_y, "trained (about 100k params)", fontsize=10, va="center")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
