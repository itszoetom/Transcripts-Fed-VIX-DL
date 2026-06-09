"""Render a Pearson-r-only version of the regression comparison figure.

The original regression_comparison.png has two panels (Pearson r and R^2);
the user wants only the Pearson r panel. We rebuild it from the JSON reports
rather than cropping the PNG so the result is at full resolution and matches
the look of the original.

Run from the repo root:
    python -m attention_fed_vix.scripts.make_pearson_only_figure
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path("outputs/figures/regression_pearson_only.png")
ACCENT_COLOR = "#1F4E79"
MODEL_COLOR = "#4C78A8"
BASELINE_COLOR = "#F58518"


def main() -> None:
    fr = json.loads(Path("outputs/final_report.json").read_text())
    bl = json.loads(Path("outputs/baseline_metrics.json").read_text())
    regimes = ["regime1_2017_2021", "regime2_2021_2025", "regime3_2025_present"]
    pretty = ["regime 1\n2017-2021\n(Trump 1)",
              "regime 2\n2021-2025\n(Biden)",
              "regime 3\n2025-present\n(Trump 2)"]

    deep_r = [fr.get(r, {}).get("regression", {}).get("pearson_r", float("nan"))
              for r in regimes]
    base_r = [bl.get(r, {}).get("tfidf_ridge", {}).get("pearson_r", float("nan"))
              for r in regimes]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(regimes))
    width = 0.36
    # Pad both sides so the wide x-tick labels are not clipped.
    ax.set_xlim(-0.7, len(regimes) - 0.3)

    ax.bar(x - width / 2, deep_r, width, label="Sentence-attention model",
           color=MODEL_COLOR, edgecolor="white", linewidth=1.2)
    ax.bar(x + width / 2, base_r, width, label="TF-IDF Ridge baseline",
           color=BASELINE_COLOR, edgecolor="white", linewidth=1.2)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(pretty, fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=12)
    ax.set_title("Per-regime Pearson r: model vs. baseline",
                 fontsize=14, fontweight="bold", color=ACCENT_COLOR, pad=10)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate the bars with their numeric values. For negative values, place
    # the label just ABOVE the zero baseline (not below the bar) so it does
    # not collide with the x-axis tick labels beneath the chart.
    for xi, r_deep, r_base in zip(x, deep_r, base_r):
        deep_y = r_deep + 0.012 if r_deep >= 0 else 0.012
        base_y = r_base + 0.012 if r_base >= 0 else 0.012
        ax.text(xi - width / 2, deep_y, f"{r_deep:+.2f}",
                ha="center", va="bottom",
                fontsize=10, color=MODEL_COLOR, fontweight="bold")
        ax.text(xi + width / 2, base_y, f"{r_base:+.2f}",
                ha="center", va="bottom",
                fontsize=10, color=BASELINE_COLOR, fontweight="bold")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
