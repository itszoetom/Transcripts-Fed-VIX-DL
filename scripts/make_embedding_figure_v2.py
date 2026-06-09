"""Render a clean, axis-labeled comparison of two FinBERT sentence embeddings.

Replaces the unlabeled strip-style figures on slide 4 with one figure that:
  - Shows TWO sentences with visibly different embeddings
  - Has a real x-axis (embedding dimension 0 to 767)
  - Has a labeled colorbar so the reader can tell the colors are scaled values
  - Picks one boilerplate-header sentence vs one economic-content sentence so
    the visual contrast between the two embeddings is sharp

Run from the repo root:
    python scripts/make_embedding_figure_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT = Path("outputs/figures/embedding_comparison.png")
ACCENT = "#1F4E79"
BODY = "#1A1A1A"


def _shorten(s: str, max_len: int = 90) -> str:
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "..."


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    emb_dict = torch.load("data/example/example_embeddings.pt",
                          map_location="cpu", weights_only=True)

    # Two sentences picked to maximize visual contrast in embedding space:
    # - Sentence A is the FRB header boilerplate (metadata-like text)
    # - Sentence B is a substantive economic-content sentence
    doc = next(d for d in docs if d["doc_id"] == "hh_19970226")
    sent_a = doc["sentences"][0]
    sent_b = doc["sentences"][1]
    emb_a = emb_dict["hh_19970226"][0].numpy()
    emb_b = emb_dict["hh_19970226"][1].numpy()

    # Build a 2-row x 768-col heatmap matrix so the contrast lives in one plot.
    matrix = np.stack([emb_a, emb_b], axis=0)
    vmax = float(np.percentile(np.abs(matrix), 95))

    fig = plt.figure(figsize=(5.2, 5.5))

    # Title above
    fig.text(0.02, 0.95,
             "FinBERT sentence embeddings  (each sentence becomes a 768-d vector)",
             color=ACCENT, fontsize=13, fontweight="bold", ha="left", va="top")

    # Heatmap area
    ax = fig.add_axes([0.16, 0.42, 0.74, 0.45])
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Sentence A\n(header)", "Sentence B\n(economic content)"],
                       fontsize=10, color=BODY)
    ax.set_xticks([0, 128, 256, 384, 512, 640, 767])
    ax.set_xticklabels([0, 128, 256, 384, 512, 640, 767], fontsize=10)
    ax.set_xlabel("Embedding dimension index (0 to 767)", fontsize=11, color=BODY,
                  labelpad=8)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    # Colorbar on the right
    cax = fig.add_axes([0.92, 0.42, 0.012, 0.45])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("embedding value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Sentence text below the heatmap (well below the x-axis label)
    fig.text(0.02, 0.18,
             f"A: \"{_shorten(sent_a, 100)}\"",
             color=BODY, fontsize=9, ha="left", va="center", style="italic")
    fig.text(0.02, 0.08,
             f"B: \"{_shorten(sent_b, 100)}\"",
             color=BODY, fontsize=9, ha="left", va="center", style="italic")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
