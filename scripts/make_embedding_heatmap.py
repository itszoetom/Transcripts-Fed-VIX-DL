"""Render a single sentence's 768-d FinBERT embedding as a horizontal heatmap.

For the Model slide. Makes the abstract notion of "an embedding" visually
concrete: every sentence becomes a row of 768 numbers, here colored to show
their varying magnitudes.

Run from the repo root:
    python scripts/make_embedding_heatmap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT = Path("outputs/figures/embedding_heatmap.png")
ACCENT_COLOR = "#1F4E79"


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    emb_dict = torch.load("data/example/example_embeddings.pt",
                          map_location="cpu", weights_only=True)

    # Pick a clean, short, financially-meaningful sentence to visualize.
    # The 2nd sentence of HH 1997-02-26 (Greenspan) is a short, contentful line.
    doc_id = "hh_19970226"
    sentence_idx = 1  # 0-indexed
    chosen = next(d for d in docs if d["doc_id"] == doc_id)
    sentence = chosen["sentences"][sentence_idx]
    embedding = emb_dict[doc_id][sentence_idx].numpy()  # (768,)

    fig = plt.figure(figsize=(12, 3.4))

    # Top: the sentence itself, in a stylized text box.
    ax_text = fig.add_axes([0.04, 0.66, 0.92, 0.20])
    ax_text.axis("off")
    ax_text.text(
        0, 0.5,
        f'"{sentence}"',
        fontsize=13, color="#1A1A1A", style="italic",
        va="center", ha="left", wrap=True,
    )
    ax_text.text(
        0, 1.0,
        f"Sentence from FOMC chair testimony, {doc_id.replace('hh_', 'Feb. ').replace('19970226', '26, 1997')}",
        fontsize=10, color="#555555", va="center", ha="left",
    )

    # Below: the 768-d embedding rendered as a horizontal heatmap.
    ax_heat = fig.add_axes([0.04, 0.20, 0.92, 0.32])
    im = ax_heat.imshow(
        embedding[None, :],
        aspect="auto",
        cmap="RdBu_r",
        vmin=-np.abs(embedding).max(),
        vmax=np.abs(embedding).max(),
    )
    ax_heat.set_yticks([])
    ax_heat.set_xticks([0, 128, 256, 384, 512, 640, 767])
    ax_heat.tick_params(axis="x", labelsize=10)
    ax_heat.set_xlabel("Embedding dimension index (0 to 767)", fontsize=11)

    # Colorbar on the right
    cax = fig.add_axes([0.97, 0.20, 0.015, 0.32])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Label above the heatmap
    ax_heat.set_title(
        "FinBERT turns the sentence above into 768 numbers (one row, color-coded)",
        fontsize=11, color=ACCENT_COLOR, pad=8,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
