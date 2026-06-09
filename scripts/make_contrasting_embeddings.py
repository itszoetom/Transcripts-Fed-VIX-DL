"""Render two contrasting sentence embeddings stacked.

Picks one reassuring/calm sentence and one inflation-concerned sentence from
the bundled example documents and shows their 768-d FinBERT embeddings as
horizontal heatmaps. Colormap range is set from joint percentiles so within-
embedding structure is visible (without one outlier dimension dominating).

Run from the repo root:
    python scripts/make_contrasting_embeddings.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUTPUT = Path("outputs/figures/contrasting_embeddings.png")
ACCENT_COLOR = "#1F4E79"
BODY_DARK = "#1A1A1A"


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    emb_dict = torch.load("data/example/example_embeddings.pt",
                          map_location="cpu", weights_only=True)

    # Reassuring/calm sentence: HH 1997-02-26, sentence 2
    calm_doc_id = "hh_19970226"
    calm_sent_idx = 1
    calm_sentence = next(d for d in docs if d["doc_id"] == calm_doc_id)["sentences"][calm_sent_idx]
    calm_emb = emb_dict[calm_doc_id][calm_sent_idx].numpy()

    # Inflation-concerned sentence: HH 2004-07-20, sentence 5
    concerned_doc_id = "hh_20040720"
    concerned_sent_idx = 4
    concerned_sentence = next(d for d in docs if d["doc_id"] == concerned_doc_id)["sentences"][concerned_sent_idx]
    concerned_emb = emb_dict[concerned_doc_id][concerned_sent_idx].numpy()

    # Joint percentile range so both heatmaps share a meaningful color scale.
    joint = np.concatenate([calm_emb, concerned_emb])
    vmax = float(np.percentile(np.abs(joint), 95))

    fig = plt.figure(figsize=(11, 4.8))

    # Top sentence + heatmap
    ax_t_text = fig.add_axes([0.04, 0.78, 0.92, 0.16])
    ax_t_text.axis("off")
    ax_t_text.text(0, 0.6, f'"{calm_sentence}"',
                   fontsize=11, color=BODY_DARK, style="italic",
                   va="center", ha="left", wrap=True)
    ax_t_text.text(0, 1.05, "Sentence A: from Humphrey-Hawkins testimony, Feb. 26 1997",
                   fontsize=10, color=ACCENT_COLOR, fontweight="bold",
                   va="center", ha="left")

    ax_t_heat = fig.add_axes([0.04, 0.58, 0.88, 0.13])
    im_t = ax_t_heat.imshow(calm_emb[None, :], aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax)
    ax_t_heat.set_yticks([])
    ax_t_heat.set_xticks([0, 128, 256, 384, 512, 640, 767])
    ax_t_heat.tick_params(axis="x", labelsize=9)

    # Bottom sentence + heatmap
    ax_b_text = fig.add_axes([0.04, 0.30, 0.92, 0.16])
    ax_b_text.axis("off")
    ax_b_text.text(0, 0.6, f'"{concerned_sentence}"',
                   fontsize=11, color=BODY_DARK, style="italic",
                   va="center", ha="left", wrap=True)
    ax_b_text.text(0, 1.05, "Sentence B: from Humphrey-Hawkins testimony, Jul. 20 2004",
                   fontsize=10, color=ACCENT_COLOR, fontweight="bold",
                   va="center", ha="left")

    ax_b_heat = fig.add_axes([0.04, 0.10, 0.88, 0.13])
    im_b = ax_b_heat.imshow(concerned_emb[None, :], aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax)
    ax_b_heat.set_yticks([])
    ax_b_heat.set_xticks([0, 128, 256, 384, 512, 640, 767])
    ax_b_heat.tick_params(axis="x", labelsize=9)
    ax_b_heat.set_xlabel("Embedding dimension (0 to 767)", fontsize=10)

    # Shared colorbar on the right
    cax = fig.add_axes([0.93, 0.10, 0.012, 0.61])
    cbar = fig.colorbar(im_b, cax=cax)
    cbar.set_label("value", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
