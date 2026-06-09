"""Single combined flow figure for the Model slide.

Renders the full pipeline as one horizontal diagram:
    document text  ->  FinBERT (frozen)  ->  two example embeddings
        ->  additive attention (trained)  ->  attention weights for one doc
            ->  linear head (trained)  ->  predicted 10-day VIX change

The two visualizations inside the flow (embeddings and weights) are real,
computed from the bundled example data + bundled trained model.

Run from the repo root:
    python scripts/make_combined_flow_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import AttentionConfig

OUTPUT = Path("outputs/figures/model_flow.png")

ACCENT_COLOR = "#1F4E79"
FROZEN_FILL = "#D9D9D9"
TRAINED_FILL = "#4C78A8"
OUTPUT_FILL = "#1F4E79"
BODY_DARK = "#1A1A1A"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_box(ax, xy, wh, text, fill, text_color, *, fontsize: int = 11) -> None:
    """Rounded rectangle with centered text."""
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.10",
        linewidth=1.4, edgecolor=ACCENT_COLOR, facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold")


def _arrow(ax, xy_from, xy_to) -> None:
    arr = FancyArrowPatch(
        xy_from, xy_to,
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color=ACCENT_COLOR,
    )
    ax.add_patch(arr)


def _add_text_panel(ax, xy, wh, header: str, sentences: list[str],
                    *, fontsize: int = 8) -> None:
    """Card-style box with numbered sentences."""
    x, y = xy
    w, h = wh
    bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=1.0, edgecolor=ACCENT_COLOR, facecolor="white")
    ax.add_patch(bg)
    ax.text(x + 0.05, y + h - 0.10, header, fontsize=fontsize + 1,
            color=ACCENT_COLOR, fontweight="bold", va="top", ha="left")
    for i, s in enumerate(sentences, start=1):
        text_y = y + h - 0.30 - i * 0.20
        ax.text(x + 0.10, text_y, f"[{i}]", fontsize=fontsize - 1,
                color=ACCENT_COLOR, fontweight="bold", va="top", ha="left",
                family="monospace")
        ax.text(x + 0.35, text_y, s, fontsize=fontsize, color=BODY_DARK,
                va="top", ha="left")


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    emb_dict = torch.load("data/example/example_embeddings.pt",
                          map_location="cpu", weights_only=True)
    model = SentenceAttentionModel(AttentionConfig())
    model.load_state_dict(torch.load("data/example/example_model.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()

    # Embeddings for the two contrasting sentences (computed inline so the
    # figure stays self-contained).
    calm = next(d for d in docs if d["doc_id"] == "hh_19970226")
    concerned = next(d for d in docs if d["doc_id"] == "hh_20040720")
    calm_emb = emb_dict["hh_19970226"][1].numpy()
    concerned_emb = emb_dict["hh_20040720"][4].numpy()
    joint = np.concatenate([calm_emb, concerned_emb])
    vmax_emb = float(np.percentile(np.abs(joint), 95))

    # Run the trained model on one document to get real attention weights.
    attn_doc = next(d for d in docs if d["doc_id"] == "hh_19970722")
    attn_sents = attn_doc["sentences"]
    attn_emb = emb_dict["hh_19970722"]
    with torch.no_grad():
        out = model(attn_emb.unsqueeze(0),
                    torch.ones(1, attn_emb.shape[0], dtype=torch.float32))
    attn_weights = out.attention_weights[0].numpy()
    attn_pred = float(out.prediction[0])

    # ----- layout -----
    fig = plt.figure(figsize=(14, 5.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    # Header
    ax.text(7, 5.2, "Pipeline: text  ->  embeddings  ->  attention  ->  prediction",
            ha="center", va="center", fontsize=15, fontweight="bold",
            color=ACCENT_COLOR)

    # 1. Sentences (left card)
    sample_sentences = [
        "The performance of the U.S. economy ... has been quite favorable.",
        "Real GDP growth picked up to more than three percent ...",
        "Employers added more than two-and-a-half million workers ...",
        "(... 77 more sentences)",
    ]
    _add_text_panel(ax, (0.1, 1.8), (2.5, 2.8),
                    "Document (80 sentences)",
                    sample_sentences, fontsize=8)

    # Arrow 1
    _arrow(ax, (2.65, 3.2), (3.0, 3.2))

    # 2. FinBERT box (frozen)
    _add_box(ax, (3.05, 2.3), (1.3, 1.6), "FinBERT\n(frozen)\n110M params",
             FROZEN_FILL, BODY_DARK, fontsize=10)
    ax.text(3.7, 2.05, "domain-pretrained", color=BODY_DARK,
            fontsize=8, style="italic", ha="center")

    # Arrow 2
    _arrow(ax, (4.4, 3.2), (4.75, 3.2))

    # 3. Two contrasting embedding strips
    # Top strip: calm sentence
    emb_box_x, emb_box_y = 4.85, 1.85
    emb_box_w, emb_box_h = 2.7, 2.7
    bg = FancyBboxPatch((emb_box_x, emb_box_y), emb_box_w, emb_box_h,
                        boxstyle="round,pad=0.03,rounding_size=0.08",
                        linewidth=1.0, edgecolor=ACCENT_COLOR, facecolor="white")
    ax.add_patch(bg)
    ax.text(emb_box_x + emb_box_w / 2, emb_box_y + emb_box_h - 0.18,
            "768-d sentence embeddings",
            ha="center", va="top", fontsize=9, color=ACCENT_COLOR, fontweight="bold")
    ax.text(emb_box_x + 0.1, emb_box_y + emb_box_h - 0.5,
            "Sentence A (1997 'favorable')", fontsize=7, color=BODY_DARK)
    ax.text(emb_box_x + 0.1, emb_box_y + emb_box_h - 1.45,
            "Sentence B (2004 'inflation')", fontsize=7, color=BODY_DARK)
    # Calm strip
    ax_a = fig.add_axes([emb_box_x / 14, (emb_box_y + 1.50) / 5.4,
                         (emb_box_w - 0.1) / 14, 0.10])
    ax_a.imshow(calm_emb[None, :], aspect="auto", cmap="RdBu_r",
                vmin=-vmax_emb, vmax=vmax_emb)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    for s in ["top", "bottom", "left", "right"]:
        ax_a.spines[s].set_visible(False)
    # Concerned strip
    ax_b = fig.add_axes([emb_box_x / 14, (emb_box_y + 0.30) / 5.4,
                         (emb_box_w - 0.1) / 14, 0.10])
    ax_b.imshow(concerned_emb[None, :], aspect="auto", cmap="RdBu_r",
                vmin=-vmax_emb, vmax=vmax_emb)
    ax_b.set_xticks([]); ax_b.set_yticks([])
    for s in ["top", "bottom", "left", "right"]:
        ax_b.spines[s].set_visible(False)
    ax.text(emb_box_x + emb_box_w / 2, emb_box_y + 0.18,
            "different sentences -> different vectors",
            ha="center", va="center", fontsize=7, color=BODY_DARK, style="italic")

    # Arrow 3
    _arrow(ax, (7.6, 3.2), (8.0, 3.2))

    # 4. Additive attention box (trained)
    _add_box(ax, (8.05, 2.3), (1.4, 1.6),
             "Additive\nattention\n~99k params",
             TRAINED_FILL, "white", fontsize=10)
    ax.text(8.75, 2.05, "(trained)", color=ACCENT_COLOR,
            fontsize=8, style="italic", ha="center")

    # Arrow 4
    _arrow(ax, (9.5, 3.2), (9.85, 3.2))

    # 5. Attention weights bar chart (real, from bundled model)
    wt_x, wt_y, wt_w, wt_h = 9.9, 1.85, 2.5, 2.7
    bg = FancyBboxPatch((wt_x, wt_y), wt_w, wt_h,
                        boxstyle="round,pad=0.03,rounding_size=0.08",
                        linewidth=1.0, edgecolor=ACCENT_COLOR, facecolor="white")
    ax.add_patch(bg)
    ax.text(wt_x + wt_w / 2, wt_y + wt_h - 0.18,
            "Attention weight per sentence",
            ha="center", va="top", fontsize=9, color=ACCENT_COLOR, fontweight="bold")
    # Render bar chart in a sub-axes (top-attended sentence highlighted)
    n_show = min(15, len(attn_weights))
    weights_show = attn_weights[:n_show]
    top_idx = int(np.argmax(weights_show))
    ax_w = fig.add_axes([(wt_x + 0.15) / 14, (wt_y + 0.30) / 5.4,
                         (wt_w - 0.30) / 14, (wt_h - 0.90) / 5.4])
    bars = ax_w.barh(np.arange(n_show), weights_show,
                     color=TRAINED_FILL, edgecolor="white", height=0.7)
    bars[top_idx].set_color(ACCENT_COLOR)
    ax_w.invert_yaxis()
    ax_w.set_yticks([])
    ax_w.tick_params(axis="x", labelsize=7)
    ax_w.set_xlabel("weight", fontsize=8)
    for s in ["top", "right"]:
        ax_w.spines[s].set_visible(False)
    ax_w.set_xlim(0, max(weights_show.max() * 1.15, 0.1))

    # Arrow 5
    _arrow(ax, (12.45, 3.2), (12.8, 3.2))

    # 6. Linear head (trained)
    _add_box(ax, (12.85, 2.6), (1.0, 0.9),
             "Linear\nhead",
             TRAINED_FILL, "white", fontsize=10)
    ax.text(13.35, 2.4, "y = w*d + b", fontsize=8,
            color=BODY_DARK, style="italic", ha="center")
    ax.text(13.35, 2.2, "(trained)", fontsize=7,
            color=ACCENT_COLOR, style="italic", ha="center")

    # Down arrow from linear head to output box
    _arrow(ax, (13.35, 2.55), (13.35, 1.4))

    # 7. Output (10-day VIX change)
    _add_box(ax, (12.6, 0.45), (1.5, 0.9),
             "predicted\n10-day\nVIX change",
             OUTPUT_FILL, "white", fontsize=9)

    # Legend at the very bottom-left
    leg_y = 0.45
    ax.add_patch(FancyBboxPatch((0.15, leg_y - 0.10), 0.30, 0.25,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=FROZEN_FILL, edgecolor=ACCENT_COLOR))
    ax.text(0.55, leg_y + 0.02, "frozen (no gradients)",
            fontsize=8, va="center", color=BODY_DARK)
    ax.add_patch(FancyBboxPatch((2.55, leg_y - 0.10), 0.30, 0.25,
                                boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=TRAINED_FILL, edgecolor=ACCENT_COLOR))
    ax.text(2.95, leg_y + 0.02, "trained (~100k params)",
            fontsize=8, va="center", color=BODY_DARK)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
