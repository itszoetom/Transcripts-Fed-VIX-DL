"""Render attention weights for ONE document.

For the Model slide. Shows the learned attention weights assigned to each
sentence, side-by-side with the sentence text, demonstrating which sentences
the model attends to. Uses the bundled example model + embeddings so no
Talapas data is required.

Run from the repo root:
    python scripts/make_attention_weights_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import AttentionConfig

OUTPUT = Path("outputs/figures/attention_weights_one_doc.png")
ACCENT_COLOR = "#1F4E79"
BAR_COLOR = "#4C78A8"


def main() -> None:
    docs = json.loads(Path("data/example/example_documents.json").read_text())
    emb_dict = torch.load("data/example/example_embeddings.pt",
                          map_location="cpu", weights_only=True)

    # Load the bundled example model.
    model = SentenceAttentionModel(AttentionConfig())
    model.load_state_dict(torch.load("data/example/example_model.pt",
                                     map_location="cpu", weights_only=True))
    model.eval()

    # Pick an example document with informative sentences (HH 1997-07-22 has
    # clear policy-rationale language).
    doc_id = "hh_19970722"
    chosen = next(d for d in docs if d["doc_id"] == doc_id)
    sentences = chosen["sentences"]
    embeddings = emb_dict[doc_id]  # (n_sents, 768)
    n_sents = embeddings.shape[0]

    # Forward pass on this one document (batch of 1).
    emb_batch = embeddings.unsqueeze(0)  # (1, n_sents, 768)
    mask = torch.ones(1, n_sents, dtype=torch.float32)
    with torch.no_grad():
        out = model(emb_batch, mask)
    weights = out.attention_weights[0].numpy()  # (n_sents,)
    prediction = float(out.prediction[0])
    target = float(chosen["target"])

    # Truncate very long sentences for readability in the figure.
    def _short(s: str, max_len: int = 110) -> str:
        return s if len(s) <= max_len else s[: max_len - 1] + "..."

    short_sents = [_short(s) for s in sentences]

    fig, ax = plt.subplots(figsize=(13, 0.42 * n_sents + 1.2))
    y_pos = np.arange(n_sents)
    bars = ax.barh(y_pos, weights, color=BAR_COLOR, edgecolor="white",
                   alpha=0.9, height=0.7)

    # Highlight the top-1 attended sentence in dark navy.
    top_idx = int(np.argmax(weights))
    bars[top_idx].set_color(ACCENT_COLOR)
    bars[top_idx].set_alpha(1.0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_sents, fontsize=7, family="monospace")
    ax.invert_yaxis()  # sentence 1 at top
    ax.set_xlabel("Attention weight", fontsize=10)
    ax.set_xlim(0, max(weights.max() * 1.15, 0.1))
    ax.grid(True, axis="x", alpha=0.3)

    title = (
        f"Attention weights for one document  "
        f"({doc_id} - Humphrey-Hawkins testimony, 1997-07-22)\n"
        f"True 10-day VIX change = {target:+.2f}    "
        f"Model prediction = {prediction:+.2f}"
    )
    ax.set_title(title, fontsize=11, color=ACCENT_COLOR, loc="left", pad=8)

    # Annotate top-1 sentence
    ax.text(
        weights[top_idx] * 1.01, top_idx,
        f"  <- highest attention",
        fontsize=8, color=ACCENT_COLOR, va="center", ha="left",
        fontweight="bold",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
