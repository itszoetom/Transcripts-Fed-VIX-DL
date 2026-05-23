"""SentenceAttentionModel: additive attention aggregator + linear regression head.

The model takes per-sentence FinBERT embeddings for one document, learns a
weighted sum over sentences (the "document vector"), and maps that to a
scalar VIX-change prediction.

Mathematical form (additive / Bahdanau-style attention):

    u_i    = tanh(W h_i + b)            # (attn_dim,)   per-sentence projection
    s_i    = v^T u_i                    # scalar score per sentence
    s_i  <- s_i  if mask_i = 1  else  -inf  (pad sentences ignored by softmax)
    alpha  = softmax(s)                 # (N,)          attention weights
    doc    = sum_i alpha_i * h_i        # (768,)        aggregated document vector
    y      = w^T doc + b_reg            # scalar        VIX change prediction

Why additive (not dot-product, not multi-head self-attention):
    Additive attention is the classical Hierarchical Attention Network
    formulation (Yang et al., 2016) and is the standard reference for
    "attend over sentence embeddings to make one document vector." Compared
    to dot-product attention with a learned query, additive attention has a
    learned linear transform of each sentence before scoring (W, b), which
    helps when the underlying sentence embeddings (frozen FinBERT) are not
    optimized for the regression target. Compared to a self-attention layer
    with a [DOC] token, additive attention adds ~100k parameters instead of
    ~2.4M, critical when training on ~250 pre-2017 documents.

The forward pass returns *both* predictions and attention weights so the
weights can be inspected and visualized post-hoc (the project spec calls for
this explicitly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn as nn


@dataclass
class AttentionConfig:
    """Hyperparameters of the SentenceAttentionModel.

    Attributes:
        embed_dim: Dimensionality of the per-sentence input embeddings (768
                   for FinBERT-base).
        attn_dim:  Dimensionality of the additive-attention hidden projection.
        dropout:   Dropout applied to the document vector before the regression
                   head. Helps regularize the small head against the tiny
                   dataset.
    """

    embed_dim: int = 768
    attn_dim: int = 128
    dropout: float = 0.1


class ModelOutput(NamedTuple):
    """Forward-pass output bundle.

    Attributes:
        prediction:         (B,) predicted 3-day VIX change.
        attention_weights:  (B, N) attention weight per sentence (softmaxed,
                            padded positions == 0). Returned so the model is
                            inspectable / visualizable after training.
        doc_vector:         (B, embed_dim) the aggregated document vector,
                            useful for downstream nearest-neighbor /
                            clustering analysis.
    """

    prediction: torch.Tensor
    attention_weights: torch.Tensor
    doc_vector: torch.Tensor


class SentenceAttentionModel(nn.Module):
    """Frozen-encoder + additive-attention aggregator + linear regression head.

    The encoder itself is *not* held inside this module, embeddings are
    pre-computed (see scripts/precompute_embeddings.py) and passed in via the
    forward pass. This keeps the only trained parameters here (attention W/b/v
    and the regression head w/b_reg), and keeps the model object cheap to
    save/load.

    Usage:
        model = SentenceAttentionModel(AttentionConfig())
        out = model(embeddings, mask)   # embeddings: (B, N, 768), mask: (B, N)
        loss = mse(out.prediction, target)
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        super().__init__()
        self.config = config or AttentionConfig()

        # Additive attention parameters.
        # W : (embed_dim -> attn_dim) and b : (attn_dim,)
        self.attn_proj = nn.Linear(self.config.embed_dim, self.config.attn_dim, bias=True)
        # v : (attn_dim,), implemented as a Linear(attn_dim -> 1) without bias.
        self.attn_query = nn.Linear(self.config.attn_dim, 1, bias=False)

        # Light dropout on the doc vector. Useful with small data; turn off via
        # config.dropout=0.0 if undesired.
        self.dropout = nn.Dropout(self.config.dropout)

        # Linear regression head. One output: 3-day VIX change.
        self.head = nn.Linear(self.config.embed_dim, 1)

        # Sensible init for the regression head, small Gaussian works fine.
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> ModelOutput:
        """Aggregate sentence embeddings into one prediction per document.

        Args:
            embeddings: (B, N, embed_dim), per-sentence vectors. Pad positions
                        may contain arbitrary values; they're masked out by
                        `mask` before softmax.
            mask:       (B, N) 0/1 tensor; 1 = real sentence, 0 = pad.

        Returns:
            ModelOutput(prediction, attention_weights, doc_vector).
        """
        # u_i = tanh(W h_i + b) -> (B, N, attn_dim)
        u = torch.tanh(self.attn_proj(embeddings))
        # s_i = v^T u_i -> (B, N)
        scores = self.attn_query(u).squeeze(-1)

        # Masked softmax: set scores on padded positions to -inf so they
        # contribute zero probability mass. Using a large negative number
        # rather than -float('inf') is numerically safer with fp16.
        masked_scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(masked_scores, dim=-1)  # (B, N)

        # Zero out attention on padded sentences entirely (softmax already
        # gives ~0 there, but this guarantees exact 0 for downstream viz).
        attn = attn * mask.float()

        # Document vector: weighted sum over sentences.  (B, N, 1) * (B, N, E) -> (B, E)
        doc_vec = (attn.unsqueeze(-1) * embeddings).sum(dim=1)
        doc_vec = self.dropout(doc_vec)

        # Linear regression head -> scalar per document.
        pred = self.head(doc_vec).squeeze(-1)  # (B,)

        return ModelOutput(prediction=pred, attention_weights=attn, doc_vector=doc_vec)

    def trainable_parameters(self) -> list[nn.Parameter]:
        """Return only parameters that require grad, by design, all of them."""
        return [p for p in self.parameters() if p.requires_grad]
