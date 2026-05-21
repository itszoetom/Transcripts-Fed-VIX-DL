"""Frozen FinBERT sentence encoder.

Wraps `yiyanghkust/finbert-pretrain` with a mask-weighted mean-pooling head to
produce one 768-d vector per sentence.

Why fully frozen:
    The encoder weights are *never* updated during training. Two reasons:

    1. With ~250 training documents (pre-2017), fine-tuning a 110M-parameter
       BERT is hopelessly under-data — it would overfit catastrophically and
       lose the financial-domain pretraining signal we're paying for by
       picking FinBERT in the first place.
    2. Because the encoder is frozen and deterministic (no dropout active in
       eval mode), every sentence's embedding is fully determined by the
       sentence text alone. That means we can pre-compute embeddings once
       and train the attention+head on cached vectors — orders of magnitude
       faster than re-encoding each epoch.

Why mask-weighted mean pooling instead of [CLS]:
    `yiyanghkust/finbert-pretrain` is the masked-language-model pretrained
    checkpoint; it has *not* been fine-tuned on a sentence classification
    objective, so its [CLS] token has no meaningful sentence-level semantics
    (Reimers & Gurevych, "Sentence-BERT", 2019, Section 3.3). Mean pooling
    over the non-padding tokens gives a substantially better sentence
    representation in this setting.

This module is the only place that imports `transformers`.
"""

from __future__ import annotations

import logging
from typing import Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Default checkpoint name. Configurable via FrozenFinBERTEncoder(...) for
# robustness — e.g., swapping in the sentiment-fine-tuned `yiyanghkust/finbert-tone`
# for ablation experiments would only require changing this string.
DEFAULT_FINBERT_CHECKPOINT = "yiyanghkust/finbert-pretrain"

# Tokenizer source. The `yiyanghkust/finbert-pretrain` HuggingFace repo ships
# only the model weights and config — every tokenizer file (tokenizer_config.json,
# tokenizer.json, vocab.txt, special_tokens_map.json) is missing from the repo,
# so AutoTokenizer.from_pretrained(checkpoint) fails with a confusing
# "you need sentencepiece" error.
#
# FinBERT was initialized from BERT-base-uncased and reuses that exact 30522-token
# WordPiece vocabulary (per Yang et al. 2020, "FinBERT: A Pretrained Language
# Model for Financial Communications"). So we load the tokenizer from
# `bert-base-uncased` directly — same vocab, no sentencepiece needed.
DEFAULT_TOKENIZER_CHECKPOINT = "bert-base-uncased"

# Per-sentence max tokens fed to FinBERT. FinBERT inherits BERT's 512 hard
# limit. Fed-document sentences are virtually always <128 tokens; we set 256
# as a safety margin while keeping per-batch memory predictable.
MAX_SENTENCE_TOKENS = 256


class FrozenFinBERTEncoder(nn.Module):
    """Frozen FinBERT with attention-mask-weighted mean pooling.

    Usage:
        encoder = FrozenFinBERTEncoder(device="cuda")
        emb = encoder.encode_sentences(["The economy expanded.", "Risks remain."])
        # emb is a (2, 768) tensor on the encoder's device.

    The module exposes both an `encode_sentences` convenience method (takes
    list[str], returns tensor) and a standard `forward(input_ids, attention_mask)`
    suitable for batched torch usage.
    """

    def __init__(
        self,
        checkpoint: str = DEFAULT_FINBERT_CHECKPOINT,
        device: str | torch.device = "cpu",
        max_length: int = MAX_SENTENCE_TOKENS,
        tokenizer_checkpoint: str = DEFAULT_TOKENIZER_CHECKPOINT,
    ) -> None:
        super().__init__()
        # Local import keeps `import transcripts_fed_vix.models` cheap for
        # callers that just want config-level introspection.
        from transformers import AutoTokenizer, BertModel

        self.checkpoint = checkpoint
        self.tokenizer_checkpoint = tokenizer_checkpoint
        self.max_length = max_length

        # Tokenizer loaded from a separate checkpoint because finbert-pretrain
        # ships no tokenizer files; see DEFAULT_TOKENIZER_CHECKPOINT comment.
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)

        # We load via BertModel rather than AutoModel because finbert-pretrain's
        # config.json predates the `model_type` field that newer transformers
        # AutoConfig requires. FinBERT IS BERT-base architecture (initialized
        # from bert-base-uncased per Yang et al. 2020), so BertModel is the
        # correct, type-safe choice — it doesn't need model_type to dispatch.
        backbone = BertModel.from_pretrained(checkpoint)

        # Freeze every parameter — belt-and-suspenders: also set training=False
        # in forward(). Doing both means an accidental `model.train()` upstream
        # cannot start updating these weights.
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone

        self.to(device)
        self._device = torch.device(device)

    # ----- Standard nn.Module forward --------------------------------------

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a batch of tokenized sentences to 768-d vectors.

        Args:
            input_ids:      (B, T) long tensor of token ids.
            attention_mask: (B, T) 0/1 mask; 1 = real token, 0 = pad.

        Returns:
            (B, 768) float tensor of mean-pooled sentence embeddings.
        """
        # Force eval mode even if the parent module was set to train().
        self.backbone.eval()
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(out.last_hidden_state, attention_mask)

    @staticmethod
    def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attention-mask-weighted mean over the token dimension.

        hidden: (B, T, H) — token-level encoder outputs.
        mask:   (B, T)    — 0/1 padding mask.

        Returns (B, H). Pad tokens are zeroed out before averaging; the
        denominator is the count of real tokens per row, clamped to >=1 to
        avoid div-by-zero on (unexpected) all-pad rows.
        """
        mask_f = mask.unsqueeze(-1).float()  # (B, T, 1)
        summed = (hidden * mask_f).sum(dim=1)  # (B, H)
        counts = mask_f.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / counts

    # ----- Convenience for callers passing raw sentence strings -----------

    @torch.no_grad()
    def encode_sentences(
        self,
        sentences: Iterable[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Tokenize and encode a list of sentences in mini-batches.

        Args:
            sentences:  Iterable of sentence strings.
            batch_size: Mini-batch size for tokenization + forward pass.

        Returns:
            A (N, 768) tensor on the encoder's device.
        """
        sentences = list(sentences)
        if not sentences:
            return torch.empty((0, self.hidden_size), device=self._device)
        outs: list[torch.Tensor] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self._device)
            attn = enc["attention_mask"].to(self._device)
            outs.append(self.forward(input_ids, attn))
        return torch.cat(outs, dim=0)

    @property
    def hidden_size(self) -> int:
        """Embedding dimension (768 for FinBERT-base)."""
        return int(self.backbone.config.hidden_size)
