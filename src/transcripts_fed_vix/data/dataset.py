"""torch.utils.data.Dataset over pre-computed sentence embeddings.

Because FinBERT is frozen, every sentence's embedding is deterministic and can
be computed once, cached, and reused every epoch. That's done by
scripts/precompute_embeddings.py, which writes a dict

    {doc_id: torch.FloatTensor of shape (N_sents, 768)}

to data/processed/sentence_embeddings.pt. This Dataset reads that file plus
the processed parquet (doc_id + target) and yields per-document items. The
custom `collate_padded` collator pads to the per-batch max sentence count and
builds an attention mask, which the model uses to ignore padded positions in
the softmax.

Why pad-to-batch-max instead of pad-to-80:
    Most documents fill all 80 sentences (we truncated at 80), so the
    difference is small — but pad-to-batch-max saves a bit of compute and
    keeps the collator generic in case the cap is later changed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetItem:
    """Container for one document's training input."""

    doc_id: str
    embeddings: torch.Tensor  # (N_sents, 768)
    target: float
    release_date: pd.Timestamp


class EmbeddingDocDataset(Dataset):
    """Pre-computed-embedding dataset for the sentence-attention model.

    Args:
        documents_df:    DataFrame from build_processed_dataset() (or a temporal
                         subset thereof). Must contain doc_id, target,
                         release_date columns. Order is preserved (do NOT
                         shuffle before passing in — temporal order matters).
        embeddings_path: Path to the .pt file containing {doc_id: tensor}.
    """

    def __init__(self, documents_df: pd.DataFrame, embeddings_path: Path) -> None:
        super().__init__()
        # Use map_location='cpu' so loading on a CPU-only machine for tests
        # never silently fails on a CUDA-saved checkpoint.
        self.embeddings: dict[str, torch.Tensor] = torch.load(
            embeddings_path, map_location="cpu", weights_only=True
        )
        # Defensive: filter the dataframe down to documents we actually have
        # embeddings for.
        missing = set(documents_df["doc_id"]) - set(self.embeddings.keys())
        if missing:
            logger.warning(
                "embeddings missing for %d docs; filtering them out: %s",
                len(missing),
                sorted(missing)[:5],
            )
            documents_df = documents_df[documents_df["doc_id"].isin(self.embeddings.keys())]
        self.df = documents_df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> DatasetItem:
        row = self.df.iloc[idx]
        doc_id = row["doc_id"]
        return DatasetItem(
            doc_id=doc_id,
            embeddings=self.embeddings[doc_id],
            target=float(row["target"]),
            release_date=row["release_date"],
        )


def collate_padded(batch: Sequence[DatasetItem]) -> dict[str, torch.Tensor | list]:
    """Pad a batch of variable-length sentence-embedding tensors.

    Returns:
        dict with:
            embeddings: (B, N_max, 768) float tensor (zero-padded)
            mask:       (B, N_max)      0/1 float tensor (1 = real sentence)
            target:     (B,)            float tensor
            doc_ids:    list[str]       per-row document ids
            release_dates: list[Timestamp]
    """
    n_max = max(item.embeddings.shape[0] for item in batch)
    embed_dim = batch[0].embeddings.shape[1]
    B = len(batch)

    embeddings = torch.zeros((B, n_max, embed_dim), dtype=torch.float32)
    mask = torch.zeros((B, n_max), dtype=torch.float32)
    target = torch.zeros((B,), dtype=torch.float32)

    for i, item in enumerate(batch):
        n = item.embeddings.shape[0]
        embeddings[i, :n] = item.embeddings.to(torch.float32)
        mask[i, :n] = 1.0
        target[i] = item.target

    return {
        "embeddings": embeddings,
        "mask": mask,
        "target": target,
        "doc_ids": [item.doc_id for item in batch],
        "release_dates": [item.release_date for item in batch],
    }
