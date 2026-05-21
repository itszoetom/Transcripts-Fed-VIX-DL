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

The module also exposes `get_example_dataloader()` for the project's data demo
notebook — it loads a small bundled set of ~10 documents from data/example/
and returns a DataLoader that anyone can run after `pip install -e .` without
needing access to Talapas, FRED, or the full scrape.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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


# ---------------------------------------------------------------------------
# Example/demo dataloader for the milestone notebook
# ---------------------------------------------------------------------------


def _default_example_dir() -> Path:
    """Locate `data/example/` relative to the installed package.

    The example files are committed to the repo, so they're always findable
    relative to the project root (two levels up from this file's package
    install location). Falls back to CWD / data/example for editable installs.
    """
    # data/example lives at <repo-root>/data/example
    repo_root = Path(__file__).resolve().parents[3]
    candidate = repo_root / "data" / "example"
    if candidate.exists():
        return candidate
    return Path.cwd() / "data" / "example"


def load_example_documents(example_dir: Path | None = None) -> pd.DataFrame:
    """Load the bundled example documents JSON as a DataFrame.

    Schema matches the full processed parquet (doc_id, source, release_date,
    aligned_trading_date, vix_t, vix_t_plus_3, target, sentences), so any
    code that works on the full dataset works on the example.
    """
    example_dir = example_dir or _default_example_dir()
    json_path = example_dir / "example_documents.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Example data missing: {json_path}. "
            "Run scripts/export_examples.py on the cluster (or wherever the "
            "full processed parquet lives) to generate the bundled examples."
        )
    records = json.loads(json_path.read_text())
    df = pd.DataFrame(records)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["aligned_trading_date"] = pd.to_datetime(df["aligned_trading_date"])
    return df


def get_example_dataloader(
    batch_size: int = 4,
    example_dir: Path | None = None,
) -> DataLoader:
    """Return a DataLoader over the ~10 bundled example documents.

    Mirrors the `get_data_loaders(name="example")` pattern from the project
    milestone example repo. Use this in the data demo notebook to show what
    a batch of inputs looks like end-to-end, without depending on the full
    scraped dataset.

    Args:
        batch_size:  Per-batch document count.
        example_dir: Override directory containing the example files;
                     defaults to `<repo-root>/data/example/`.

    Returns:
        A torch DataLoader yielding dicts with keys:
            embeddings: (B, N_max, 768) — frozen-FinBERT sentence embeddings
            mask:       (B, N_max)      — 0/1 mask, 1 = real sentence
            target:     (B,)            — 3-day forward VIX change
            doc_ids:    list[str]       — document identifiers
            release_dates: list[Timestamp]
    """
    example_dir = example_dir or _default_example_dir()
    df = load_example_documents(example_dir)
    emb_path = example_dir / "example_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Example embeddings missing: {emb_path}. "
            "Run scripts/export_examples.py to generate."
        )
    ds = EmbeddingDocDataset(df, embeddings_path=emb_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_padded)
