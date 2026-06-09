"""Pre-compute FinBERT sentence embeddings for the entire processed dataset.

Because the FinBERT encoder is frozen and deterministic, every sentence's
embedding is fixed by the sentence's text. We compute the embeddings ONCE
here, save them to disk, and reuse them every epoch in the training loop.

This saves 2 to 3 orders of magnitude of compute over re-running BERT inside
each training step, and it produces no information leakage: the encoder is
the same model with the same weights regardless of when it runs.

Usage:
    python -m attention_fed_vix.scripts.precompute_embeddings --config configs/default.yaml

Outputs:
    data/processed/sentence_embeddings.pt
        dict mapping doc_id -> torch.FloatTensor of shape (N_sents, 768).

If the output file already exists, this script no-ops by default; pass
--force to recompute.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from attention_fed_vix.models import FrozenFinBERTEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("precompute_embeddings")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--force", action="store_true",
                        help="Recompute even if the output file already exists.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    processed_dir = Path(cfg["data"]["processed_dir"])
    documents_path = processed_dir / cfg["data"]["documents_file"]
    out_path = processed_dir / cfg["data"]["embeddings_file"]

    if out_path.exists() and not args.force:
        logger.info("embeddings already exist at %s, skipping (pass --force to recompute)", out_path)
        return

    documents = pd.read_parquet(documents_path)
    logger.info("loaded %d documents from %s", len(documents), documents_path)

    # Honor the device override in config; fall back to CUDA if available, else CPU.
    device_str = cfg["model"].get("device", "auto")
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("using device: %s", device_str)

    encoder = FrozenFinBERTEncoder(
        checkpoint=cfg["model"]["encoder_checkpoint"],
        device=device_str,
        max_length=cfg["model"]["max_sentence_tokens"],
    )

    # Compute embeddings doc-by-doc. We could batch across documents, but per-doc
    # batching keeps memory simple and predictable, and the cost is dominated by
    # the BERT forward pass either way.
    embeddings: dict[str, torch.Tensor] = {}
    for _, row in tqdm(documents.iterrows(), total=len(documents), desc="encoding"):
        sents = list(row["sentences"])
        emb = encoder.encode_sentences(sents, batch_size=cfg["model"]["sentence_encode_batch_size"])
        # Move to CPU + float32 before saving, keeps the file portable and
        # small enough to load on a CPU-only machine for analysis.
        embeddings[row["doc_id"]] = emb.cpu().to(torch.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    logger.info(
        "wrote embeddings for %d documents to %s (total ~%.1f MB)",
        len(embeddings),
        out_path,
        out_path.stat().st_size / 1e6,
    )


if __name__ == "__main__":
    main()
