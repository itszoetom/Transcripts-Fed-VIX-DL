"""Export a small, repo-committable demo dataset.

Picks ~10 documents that span the temporal and source structure of the full
corpus (oldest, middle thirds, newest, with at least 2 from each source),
and writes:

    data/example/example_documents.json   — sentences + target + dates
    data/example/example_embeddings.pt    — pre-computed FinBERT embeddings
                                            for the example docs (first
                                            20 sentences each, to keep the
                                            file under 1 MB)
    data/example/example_model.pt         — copy of outputs/model.pt so the
                                            notebook can show attention
                                            without retraining

The notebook `notebooks/data_demo.ipynb` loads these and demonstrates the
DataLoader API end-to-end without needing FRED, Talapas, or a re-scrape.

Run this once on Talapas after the main pipeline has produced
`data/processed/documents.parquet`, `data/processed/sentence_embeddings.pt`,
and `outputs/model.pt`. Commit the resulting `data/example/*` files to the
repo.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("export_examples")

# Number of example documents and how many sentences per doc to keep.
# The 20-sentence cap is purely to keep the embeddings file small (~600 KB
# instead of 2.5 MB at 80 sentences); the notebook is for demonstration only.
N_EXAMPLES = 10
SENTENCES_PER_EXAMPLE = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    processed_dir = Path(cfg["data"]["processed_dir"])
    outputs_dir = Path(cfg["outputs"]["dir"])
    example_dir = Path("data/example")
    example_dir.mkdir(parents=True, exist_ok=True)

    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    model_path = outputs_dir / cfg["outputs"]["model_file"]

    df = pd.read_parquet(documents_path).sort_values("release_date").reset_index(drop=True)
    all_emb = torch.load(embeddings_path, map_location="cpu", weights_only=True)

    # Pick representative docs: 2 oldest, 2 from mid, 2 from late, 2 newest,
    # plus 2 HH testimony docs picked deterministically. Deduplicate by doc_id.
    n = len(df)
    third = n // 3
    selected_idx = list({
        *df.index[:2].tolist(),
        *df.index[third : third + 2].tolist(),
        *df.index[2 * third : 2 * third + 2].tolist(),
        *df.index[-2:].tolist(),
        *df[df["source"] == "humphrey_hawkins"].head(2).index.tolist(),
    })
    selected_idx = sorted(selected_idx)[:N_EXAMPLES]
    picks = df.loc[selected_idx].reset_index(drop=True)

    # Build the JSON record list. Keep only first SENTENCES_PER_EXAMPLE
    # sentences per doc so the file is small and the notebook stays readable.
    records: list[dict] = []
    example_embeddings: dict[str, torch.Tensor] = {}
    for _, row in picks.iterrows():
        doc_id = row["doc_id"]
        sents = list(row["sentences"])[:SENTENCES_PER_EXAMPLE]
        records.append({
            "doc_id": doc_id,
            "source": row["source"],
            "release_date": pd.Timestamp(row["release_date"]).date().isoformat(),
            "aligned_trading_date": pd.Timestamp(row["aligned_trading_date"]).date().isoformat(),
            "vix_t": float(row["vix_t"]),
            "vix_t_plus_3": float(row["vix_t_plus_3"]),
            "target": float(row["target"]),
            "sentences": sents,
        })
        # Slice the cached embeddings to the same sentence count.
        if doc_id in all_emb:
            example_embeddings[doc_id] = all_emb[doc_id][:SENTENCES_PER_EXAMPLE].clone()
        else:
            logger.warning("no cached embeddings for %s — skipping", doc_id)

    (example_dir / "example_documents.json").write_text(json.dumps(records, indent=2))
    torch.save(example_embeddings, example_dir / "example_embeddings.pt")
    if model_path.exists():
        shutil.copy(model_path, example_dir / "example_model.pt")
        logger.info("copied trained model to data/example/example_model.pt")

    logger.info(
        "wrote %d example docs (%d sentences each) + %d embedding entries to %s",
        len(records), SENTENCES_PER_EXAMPLE, len(example_embeddings), example_dir,
    )


if __name__ == "__main__":
    main()
