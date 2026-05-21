"""Build the processed dataset (scrape -> segment -> VIX-align -> parquet).

Thin CLI wrapper around `transcripts_fed_vix.data.build.build_processed_dataset`.
Idempotent: re-runs are no-ops if the processed parquet already exists.

Usage:
    python scripts/build_data.py --config configs/default.yaml
    python scripts/build_data.py --config configs/default.yaml --force
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from transcripts_fed_vix.data.build import build_processed_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--force", action="store_true",
                        help="Rebuild the parquet even if it already exists.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    df = build_processed_dataset(
        raw_dir=Path(cfg["data"]["raw_dir"]),
        processed_dir=Path(cfg["data"]["processed_dir"]),
        processed_filename=cfg["data"]["documents_file"],
        force=args.force,
    )
    print(f"processed dataset: {len(df)} rows | "
          f"date range: {df['release_date'].min().date()} .. {df['release_date'].max().date()}")


if __name__ == "__main__":
    main()
