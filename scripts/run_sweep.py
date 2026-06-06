"""Run a hyperparameter sweep over the sentence-attention model.

Reads `configs/sweep.yaml`, enumerates the Cartesian product of the `grid`
section, and trains one model per combination. After all runs finish, the
winning configuration (lowest val MSE by default) is promoted into `outputs/`
as the canonical "current best" model. Downstream scripts (run_regime_analysis.py,
run_baselines.py, make_plots.py) then operate on that winner.

Per-run artifacts live under `outputs/sweep/<run_name>/`:
    config.yaml           the merged config used for this run
    model.pt              best checkpoint by val MSE (early-stopped)
    train_metrics.json    per-epoch train/val MSE, R^2, Pearson r
    final_report.json     per-segment metrics on val + 3 test regimes
    eval_summary.json     compact one-row summary used to pick the winner

After the sweep, `outputs/sweep/sweep_summary.json` collects every run's
hyperparameters + val/regime metrics + status, and `outputs/sweep/sweep_table.csv`
provides the same information for spreadsheet inspection.

Usage:
    python scripts/run_sweep.py --config configs/sweep.yaml
    python scripts/run_sweep.py --config configs/sweep.yaml --skip-completed

The --skip-completed flag is the safety mechanism for re-running on Talapas
after an interrupted sweep: any run whose eval_summary.json already exists is
skipped, so the sweep picks up where it left off.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import shutil
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from transcripts_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import attention_config_from_dict
from transcripts_fed_vix.training import train
from transcripts_fed_vix.training.eval import regression_metrics
from transcripts_fed_vix.training.loop import TrainConfig
from transcripts_fed_vix.utils import set_seed, make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_sweep")


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------


def _grid_points(grid: dict[str, list]) -> list[dict[str, Any]]:
    """Return every combination of the grid as a list of {name: value} dicts."""
    names = sorted(grid.keys())
    values = [grid[n] for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def _run_name(point: dict[str, Any]) -> str:
    """Compact, filesystem-safe identifier for a grid point."""
    parts = []
    for k in sorted(point.keys()):
        v = point[k]
        if isinstance(v, float):
            v_str = f"{v:g}"
        else:
            v_str = str(v)
        parts.append(f"{k}={v_str}")
    return "_".join(parts)


def _apply_overrides(base: dict, point: dict[str, Any], fixed: dict | None) -> dict:
    """Merge a grid point onto the base config; route known keys to their sections.

    The grid uses flat names (e.g. `learning_rate`) but the YAML is structured
    (`training.learning_rate`). We route each grid key into the right section
    based on a small known mapping; unknown keys land at top-level.
    """
    cfg = copy.deepcopy(base)
    if fixed:
        cfg = _deep_merge(cfg, fixed)

    routing = {
        "learning_rate": ("training", "learning_rate"),
        "weight_decay": ("training", "weight_decay"),
        "batch_size": ("training", "batch_size"),
        "epochs": ("training", "epochs"),
        "warmup_fraction": ("training", "warmup_fraction"),
        "grad_clip_norm": ("training", "grad_clip_norm"),
        "early_stop_patience": ("training", "early_stop_patience"),
        "attn_dim": ("model", "attn_dim"),
        "dropout": ("model", "dropout"),
    }
    for key, value in point.items():
        if key in routing:
            section, name = routing[key]
            cfg.setdefault(section, {})[name] = value
        else:
            cfg[key] = value
    return cfg


def _deep_merge(dst: dict, src: dict) -> dict:
    """Merge `src` into `dst` recursively; values in src win."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


# ---------------------------------------------------------------------------
# Per-run training + evaluation
# ---------------------------------------------------------------------------


def _build_loader(df: pd.DataFrame, embeddings_path: Path, batch_size: int) -> DataLoader:
    """Non-shuffled DataLoader; temporal order is preserved by spec."""
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_padded)


def _predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Run inference, return (predictions, targets) arrays."""
    model.eval()
    preds: list[float] = []
    tgts: list[float] = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            out = model(embeddings, mask)
            preds.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
    return np.asarray(preds), np.asarray(tgts)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _train_one_run(
    cfg: dict,
    point: dict[str, Any],
    run_dir: Path,
    documents: pd.DataFrame,
    embeddings_path: Path,
) -> dict:
    """Train one configuration, save artifacts under run_dir, return summary."""
    set_seed(int(cfg["seed"]))
    device = _resolve_device(cfg["model"].get("device", "auto"))

    splits = make_temporal_splits(
        documents,
        dates=SplitDates(
            train_end=date.fromisoformat(cfg["splits"]["train_end_date"]),
            regime2_start=date.fromisoformat(cfg["splits"]["regime2_start_date"]),
            regime3_start=date.fromisoformat(cfg["splits"]["regime3_start_date"]),
            val_fraction=float(cfg["splits"]["val_fraction"]),
        ),
    )

    train_cfg = TrainConfig(
        learning_rate=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        batch_size=int(cfg["training"]["batch_size"]),
        epochs=int(cfg["training"]["epochs"]),
        warmup_fraction=float(cfg["training"]["warmup_fraction"]),
        grad_clip_norm=float(cfg["training"]["grad_clip_norm"]),
        early_stop_patience=int(cfg["training"]["early_stop_patience"]),
    )

    model_cfg = attention_config_from_dict(cfg["model"])
    model = SentenceAttentionModel(model_cfg)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    train_loader = _build_loader(splits.train, embeddings_path, train_cfg.batch_size)
    val_loader = _build_loader(splits.val, embeddings_path, train_cfg.batch_size)

    model_save_path = run_dir / "model.pt"
    metrics_save_path = run_dir / "train_metrics.json"
    result = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=device,
        model_save_path=model_save_path,
        metrics_save_path=metrics_save_path,
    )

    # Reload best weights and evaluate on val + each test regime.
    model.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    model.to(device)

    final_report: dict[str, Any] = {
        "grid_point": point,
        "best_epoch": result.best_epoch,
        "best_val_mse": result.best_val_mse,
        "n_trainable_params": int(n_trainable),
    }

    for name, split_df in [
        ("val", splits.val),
        ("regime1_2017_2021", splits.regime1),
        ("regime2_2021_2025", splits.regime2),
        ("regime3_2025_present", splits.regime3),
    ]:
        if len(split_df) == 0:
            continue
        loader = _build_loader(split_df, embeddings_path, train_cfg.batch_size)
        preds, tgts = _predict(model, loader, device)
        rm = regression_metrics(preds, tgts)
        final_report[name] = {"regression": rm.to_dict()}

    (run_dir / "final_report.json").write_text(json.dumps(final_report, indent=2))

    # Compact summary used for sweep ranking + the spreadsheet/CSV.
    summary = {
        "run_name": run_dir.name,
        **point,
        "best_epoch": result.best_epoch,
        "val_mse": float(final_report["val"]["regression"]["mse"]),
        "val_r2": float(final_report["val"]["regression"]["r2"]),
        "val_pearson_r": float(final_report["val"]["regression"]["pearson_r"]),
        "regime1_pearson_r": _safe_get(final_report, "regime1_2017_2021", "regression", "pearson_r"),
        "regime2_pearson_r": _safe_get(final_report, "regime2_2021_2025", "regression", "pearson_r"),
        "regime3_pearson_r": _safe_get(final_report, "regime3_2025_present", "regression", "pearson_r"),
        "n_trainable_params": int(n_trainable),
    }
    (run_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _safe_get(d: dict, *path: str, default: float = float("nan")) -> float:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sweep.yaml"),
                        help="Path to the sweep specification YAML.")
    parser.add_argument("--skip-completed", action="store_true",
                        help="Skip runs whose eval_summary.json already exists. "
                             "Use this to resume a partially-completed sweep.")
    args = parser.parse_args()

    sweep_cfg = yaml.safe_load(args.config.read_text())
    base_path = Path(sweep_cfg["base_config"])
    base_cfg = yaml.safe_load(base_path.read_text())
    sweep_dir = Path(sweep_cfg["sweep_dir"])
    sweep_dir.mkdir(parents=True, exist_ok=True)
    fixed = sweep_cfg.get("fixed_overrides")

    processed_dir = Path(base_cfg["data"]["processed_dir"])
    documents_path = processed_dir / base_cfg["data"]["documents_file"]
    embeddings_path = processed_dir / base_cfg["data"]["embeddings_file"]
    if not documents_path.exists():
        sys.exit(f"Missing processed dataset: {documents_path}. Run build_data.py first.")
    if not embeddings_path.exists():
        sys.exit(f"Missing embeddings: {embeddings_path}. Run precompute_embeddings.py first.")
    documents = pd.read_parquet(documents_path)

    grid = sweep_cfg["grid"]
    points = _grid_points(grid)
    logger.info("sweep: %d grid points; fixed overrides %s", len(points), fixed)

    summaries: list[dict] = []
    for i, point in enumerate(points, 1):
        run_name = _run_name(point)
        run_dir = sweep_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "eval_summary.json"

        if args.skip_completed and summary_path.exists():
            logger.info("(%d/%d) skip already-completed %s", i, len(points), run_name)
            summaries.append(json.loads(summary_path.read_text()))
            continue

        logger.info("(%d/%d) training %s", i, len(points), run_name)
        cfg = _apply_overrides(base_cfg, point, fixed)
        # Save the realized config for traceability.
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        try:
            summary = _train_one_run(cfg, point, run_dir, documents, embeddings_path)
            summary["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 , we record failures and continue
            logger.exception("run %s failed: %s", run_name, exc)
            summary = {"run_name": run_name, **point, "status": "failed", "error": str(exc)}
        summaries.append(summary)

    # Write the full summary + CSV.
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summaries, indent=2))
    _write_csv(sweep_dir / "sweep_table.csv", summaries)

    # Pick the winner.
    selection_metric = sweep_cfg.get("selection_metric", "val_mse")
    eligible = [s for s in summaries if s.get("status") == "ok" and selection_metric in s]
    if not eligible:
        sys.exit(f"No successful runs to select from on metric {selection_metric!r}")
    minimize = selection_metric in ("val_mse",)  # everything else (Pearson, R^2) maximized
    winner = min(eligible, key=lambda s: s[selection_metric]) if minimize \
             else max(eligible, key=lambda s: s[selection_metric])
    logger.info("winner: %s with %s=%.5f", winner["run_name"], selection_metric, winner[selection_metric])

    # Promote winner to outputs/ for downstream scripts.
    outputs_dir = Path(base_cfg["outputs"]["dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    winner_dir = sweep_dir / winner["run_name"]
    shutil.copy(winner_dir / "model.pt", outputs_dir / base_cfg["outputs"]["model_file"])
    shutil.copy(winner_dir / "train_metrics.json", outputs_dir / base_cfg["outputs"]["metrics_file"])
    shutil.copy(winner_dir / "final_report.json", outputs_dir / base_cfg["outputs"]["final_report_file"])
    shutil.copy(winner_dir / "config.yaml", outputs_dir / "winning_config.yaml")
    (outputs_dir / "sweep_winner.json").write_text(json.dumps(winner, indent=2))
    logger.info("promoted winner artifacts to %s", outputs_dir)


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Dump per-run summaries to a CSV for spreadsheet inspection."""
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


if __name__ == "__main__":
    main()
