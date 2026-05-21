"""Project figure generation.

One function per figure; each takes paths/dataframes/models, writes a PNG to
the configured output directory, and returns the saved path. The script
`scripts/make_plots.py` is the CLI orchestrator that calls all of these.

All figures use matplotlib with the default style. The intent is reproducible,
publication-ready figures (300 dpi, tight bbox) without any seaborn dependency
beyond what's already in the project.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required on SLURM nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import EmbeddingDocDataset, collate_padded
from ..models import SentenceAttentionModel
from ..models.attention import AttentionConfig
from ..utils.splits import TemporalSplits

logger = logging.getLogger(__name__)

# All figures save at this resolution. 300 dpi is enough for inclusion in a
# LaTeX paper at half-page width.
DPI = 300


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> Path:
    """Save a figure at the project's standard resolution and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote figure %s", path)
    return path


def _load_model_for_inference(
    model_path: Path,
    cfg: dict,
    device: torch.device,
) -> SentenceAttentionModel:
    """Load the trained SentenceAttentionModel for inference."""
    attn_cfg = AttentionConfig(
        embed_dim=int(cfg["model"]["embed_dim"]),
        attn_dim=int(cfg["model"]["attn_dim"]),
        dropout=float(cfg["model"]["dropout"]),
    )
    model = SentenceAttentionModel(attn_cfg)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def _predict_split(
    model: SentenceAttentionModel,
    df: pd.DataFrame,
    embeddings_path: Path,
    batch_size: int,
    device: torch.device,
    *,
    return_attention: bool = False,
) -> dict:
    """Run inference over a split's documents.

    Returns a dict with predictions/targets/release_dates arrays; if
    return_attention is True, also returns per-doc attention weight lists.
    """
    if len(df) == 0:
        return {
            "predictions": np.array([]),
            "targets": np.array([]),
            "release_dates": [],
            "doc_ids": [],
            "attentions": [],
        }
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=collate_padded)
    preds: list[float] = []
    tgts: list[float] = []
    dates: list[pd.Timestamp] = []
    ids: list[str] = []
    attns: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            embeddings = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            out = model(embeddings, mask)
            preds.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
            dates.extend(batch["release_dates"])
            ids.extend(batch["doc_ids"])
            if return_attention:
                # attention_weights: (B, N_max); strip padding using mask.
                attn_np = out.attention_weights.detach().cpu().numpy()
                mask_np = mask.detach().cpu().numpy()
                for i in range(attn_np.shape[0]):
                    n_real = int(mask_np[i].sum())
                    attns.append(attn_np[i, :n_real])
    return {
        "predictions": np.asarray(preds),
        "targets": np.asarray(tgts),
        "release_dates": dates,
        "doc_ids": ids,
        "attentions": attns,
    }


# ---------------------------------------------------------------------------
# 1. Training curves
# ---------------------------------------------------------------------------


def plot_training_curve(metrics_path: Path, out_path: Path) -> Path:
    """Plot train MSE + val MSE per epoch, with the best-epoch (early-stop) marker."""
    data = json.loads(metrics_path.read_text())
    history = data["history"]
    epochs = [h["epoch"] for h in history]
    train_mse = [h["train_loss_mse"] for h in history]
    val_mse = [h["val_mse"] for h in history]
    best_epoch = data["best_epoch"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, train_mse, marker="o", label="train MSE")
    ax.plot(epochs, val_mse, marker="s", label="val MSE")
    if best_epoch >= 1:
        ax.axvline(best_epoch, color="red", linestyle="--", alpha=0.6,
                   label=f"best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Training curve: train vs validation MSE per epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_training_pearson(metrics_path: Path, out_path: Path) -> Path:
    """Plot val Pearson r per epoch."""
    history = json.loads(metrics_path.read_text())["history"]
    epochs = [h["epoch"] for h in history]
    val_r = [h["val_pearson_r"] for h in history]
    val_r2 = [h["val_r2"] for h in history]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(epochs, val_r, marker="o", label="val Pearson r")
    ax.plot(epochs, val_r2, marker="s", label="val R^2")
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation correlation / R^2 per epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


def plot_lr_schedule(metrics_path: Path, out_path: Path) -> Path:
    """Plot the realized LR schedule per epoch (warmup + post-warmup constant)."""
    history = json.loads(metrics_path.read_text())["history"]
    epochs = [h["epoch"] for h in history]
    lrs = [h["lr"] for h in history]

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(epochs, lrs, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Realized LR schedule (linear warmup → constant)")
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 2. Predicted vs actual scatter (per temporal segment)
# ---------------------------------------------------------------------------


def plot_predicted_vs_actual(
    splits: TemporalSplits,
    embeddings_path: Path,
    model: SentenceAttentionModel,
    out_path: Path,
    *,
    batch_size: int,
    device: torch.device,
) -> Path:
    """One scatter panel per non-train segment with y=x reference line."""
    segments = [
        ("val (pre-2017)", splits.val),
        ("regime1: 2017–2021", splits.regime1),
        ("regime2: 2021–2025", splits.regime2),
        ("regime3: 2025–present", splits.regime3),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.0), sharex=False, sharey=False)
    for ax, (name, df) in zip(axes.flat, segments):
        out = _predict_split(model, df, embeddings_path, batch_size, device)
        preds, tgts = out["predictions"], out["targets"]
        ax.scatter(tgts, preds, alpha=0.6, s=18)
        if len(tgts) > 0:
            lo = float(min(tgts.min(), preds.min()))
            hi = float(max(tgts.max(), preds.max()))
            pad = 0.05 * (hi - lo + 1e-9)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                    color="gray", linestyle="--", alpha=0.6, label="y = x")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True 3-day VIX change")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{name}  (n={len(df)})")
        ax.grid(True, alpha=0.3)
        if len(tgts) > 0:
            ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Predicted vs actual 3-day VIX change, by temporal segment", y=1.01)
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 3. Residuals over time
# ---------------------------------------------------------------------------


def plot_residuals_over_time(
    splits: TemporalSplits,
    embeddings_path: Path,
    model: SentenceAttentionModel,
    out_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    breakpoints: Sequence[date],
) -> Path:
    """Residuals (target - prediction) scattered over release_date for all docs.

    Vertical lines mark the configured Chow breakpoints; horizontal lines
    mark per-segment mean residuals so the reader can see whether the residual
    bias shifts across regimes.
    """
    # Run inference on every segment and concat into a single timeline.
    pieces: list[dict] = []
    labels: list[str] = []
    for name, df in [
        ("train", splits.train),
        ("val", splits.val),
        ("regime1", splits.regime1),
        ("regime2", splits.regime2),
        ("regime3", splits.regime3),
    ]:
        if len(df) == 0:
            continue
        out = _predict_split(model, df, embeddings_path, batch_size, device)
        pieces.append(out)
        labels.append(name)

    all_dates: list[pd.Timestamp] = []
    all_resid: list[float] = []
    segment_means: list[tuple[str, float, float, float]] = []  # (name, start, end, mean)
    for label, p in zip(labels, pieces):
        dates_p = pd.to_datetime(p["release_dates"])
        resid = p["targets"] - p["predictions"]
        all_dates.extend(dates_p.tolist())
        all_resid.extend(resid.tolist())
        if len(resid) > 0:
            segment_means.append((label,
                                  float(pd.Timestamp(min(dates_p)).timestamp()),
                                  float(pd.Timestamp(max(dates_p)).timestamp()),
                                  float(np.mean(resid))))

    df_all = pd.DataFrame({"date": all_dates, "residual": all_resid})
    df_all = df_all.sort_values("date")

    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.scatter(df_all["date"], df_all["residual"], alpha=0.5, s=14)
    for bp in breakpoints:
        ax.axvline(pd.Timestamp(bp), color="red", linestyle="--", alpha=0.6,
                   label=f"breakpoint {bp.isoformat()}")
    ax.set_xlabel("Release date")
    ax.set_ylabel("Residual (target − prediction)")
    ax.set_title("Model residuals over time, with regime-change breakpoints")
    # Dedupe legend entries.
    handles, labels_legend = ax.get_legend_handles_labels()
    seen = set()
    keep = []
    for h, l in zip(handles, labels_legend):
        if l not in seen:
            seen.add(l)
            keep.append((h, l))
    if keep:
        ax.legend(*zip(*keep), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 4. Per-regime comparison bar charts (deep model vs baselines)
# ---------------------------------------------------------------------------


def plot_regression_comparison(
    final_report_path: Path,
    baseline_metrics_path: Path,
    out_path: Path,
) -> Path:
    """Grouped bar chart: deep model vs TF-IDF Ridge on Pearson r and R^2."""
    fr = json.loads(final_report_path.read_text())
    bl = json.loads(baseline_metrics_path.read_text())
    regimes = ["regime1_2017_2021", "regime2_2021_2025", "regime3_2025_present"]

    deep_r = [fr.get(r, {}).get("regression", {}).get("pearson_r", float("nan")) for r in regimes]
    base_r = [bl.get(r, {}).get("tfidf_ridge", {}).get("pearson_r", float("nan")) for r in regimes]
    deep_r2 = [fr.get(r, {}).get("regression", {}).get("r2", float("nan")) for r in regimes]
    base_r2 = [bl.get(r, {}).get("tfidf_ridge", {}).get("r2", float("nan")) for r in regimes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(regimes))
    width = 0.38

    axes[0].bar(x - width / 2, deep_r, width, label="Sentence-attention model")
    axes[0].bar(x + width / 2, base_r, width, label="TF-IDF Ridge baseline")
    axes[0].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[0].set_xticks(x); axes[0].set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axes[0].set_ylabel("Pearson r")
    axes[0].set_title("Pearson r by regime")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width / 2, deep_r2, width, label="Sentence-attention model")
    axes[1].bar(x + width / 2, base_r2, width, label="TF-IDF Ridge baseline")
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    axes[1].set_xticks(x); axes[1].set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axes[1].set_ylabel("R^2")
    axes[1].set_title("R^2 by regime")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Regression performance: model vs baseline, by temporal regime")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_binary_comparison(
    final_report_path: Path,
    baseline_metrics_path: Path,
    out_path: Path,
) -> Path:
    """Grouped bar chart: deep model vs BoW Logistic regression on AUC + F1."""
    fr = json.loads(final_report_path.read_text())
    bl = json.loads(baseline_metrics_path.read_text())
    regimes = ["regime1_2017_2021", "regime2_2021_2025", "regime3_2025_present"]

    deep_auc = [fr.get(r, {}).get("binary", {}).get("auc_roc", float("nan")) for r in regimes]
    base_auc = [bl.get(r, {}).get("bow_logreg", {}).get("auc_roc", float("nan")) for r in regimes]
    deep_f1 = [fr.get(r, {}).get("binary", {}).get("f1", float("nan")) for r in regimes]
    base_f1 = [bl.get(r, {}).get("bow_logreg", {}).get("f1", float("nan")) for r in regimes]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(regimes))
    width = 0.38

    axes[0].bar(x - width / 2, deep_auc, width, label="Sentence-attention model")
    axes[0].bar(x + width / 2, base_auc, width, label="BoW Logistic baseline")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5,
                    label="chance (0.5)")
    axes[0].set_xticks(x); axes[0].set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC-ROC by regime")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].bar(x - width / 2, deep_f1, width, label="Sentence-attention model")
    axes[1].bar(x + width / 2, base_f1, width, label="BoW Logistic baseline")
    axes[1].set_xticks(x); axes[1].set_xticklabels([r.replace("_", "\n") for r in regimes], fontsize=8)
    axes[1].set_ylabel("F1")
    axes[1].set_title("F1 by regime")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Binary classification performance: model vs baseline, by regime")
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 5. Target distribution by regime
# ---------------------------------------------------------------------------


def plot_target_distribution(
    splits: TemporalSplits,
    out_path: Path,
) -> Path:
    """Overlaid histogram of the 3-day VIX-change target, per temporal segment."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-15, 15, 40)
    for name, df in [
        ("train (1993–~2014)", splits.train),
        ("val (~2014–2017)", splits.val),
        ("regime1 (2017–2021)", splits.regime1),
        ("regime2 (2021–2025)", splits.regime2),
        ("regime3 (2025–)", splits.regime3),
    ]:
        if len(df) == 0:
            continue
        ax.hist(df["target"], bins=bins, alpha=0.45, label=f"{name}  n={len(df)}")
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("3-day VIX change")
    ax.set_ylabel("Document count")
    ax.set_title("Target distribution by temporal segment")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 6. Attention heatmaps
# ---------------------------------------------------------------------------


def plot_attention_examples(
    documents_df: pd.DataFrame,
    embeddings_path: Path,
    model: SentenceAttentionModel,
    out_path: Path,
    *,
    batch_size: int,
    device: torch.device,
    n_examples: int = 4,
    max_label_chars: int = 90,
) -> Path:
    """For a handful of representative docs, show per-sentence attention weights.

    We pick documents that span the date range (oldest, ~33rd percentile,
    ~66th percentile, most recent), so the figure illustrates whether the model
    attends to different sentences across eras.
    """
    if len(documents_df) == 0:
        # Make an empty placeholder so the script doesn't crash.
        fig = plt.figure(); _save(fig, out_path); return out_path

    df_sorted = documents_df.sort_values("release_date").reset_index(drop=True)
    pick_idx = np.linspace(0, len(df_sorted) - 1, n_examples).round().astype(int)
    picks = df_sorted.iloc[pick_idx].reset_index(drop=True)

    out = _predict_split(model, picks, embeddings_path, batch_size, device,
                         return_attention=True)

    fig, axes = plt.subplots(n_examples, 1, figsize=(11, 2.4 * n_examples))
    if n_examples == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        row = picks.iloc[i]
        attn = out["attentions"][i]
        sents = list(row["sentences"])[: len(attn)]
        idx = np.arange(len(attn))
        ax.barh(idx, attn, color="#4C78A8")
        labels = [
            (s[:max_label_chars] + "…") if len(s) > max_label_chars else s
            for s in sents
        ]
        ax.set_yticks(idx)
        ax.set_yticklabels(labels, fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel("Attention weight")
        title = (
            f"{row['doc_id']}  ({pd.Timestamp(row['release_date']).date()})  "
            f"target={float(row['target']):+.2f}  pred={float(out['predictions'][i]):+.2f}"
        )
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(True, axis="x", alpha=0.3)

    fig.suptitle("Per-sentence attention weights for representative documents", y=1.001)
    fig.tight_layout()
    return _save(fig, out_path)


# ---------------------------------------------------------------------------
# 7. Sentence count distribution
# ---------------------------------------------------------------------------


def plot_sentence_count_distribution(
    documents_df: pd.DataFrame,
    out_path: Path,
    *,
    cap: int,
) -> Path:
    """Histogram of n_sentences per document (sanity check: most should hit the cap)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(documents_df["n_sentences"], bins=np.arange(0, cap + 5, 2))
    ax.axvline(cap, color="red", linestyle="--", alpha=0.6, label=f"cap = {cap}")
    ax.set_xlabel("Sentences per document (after segmentation, pre-cap)")
    ax.set_ylabel("Document count")
    ax.set_title("Sentence count distribution across the corpus")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, out_path)
