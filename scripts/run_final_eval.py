"""Final out-of-sample evaluation: pooled regression + directional classifier.

This is the "one last job" script. It does two things on the POOLED test set
(regime1 + regime2 + regime3 concatenated, i.e. every post-2017 document the
model never saw), instead of the per-regime breakdown:

  PART A - Pooled regression (re-uses the already-trained model, NO retrain)
      * Load the trained SentenceAttentionModel from --model.
      * Predict on the pooled OOS test set.
      * Report pooled MSE / R^2 / Pearson r / p, plus the prediction spread
        (pred std vs target std) that shows whether the model actually moves
        off the mean or just predicts ~0.
      * Directional accuracy of sign(prediction) vs sign(target), against the
        majority-class baseline.
      * Save a pooled predicted-vs-actual scatter.

  PART B - Directional classifier (cheap retrain of the ~100k-param head only)
      * Same architecture; the head's scalar output is treated as a logit.
      * Binary target = (10-day VIX change > 0). Trained with BCE on the same
        train split, early-stopped on validation BCE.
      * Report pooled accuracy / AUC vs the majority-class baseline, and save a
        confusion-matrix + ROC figure.

Why this answers the "make it predict more drastically" question: scaling the
regression outputs up cannot change Pearson r (it is scale-invariant) and only
worsens MSE, so a "confidence boost" is a non-starter. Reframing as direction
(up/down) is the legitimate way to ask whether ANY signal is present.

Run on Talapas (where data/processed/{documents.parquet,sentence_embeddings.pt}
live), from the repo root:

    python scripts/run_final_eval.py \
        --config outputs/winning_config.yaml \
        --model  outputs/model.pt \
        --out    outputs/final_eval

NOTE: the targets in data/processed must match the horizon the model was
trained on (10-day for the bundled winning model). The script prints the
target stats so you can sanity-check before trusting the numbers.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from transcripts_fed_vix.data.dataset import EmbeddingDocDataset, collate_padded
from transcripts_fed_vix.models import SentenceAttentionModel
from transcripts_fed_vix.models.attention import attention_config_from_dict
from transcripts_fed_vix.training.eval import regression_metrics
from transcripts_fed_vix.utils import set_seed, make_temporal_splits, SplitDates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("final_eval")


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _loader(df: pd.DataFrame, embeddings_path: Path, batch_size: int) -> DataLoader:
    ds = EmbeddingDocDataset(df, embeddings_path=embeddings_path)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                      collate_fn=collate_padded)


def _forward_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run the model and return (scalar_outputs, targets) as numpy arrays.

    The scalar output doubles as a regression prediction (Part A) and as a
    classification logit (Part B); the caller decides how to interpret it.
    """
    model.eval()
    outs, tgts = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["embeddings"].to(device), batch["mask"].to(device))
            outs.extend(out.prediction.detach().cpu().tolist())
            tgts.extend(batch["target"].tolist())
    return np.asarray(outs, dtype=float), np.asarray(tgts, dtype=float)


def _directional(preds: np.ndarray, tgts: np.ndarray) -> dict:
    """sign(pred) vs sign(target) accuracy against the majority-class baseline.

    Zero-change targets are dropped (no direction to predict).
    """
    keep = tgts != 0.0
    p_up = preds[keep] > 0.0
    t_up = tgts[keep] > 0.0
    acc = float((p_up == t_up).mean())
    base_rate = float(t_up.mean())
    majority = max(base_rate, 1.0 - base_rate)
    return {
        "n": int(keep.sum()),
        "accuracy": acc,
        "frac_actually_up": base_rate,
        "majority_class_baseline": majority,
        "beats_baseline": acc > majority,
    }


# ---------------------------------------------------------------------------
# PART B helpers: train the directional head with BCE on cached embeddings.
# ---------------------------------------------------------------------------

def _train_direction_classifier(model, train_loader, val_loader, cfg, device, save_path):
    """Train the same architecture as a binary up/down classifier (BCE).

    Only the attention + head parameters train (the encoder is frozen and its
    embeddings are pre-computed). Early-stops on validation BCE, mirroring the
    regression loop's early-stopping policy.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg["training"]["learning_rate"]),
                            weight_decay=float(cfg["training"]["weight_decay"]))
    loss_fn = nn.BCEWithLogitsLoss()
    epochs = int(cfg["training"]["epochs"])
    patience = int(cfg["training"]["early_stop_patience"])
    clip = float(cfg["training"]["grad_clip_norm"])

    best_val = float("inf")
    best_epoch = -1
    waited = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            emb = batch["embeddings"].to(device)
            mask = batch["mask"].to(device)
            y = (batch["target"].to(device) > 0).float()   # binary up/down label
            opt.zero_grad(set_to_none=True)
            logit = model(emb, mask).prediction
            loss = loss_fn(logit, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=clip)
            opt.step()

        # ----- validation BCE -----
        model.eval()
        vloss, nb = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                emb = batch["embeddings"].to(device)
                mask = batch["mask"].to(device)
                y = (batch["target"].to(device) > 0).float()
                vloss += float(loss_fn(model(emb, mask).prediction, y).item())
                nb += 1
        vloss /= max(1, nb)
        logger.info("clf epoch=%d val_bce=%.4f", epoch, vloss)

        if vloss < best_val - 1e-8:
            best_val, best_epoch, waited = vloss, epoch, 0
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
        else:
            waited += 1
            if waited >= patience:
                logger.info("clf early stop at epoch %d", epoch)
                break
    return best_epoch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=Path("outputs/winning_config.yaml"))
    ap.add_argument("--model", type=Path, default=Path("outputs/model.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/final_eval"))
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    set_seed(int(cfg["seed"]))
    device = _resolve_device(cfg["model"].get("device", "auto"))

    processed_dir = Path(cfg["data"]["processed_dir"])
    documents_path = processed_dir / cfg["data"]["documents_file"]
    embeddings_path = processed_dir / cfg["data"]["embeddings_file"]
    batch_size = int(cfg["training"]["batch_size"])
    fig_dir = args.out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    documents = pd.read_parquet(documents_path)
    logger.info("loaded %d documents; horizon=%s days; target mean=%.3f std=%.3f",
                len(documents), cfg["data"].get("target_horizon_trading_days"),
                documents["target"].mean(), documents["target"].std())

    splits = make_temporal_splits(
        documents,
        dates=SplitDates(
            train_end=date.fromisoformat(cfg["splits"]["train_end_date"]),
            regime2_start=date.fromisoformat(cfg["splits"]["regime2_start_date"]),
            regime3_start=date.fromisoformat(cfg["splits"]["regime3_start_date"]),
            val_fraction=float(cfg["splits"]["val_fraction"]),
        ),
    )
    pooled_test = pd.concat([splits.regime1, splits.regime2, splits.regime3]).reset_index(drop=True)
    logger.info("pooled OOS test set: n=%d (r1=%d r2=%d r3=%d)",
                len(pooled_test), len(splits.regime1), len(splits.regime2), len(splits.regime3))

    model_cfg = attention_config_from_dict(cfg["model"])

    # ===================================================================
    # PART A - pooled regression, existing trained model (no retrain)
    # ===================================================================
    reg_model = SentenceAttentionModel(model_cfg).to(device)
    reg_model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    preds, tgts = _forward_logits(reg_model, _loader(pooled_test, embeddings_path, batch_size), device)
    rm = regression_metrics(preds, tgts)
    direction_from_reg = _directional(preds, tgts)

    part_a = {
        "pooled_regression": rm.to_dict(),
        "prediction_std": float(preds.std()),
        "target_std": float(tgts.std()),
        "constant_mean_predictor_mse": float(tgts.var()),  # MSE if you always predict the mean
        "directional_accuracy_from_regression": direction_from_reg,
    }
    logger.info("POOLED regression: r=%.3f p=%.3f mse=%.2f | pred_std=%.3f vs target_std=%.3f",
                rm.pearson_r, rm.pearson_p, rm.mse, preds.std(), tgts.std())
    logger.info("directional acc (sign of regression): %.3f vs majority %.3f",
                direction_from_reg["accuracy"], direction_from_reg["majority_class_baseline"])

    # ---- pooled predicted-vs-actual scatter ----
    fig, ax = plt.subplots(figsize=(6.5, 6))
    lim = float(np.nanmax(np.abs(np.concatenate([preds, tgts])))) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "--", color="grey", linewidth=1, label="y = x")
    ax.scatter(tgts, preds, s=40, alpha=0.7, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("True 10-day VIX change")
    ax.set_ylabel("Predicted 10-day VIX change")
    sig = "" if np.isnan(rm.pearson_p) else f", p = {rm.pearson_p:.3f}"
    ax.set_title(f"Pooled out-of-sample (n = {rm.n}):  r = {rm.pearson_r:+.2f}{sig}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(fig_dir / "pooled_predicted_vs_actual.png", dpi=150)
    plt.close(fig)

    # ===================================================================
    # PART B - directional classifier (cheap head retrain with BCE)
    # ===================================================================
    clf = SentenceAttentionModel(model_cfg).to(device)
    clf_path = args.out / "direction_model.pt"
    best_epoch = _train_direction_classifier(
        clf,
        _loader(splits.train, embeddings_path, batch_size),
        _loader(splits.val, embeddings_path, batch_size),
        cfg, device, clf_path,
    )
    clf.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))

    logits, tgts_c = _forward_logits(clf, _loader(pooled_test, embeddings_path, batch_size), device)
    probs = 1.0 / (1.0 + np.exp(-logits))
    keep = tgts_c != 0.0
    y_true = (tgts_c[keep] > 0).astype(int)
    y_prob = probs[keep]
    y_pred = (y_prob > 0.5).astype(int)

    acc = float((y_pred == y_true).mean())
    base_rate = float(y_true.mean())
    majority = max(base_rate, 1.0 - base_rate)
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = float("nan")  # only one class present
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    part_b = {
        "best_epoch": best_epoch,
        "n": int(keep.sum()),
        "accuracy": acc,
        "auc": auc,
        "frac_actually_up": base_rate,
        "majority_class_baseline": majority,
        "beats_baseline": acc > majority,
        "confusion_matrix_rows_true_cols_pred_labels_0down_1up": cm,
    }
    logger.info("CLASSIFIER pooled: acc=%.3f auc=%.3f vs majority %.3f", acc, auc, majority)

    # ---- confusion matrix + ROC figure ----
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 5))
    cm_arr = np.array(cm)
    a0.imshow(cm_arr, cmap="Blues")
    a0.set_xticks([0, 1], ["pred down", "pred up"])
    a0.set_yticks([0, 1], ["true down", "true up"])
    for i in range(2):
        for j in range(2):
            a0.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black", fontsize=14)
    a0.set_title(f"Direction confusion (acc = {acc:.2f}, baseline = {majority:.2f})")
    if not np.isnan(auc):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        a1.plot(fpr, tpr, color="#4C72B0", linewidth=2, label=f"AUC = {auc:.2f}")
    a1.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="chance")
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("Direction ROC"); a1.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "direction_classifier.png", dpi=150)
    plt.close(fig)

    # ===================================================================
    report = {"part_a_pooled_regression": part_a, "part_b_directional_classifier": part_b}
    (args.out / "final_eval.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s and figures to %s", args.out / "final_eval.json", fig_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
