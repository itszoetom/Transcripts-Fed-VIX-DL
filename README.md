# Transcripts-Fed-VIX-DL

Predicting 3-day forward VIX change from Federal Reserve text (FOMC minutes + Humphrey-Hawkins testimony) via **frozen FinBERT + learned sentence-attention**, with structural-break analysis across U.S. political regimes.

DSCI 410/510 — Project Milestone (Zoe Tomlinson)

## TL;DR

- **Inputs:** first 80 sentences of each Fed document (FOMC minutes 1993+, HH testimony 1997+), encoded with frozen `yiyanghkust/finbert-pretrain` + mask-weighted mean pooling.
- **Aggregation:** learned additive (Bahdanau-style) attention across sentences.
- **Head:** linear regression to a scalar 3-day VIX change.
- **Trained components:** attention layer + linear head only (~100k parameters). The encoder is fully frozen.
- **Splits:** strictly temporal, anchored to U.S. presidential inauguration days (2017-01-20, 2021-01-20, 2025-01-20).
- **Baselines:** TF-IDF + Ridge (regression); BoW + Logistic Regression (binarized target).
- **Regime analysis:** R² degradation across regimes + Chow F-test on residuals.

## Quick demo (no Talapas, no FRED key needed)

```bash
git clone https://github.com/itszoetom/Transcripts-Fed-VIX-DL.git
cd Transcripts-Fed-VIX-DL
pip install -e .
jupyter notebook notebooks/data_demo.ipynb
```

The notebook loads 10 bundled example documents from `data/example/`, demonstrates the DataLoader API, and shows what a training batch looks like. No network or cluster access required.

## Documentation

- `docs/MILESTONE_REPORT.md` — milestone summary (the 4 required questions + a "changes since proposal" section).
- `docs/METHODOLOGY.md` — comprehensive methodology write-up with rationale and citations for every non-trivial decision.
- `docs/DL410-Project-Proposal.docx` — original project proposal.

## Project structure

```
src/transcripts_fed_vix/
├── data/         — scraping, sentence segmentation, VIX alignment, DataLoader
├── models/       — frozen FinBERT encoder + additive-attention head
├── training/     — manual PyTorch training loop, LR schedule, evaluation metrics
└── utils/        — seed, temporal splits, Chow test, plotting
configs/default.yaml — single canonical config
scripts/
├── build_data.py            — scrape + segment + VIX-align → processed parquet
├── precompute_embeddings.py — one-time FinBERT forward over all sentences
├── train_model.py           — train + early-stop + per-segment evaluation
├── run_regime_analysis.py   — Chow test on residuals
├── run_baselines.py         — TF-IDF Ridge + BoW Logistic
├── export_examples.py       — produce the bundled demo data
├── make_plots.py            — generate outputs/figures/*.png
└── train.sbatch             — SLURM entry point
notebooks/data_demo.ipynb    — runnable demo of the DataLoader
data/example/                — ~10 sample documents + embeddings + model (committed)
```

## Full pipeline (Talapas)

```bash
ssh ztomlins@login.talapas.uoregon.edu
cd /home/ztomlins/Transcripts-Fed-VIX-DL
git pull
pip install -e .
sbatch scripts/train.sbatch
tail -f slurm_logs/train-*.err
```

The SLURM job runs scrape → segment → VIX-align → FinBERT precompute → train → per-segment evaluation → Chow test → baselines → figure generation. Each stage is idempotent and skips already-completed work; re-runs after a successful scrape are dominated by training (~30 s).

Outputs land in `outputs/`:
- `model.pt`, `train_metrics.json` — trained model + per-epoch metrics
- `final_report.json` — per-segment regression + binary metrics
- `regime_analysis.json` — R² by regime + Chow tests
- `baseline_metrics.json` — TF-IDF Ridge + BoW Logistic
- `figures/*.png` — training curves, predicted-vs-actual, residuals-over-time with breakpoints, attention heatmaps, etc.

## Environment

- Python ≥ 3.10 (tested 3.12 on Talapas).
- CUDA-enabled torch wheel matched to your driver (Talapas A100 nodes: cu121 or cu130 works).
- `FRED_API_KEY` environment variable (free key from <https://fred.stlouisfed.org/docs/api/api_key.html>) — required only for the full scrape, not for the demo notebook.

## Data access

- The demo notebook uses 10 bundled example documents at `data/example/`. No external access required.
- Full corpus is reconstructed from `federalreserve.gov` via `scripts/build_data.py`. Raw HTML/PDF + extracted text are cached under `data/raw/` to make re-runs cheap. The build is fully deterministic given the cached files.
