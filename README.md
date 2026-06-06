# Transcripts-Fed-VIX-DL

Predicting the 3-day forward VIX change from Federal Reserve text (FOMC minutes plus Humphrey-Hawkins testimony) using a frozen FinBERT encoder, a learned additive (Bahdanau-style) sentence attention aggregator, and a linear regression head.

DSCI 410/510 Final Project, Zoe Tomlinson.

## Project purpose

The Federal Reserve is the U.S. central bank; its decisions about interest rates propagate into the cost of credit, asset prices, and inflation. The FOMC (Federal Open Market Committee) is the 12-member subcommittee that actually sets policy and releases minutes about three weeks after each of its eight annual meetings. The Fed Chair also testifies before Congress twice a year (Humphrey-Hawkins testimony). Markets parse this text line by line for signals about future rate moves.

The VIX (CBOE Volatility Index) is the market's expectation of 30-day forward volatility in the S&P 500, derived from short-dated option prices and commonly nicknamed the "investor fear gauge" (Whaley 2000). When the market expects calm, VIX falls; when traders see uncertainty ahead, VIX rises.

This project asks two questions:

1. **Predictive (RQ1):** Does the textual content of Fed communications predict the 3-day forward change in the VIX after each document's release? Prior work shows that Fed text moves Treasury yields and equity measures (Hansen, McMahon & Prat 2018; Lucca & Trebbi 2009), so a measurable signal on short-horizon volatility is plausible.
2. **Temporal generalization (RQ2):** Does that relationship hold across U.S. political regimes? The Fed is formally independent, but presidential pressure on the Chair has varied sharply between administrations (Obama, Trump 1, Biden, Trump 2). A model trained on pre-2017 text might generalize poorly to later regimes.

A full intuition-level walkthrough plus citations lives in `docs/METHODOLOGY.md`.

## Dataset

326 documents scraped from `federalreserve.gov` and aligned to a FRED VIX series:

- **FOMC minutes** (1993 to present, 268 docs across four URL conventions the Fed has used).
- **Humphrey-Hawkins / Semiannual Monetary Policy Report testimony** (1997 to present, 58 docs).

Each document is sentence-segmented with NLTK Punkt (Kiss & Strunk 2006) and truncated to the first 80 sentences, which carry the policy rationale. Each sentence is then encoded by `yiyanghkust/finbert-pretrain` and mask-weighted mean-pooled into one 768-d vector. Embeddings are deterministic and pre-computed once.

**Target:** 3-day close-to-close VIX change (FRED `VIXCLS`), aligned to the next trading day so there is no look-ahead leakage on weekend or holiday releases.

**Splits** are strictly temporal, anchored to U.S. presidential inauguration days:

| Segment   | Date range                       | Approx n |
|-----------|----------------------------------|----------|
| Train     | release_date < 2017-01-20         | 233 docs |
| Val       | last 15 percent of train (chrono) | 35 docs  |
| Regime 1  | 2017-01-20 to 2021-01-20          | 40 docs  |
| Regime 2  | 2021-01-20 to 2025-01-20          | 40 docs  |
| Regime 3  | 2025-01-20 onward                 | 13 docs  |

The corpus is rebuilt deterministically by `scripts/build_data.py` from cached HTTP responses and a FRED API key. `notebooks/data_demo.ipynb` runs against 10 bundled example documents without Talapas or FRED.

## Model

`SentenceAttentionModel` (`src/transcripts_fed_vix/models/attention.py`):

```
sentences -> FinBERT (frozen) -> 80 x 768 sentence embeddings
       -> additive attention: u = tanh(W h + b), s = v^T u, alpha = softmax(s)
       -> weighted sum -> 768-d document vector
       -> dropout(0.1) -> linear head -> scalar prediction
```

Trainable parameters: about 99,000 (attention plus linear head). FinBERT itself is fully frozen, which is the standard small-data choice (Peters, Ruder & Smith 2019). Additive attention is the canonical Hierarchical Attention Network formulation (Yang et al. 2016) and gives interpretable per-sentence weights that drive the project's main interpretability figure.

## Metrics

- **MSE** (training loss and direct comparison vs. baseline).
- **R^2** (fraction of variance explained, comparable across regimes).
- **Pearson r** (direction-of-signal, scale-free).

All three are reported per segment (val, regime 1, regime 2, regime 3) for both the deep model and a **TF-IDF plus Ridge baseline** (`scripts/run_baselines.py`).

## How to train

### Local quick demo

```bash
git clone https://github.com/itszoetom/Transcripts-Fed-VIX-DL.git
cd Transcripts-Fed-VIX-DL
pip install -e .
jupyter notebook notebooks/data_demo.ipynb
```

### Full pipeline on Talapas

```bash
ssh ztomlins@login.talapas.uoregon.edu
cd /home/ztomlins/Transcripts-Fed-VIX-DL
git pull
pip install -e .
sbatch scripts/sweep.sbatch
tail -f slurm_logs/sweep-*.err
```

`scripts/sweep.sbatch` runs scrape, sentence-segment, VIX-align, FinBERT precompute, a 6-cell hyperparameter sweep over `learning_rate` in {1e-4, 3e-4, 1e-3} crossed with `attn_dim` in {128, 256}, per-regime evaluation on the winner, the TF-IDF Ridge baseline, then figure generation. Each stage is idempotent.

## Results

After the Talapas sweep completes, fill these in from `outputs/final_report.json` and `outputs/baseline_metrics.json`:

| Segment   | Model MSE | Model R^2 | Model Pearson r | Baseline R^2 | Baseline Pearson r |
|-----------|-----------|-----------|------------------|--------------|---------------------|
| Val       | TBD       | TBD       | TBD              | n/a          | n/a                 |
| Regime 1  | TBD       | TBD       | TBD              | TBD          | TBD                 |
| Regime 2  | TBD       | TBD       | TBD              | TBD          | TBD                 |
| Regime 3  | TBD       | TBD       | TBD              | TBD          | TBD                 |

Figures (all in `outputs/figures/`):

1. `training_curve.png`: train MSE and val MSE per epoch with best-epoch marker.
2. `predicted_vs_actual.png`: scatter of predicted vs. true 3-day VIX change, one panel per non-train segment.
3. `residuals_over_time.png`: residual scatter over release date with vertical lines at the three regime boundaries.
4. `regression_comparison.png`: per-regime Pearson r and R^2 for the model vs. TF-IDF Ridge baseline.
5. `attention_examples.png`: per-sentence attention weights for four representative documents.

A worked inference example plus selected figures lives in `notebooks/eval.ipynb`.

## Limitations and use

- **Small training set.** About 233 pre-2017 documents is small even with frozen embeddings; results have meaningful variance.
- **Noisy target.** 3-day VIX change is dominated by short-horizon market microstructure; the language-extractable signal is real but modest in magnitude.
- **Per-regime sample sizes.** Regime 3 has only about 13 documents as of mid-2026, so its metrics are suggestive rather than definitive.
- **Use case.** Academic study of whether the signal exists, not a trading model. Do not use the predictions for live trading.

## Data and model paths

- **GitHub (bundled example):** `data/example/example_documents.json`, `example_embeddings.pt`, `example_model.pt` (committed; demo notebooks run offline).
- **Talapas (full corpus):** `/home/ztomlins/Transcripts-Fed-VIX-DL/data/raw/` (scraped HTML, PDF, text, VIX cache) and `/home/ztomlins/Transcripts-Fed-VIX-DL/data/processed/` (`documents.parquet`, `sentence_embeddings.pt`).
- **Talapas (trained weights):** `/home/ztomlins/Transcripts-Fed-VIX-DL/outputs/model.pt` plus the `*.json` reports for per-epoch, per-segment, and baseline metrics.

## Repository layout

```
src/transcripts_fed_vix/
  data/       scraping, sentence segmentation, VIX alignment, DataLoader
  models/     frozen FinBERT encoder + additive sentence attention head
  training/   training loop, LR schedule, regression metrics
  utils/      seed, temporal splits, plotting helpers
configs/      default.yaml (canonical) + sweep.yaml (6-run grid)
scripts/      build_data, precompute_embeddings, train_model, run_sweep,
              run_regime_analysis, run_baselines, make_plots, train.sbatch,
              sweep.sbatch
notebooks/    data_demo.ipynb (dataset + dataloader walkthrough),
              eval.ipynb (trained-model inference + figures)
docs/         METHODOLOGY.md (full writeup), MILESTONE_REPORT.md,
              DL410-Project-Proposal.docx
data/example/ 10 sample documents, embeddings, trained model
```

## Environment

Python 3.9+ (tested on 3.12 on Talapas). CUDA-enabled torch is required only for the FinBERT precompute step; everything else runs on CPU. The full scrape needs a free `FRED_API_KEY` (https://fred.stlouisfed.org/docs/api/api_key.html).
