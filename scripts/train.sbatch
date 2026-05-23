#!/bin/bash
#SBATCH --job-name=fed-vix
#SBATCH --output=slurm_logs/train-%j.out
#SBATCH --error=slurm_logs/train-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510

# Fail fast on errors; treat unset variables as errors; pipe failures propagate.
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
# Activate miniforge3 base. All project dependencies (torch, transformers,
# fredapi, pyarrow, statsmodels, beautifulsoup4, pypdf, nltk, etc.) are
# installed into user site-packages at ~/.local/lib/python3.13/site-packages
# and are visible from base because base is Python 3.13. We source conda.sh
# explicitly instead of relying on ~/.bashrc because SLURM jobs run in
# non-interactive shells that may not source .bashrc.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base

# Talapas system libstdc++ in /lib64 is older than what the conda env's
# binaries (icuuc.so.75 -> sqlite3) require. Force the linker to prefer the
# conda env's libstdc++ to fix `GLIBCXX_3.4.30 not found` on `import nltk`.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /home/ztomlins/Transcripts-Fed-VIX-DL

# Quick environment sanity check, prints to the .out log so failures are easy
# to triage.
python -c "import torch; print(f'torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# Each step is idempotent: re-running the job after a partial completion is
# safe and skips already-finished stages.

# 1. Build the processed dataset (scrape -> segment -> VIX-align -> parquet).
python scripts/build_data.py --config configs/default.yaml

# 2. Pre-compute frozen FinBERT sentence embeddings (cached to disk).
python scripts/precompute_embeddings.py --config configs/default.yaml

# 3. Train the sentence-attention model with early stopping.
python scripts/train_model.py --config configs/default.yaml

# 4. Run the temporal-generalization + Chow-test regime analysis.
python scripts/run_regime_analysis.py --config configs/default.yaml

# 5. Run the TF-IDF Ridge + BoW Logistic baselines on the same splits.
python scripts/run_baselines.py --config configs/default.yaml

# 6. Generate the standard set of project figures (training curves, scatter,
#    residuals over time, attention heatmaps, etc.).
python scripts/make_plots.py --config configs/default.yaml

echo "Pipeline complete. See outputs/ for model + metrics + figures."
