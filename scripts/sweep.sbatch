#!/bin/bash
#SBATCH --job-name=fed-vix-sweep
#SBATCH --output=slurm_logs/sweep-%j.out
#SBATCH --error=slurm_logs/sweep-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --account=dsci410_510

# Hyperparameter sweep over learning_rate and attn_dim (6 runs).
# Each run is ~30 sec on the A100 MIG; total wall-clock is a few minutes
# including data load and sweep summary writes. The winning run's artifacts
# are auto-promoted to outputs/ so the standard downstream scripts pick them
# up unchanged.
#
# After this job finishes, the recommended sequence is:
#   sbatch scripts/train.sbatch              # only re-runs regime + baseline + plots
#   python scripts/export_examples.py        # re-bundles demo with the winning model

set -euo pipefail

# Same env activation as train.sbatch.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate base
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd /home/ztomlins/Transcripts-Fed-VIX-DL

python -c "import torch; print(f'torch {torch.__version__} | cuda: {torch.cuda.is_available()} | dev: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\"}')"

# Ensure data + embeddings are in place. These are no-ops on cached state.
python scripts/build_data.py --config configs/default.yaml
python scripts/precompute_embeddings.py --config configs/default.yaml

# Run the sweep. --skip-completed makes the job resumable if it gets pre-empted.
python scripts/run_sweep.py --config configs/sweep.yaml --skip-completed

# Generate final figures from the promoted winner.
python scripts/run_regime_analysis.py --config configs/default.yaml
python scripts/run_baselines.py --config configs/default.yaml
python scripts/make_plots.py --config configs/default.yaml

echo "Sweep complete. Winner promoted to outputs/. See outputs/sweep/sweep_table.csv for the full grid."
