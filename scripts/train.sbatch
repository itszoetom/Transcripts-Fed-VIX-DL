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


cd /home/ztomlins/Transcripts-Fed-VIX-DL
source venv/bin/activate

python scripts/train_model.py --config configs/default.yaml
