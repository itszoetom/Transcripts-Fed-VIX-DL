# Transcripts-Fed-VIX-DL

Predicting VIX (CBOE Volatility Index) movements from FOMC transcript text using deep sequence models.

## Setup

### Local (development)
```bash
git clone git@github.com:itszoetom/Transcripts-Fed-VIX-DL.git
cd Transcripts-Fed-VIX-DL
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

### Talapas (training)
```bash
ssh ztomlins@login1.talapas.uoregon.edu
cd ~/Transcripts-Fed-VIX-DL
source venv/bin/activate
sbatch scripts/train.sbatch
```

## Project Structure

- `src/transcripts_fed_vix/` — package source
  - `data/` — Dataset classes for transcripts and VIX targets
  - `models/` — encoder + head architectures
  - `training/` — train/eval loops
  - `utils/` — config, logging, metrics
- `configs/` — experiment configs (YAML)
- `scripts/` — entry-point scripts and SLURM batch files
- `notebooks/` — exploration and analysis
- `data/` — raw and processed data (gitignored)
- `outputs/` — trained models and metrics (gitignored)

## Data

- FOMC transcripts: TBD
- VIX historical: Yahoo Finance / FRED
