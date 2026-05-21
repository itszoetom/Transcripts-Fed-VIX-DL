"""Reproducibility: a single set_seed() that touches all the relevant RNGs.

We seed: Python's `random`, NumPy, and PyTorch (CPU + CUDA). This does NOT
make CUDA convolutions fully deterministic (which would also require
`torch.use_deterministic_algorithms(True)` and `CUBLAS_WORKSPACE_CONFIG`); we
explicitly choose not to, because the cost (some kernels disabled) outweighs
the benefit for this small attention model. Run-to-run differences should be
within noise on this dataset size.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs.

    Args:
        seed: Integer seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
