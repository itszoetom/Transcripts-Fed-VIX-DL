"""Linear warmup learning-rate scheduler.

Standard transformer-paper schedule: linearly ramp from 0 to the configured
base learning rate over the first `warmup_steps`, then hold constant. We use
constant-after-warmup (not linear or cosine decay) because:

  - The trained head is small (~100k params) on a tiny dataset; aggressive
    decay tends to underfit before the head has converged.
  - Early stopping on val MSE makes a constant post-warmup LR perfectly
    serviceable — the optimizer stops when val stops improving.
"""

from __future__ import annotations

import torch


def linear_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Return a LambdaLR that linearly warms up over `warmup_steps`.

    After warmup, the multiplier is held at 1.0 — i.e., LR == base LR.

    Args:
        optimizer:    Any torch optimizer.
        warmup_steps: Number of training steps over which to linearly ramp
                      the LR multiplier from 0 -> 1. If 0 or negative,
                      multiplier is always 1 (no warmup).
    """
    if warmup_steps <= 0:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)

    def lr_lambda(step: int) -> float:
        # step is 0-indexed; ramp until we hit warmup_steps.
        return min(1.0, float(step + 1) / float(warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
