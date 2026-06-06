"""Training subpackage: train loop, LR schedule, evaluation metrics.

Public surface:
    loop.train
    schedule.linear_warmup
    eval.regression_metrics
"""

from .loop import train
from .schedule import linear_warmup
from .eval import regression_metrics

__all__ = [
    "train",
    "linear_warmup",
    "regression_metrics",
]
