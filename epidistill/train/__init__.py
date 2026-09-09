"""Training objective and optimisation for EpiDistill (multi-view teacher, single-view student)."""

from .losses import (
    LOSS_WEIGHTS,
    ObjectiveTerms,
    SupervisedLosses,
    attention_loss,
    distillation_losses,
    student_scale_loss,
    token_distillation_loss,
    token_distillation_weight,
)
from .optim import WeightEMA, build_optimizer, build_scheduler, clip_gradients

__all__ = [
    "LOSS_WEIGHTS",
    "ObjectiveTerms",
    "SupervisedLosses",
    "attention_loss",
    "distillation_losses",
    "student_scale_loss",
    "token_distillation_loss",
    "token_distillation_weight",
    "WeightEMA",
    "build_optimizer",
    "build_scheduler",
    "clip_gradients",
]
