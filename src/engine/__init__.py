# Copyright (c) 2026 REACT-EMG Authors
"""Training and evaluation engines for REACT-EMG."""

from .trainer import (
    Trainer,
    TrainerConfig,
    TrainingState,
)
from .evaluator import (
    Evaluator,
    EvaluatorConfig,
    EvaluationResults,
)

__all__ = [
    "Trainer",
    "TrainerConfig",
    "TrainingState",
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResults",
]
