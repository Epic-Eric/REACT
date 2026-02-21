# Copyright (c) 2026 REACT-EMG Authors
"""Utility functions for REACT-EMG."""

from .signal_proc import (
    normalize_emg,
    bandpass_filter,
    compute_rms,
    compute_envelope,
)
from .visualization import (
    plot_training_curves,
    plot_predictions,
    plot_user_embeddings,
    plot_calibration_attention,
)
from .dummy_data import (
    DummyDataGenerator,
    create_dummy_dataset,
    create_dummy_dataloaders,
)

__all__ = [
    "normalize_emg",
    "bandpass_filter",
    "compute_rms",
    "compute_envelope",
    "plot_training_curves",
    "plot_predictions",
    "plot_user_embeddings",
    "plot_calibration_attention",
    "DummyDataGenerator",
    "create_dummy_dataset",
    "create_dummy_dataloaders",
]
