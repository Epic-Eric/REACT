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
from .data import (
    CalibratedEmgDataset,
    create_dataloaders,
    create_dummy_batch,
    check_dataset_available,
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
    "CalibratedEmgDataset",
    "create_dataloaders",
    "create_dummy_batch",
    "check_dataset_available",
]
