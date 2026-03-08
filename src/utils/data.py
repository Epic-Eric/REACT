# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Data utilities for REACT-EMG.

This module provides configuration, constants, and factory functions for
creating EMG datasets and DataLoaders.  Heavy-lifting classes and helpers
live in dedicated sub-modules:

* ``src.utils.datasets``   -- CalibratedEmgDataset, PrebuiltCalibratedDataset
* ``src.utils.cache``      -- validation / calibration cache I/O
* ``src.utils.validation`` -- parallel HDF5 session validation
* ``src.utils.collate``    -- collate functions for DataLoader

All symbols are re-exported here for backward compatibility, so existing
``from src.utils.data import X`` imports continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

# -- Re-exports from sub-modules ---------------------------------------------
from .cache import (  # noqa: F401
    cache_key,
    load_calibration_cache,
    load_validation_cache,
    save_calibration_cache,
    save_validation_cache,
)
from .collate import (  # noqa: F401
    collate_calibrated_batch,
    collate_with_variable_calibration,
)
from .datasets import (  # noqa: F401
    CalibratedEmgDataset,
    LazyCalibratedEmgDataset,
    PrebuiltCalibratedDataset,
    get_session_path,
    get_user_from_session,
)
from .validation import parallel_validate_sessions  # noqa: F401


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "emg2pose_dataset_mini"
DATASET_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
)

MINI_SPLIT = {
    "train": [
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left",
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_right",
    ],
    "val": [
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-2_left",
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-2_right",
    ],
    "test": [
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-3_left",
        "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-3_right",
    ],
}


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_dir: Path = DEFAULT_DATA_DIR
    window_length: int = 10_000  # 5 seconds at 2 kHz
    stride: int = 2_000  # 1 second stride
    jitter: bool = True  # Random window offset during training
    skip_ik_failures: bool = True
    num_workers: int = 4
    batch_size: int = 32


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def precompute_dataset_cache(
    data_dir: Path,
    metadata_df: "pd.DataFrame",
    cache_dir: Path,
    window_length: int = 10_000,
    stride: int = 2_000,
    skip_ik_failures: bool = True,
    num_workers: int = 8,
) -> None:
    """Pre-validate sessions per split and save caches to persistent storage.

    Run this once (e.g. as a separate Modal function) to build per-split
    validation manifests.  Subsequent training runs will load the cache and
    skip the ~5-minute validation step entirely.

    Splits are determined by the ``split`` column in the metadata CSV.
    Train uses *stride*; val and test use *stride x 2*.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _infos(split_name: str):
        filtered = metadata_df[metadata_df["split"] == split_name]
        return [(row["filename"], row["user"]) for _, row in filtered.iterrows()]

    splits = [
        ("train", _infos("train"), stride),
        ("val", _infos("val"), stride * 2),
        ("test", _infos("test"), stride * 2),
    ]

    for split_name, infos, split_stride in splits:
        print(
            f"\n=== Pre-validating {split_name} split: {len(infos)} sessions "
            f"(stride={split_stride}) ==="
        )
        valid_sessions = parallel_validate_sessions(
            data_dir=data_dir,
            session_infos=infos,
            window_length=window_length,
            stride=split_stride,
            skip_ik_failures=skip_ik_failures,
            num_workers=num_workers,
            show_progress=True,
        )
        save_validation_cache(
            cache_dir,
            window_length,
            split_stride,
            skip_ik_failures,
            valid_sessions,
            split_name=split_name,
        )
        print(f"{split_name}: {len(valid_sessions)}/{len(infos)} sessions valid.")

    print("\nAll split caches saved.")


def create_lazy_datasets_from_metadata(
    data_dir: Path,
    metadata_df: "pd.DataFrame",
    window_length: int = 10_000,
    stride: int = 2_000,
    calibration_k: int = 30,
    min_calibration_k: int = 0,
    cache_size: int = 100,  # Ignored, kept for backwards compatibility
    num_workers: int = 8,
    cache_dir: Optional[Path] = None,
) -> Tuple[
    PrebuiltCalibratedDataset,
    PrebuiltCalibratedDataset,
    PrebuiltCalibratedDataset,
]:
    """Create pre-built train/val/test datasets from metadata DataFrame.

    Splits are determined by the ``split`` column in the metadata CSV.
    Train stride = *stride*; val/test stride = *stride x 2*.
    """

    def _infos(split_name: str) -> List[Tuple[str, str]]:
        filtered = metadata_df[metadata_df["split"] == split_name]
        return [(row["filename"], row["user"]) for _, row in filtered.iterrows()]

    train_infos = _infos("train")
    val_infos = _infos("val")
    test_infos = _infos("test")

    print(
        f"Creating datasets: train={len(train_infos)}, "
        f"val={len(val_infos)}, test={len(test_infos)}"
    )
    if cache_dir:
        print(f"Using cache directory: {cache_dir}")

    print("\n=== Building TRAIN dataset ===")
    train_dataset = PrebuiltCalibratedDataset(
        data_dir=data_dir,
        session_infos=train_infos,
        window_length=window_length,
        stride=stride,
        jitter=True,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        show_progress=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        split_name="train",
    )

    print("\n=== Building VAL dataset ===")
    val_dataset = PrebuiltCalibratedDataset(
        data_dir=data_dir,
        session_infos=val_infos,
        window_length=window_length,
        stride=stride * 2,
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        show_progress=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        split_name="val",
    )

    print("\n=== Building TEST dataset ===")
    test_dataset = PrebuiltCalibratedDataset(
        data_dir=data_dir,
        session_infos=test_infos,
        window_length=window_length,
        stride=stride * 2,
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        show_progress=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        split_name="test",
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    data_dir: Optional[Path] = None,
    split: Optional[Dict[str, List[str]]] = None,
    window_length: int = 10_000,
    stride: int = 2_000,
    batch_size: int = 32,
    num_workers: int = 4,
    calibration_k: int = 5,
    min_calibration_k: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders (mini dataset)."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    split = split or MINI_SPLIT

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Download with:\n"
            f"  cd ~ && curl '{DATASET_URL}' -o emg2pose_dataset_mini.tar\n"
            f"  tar -xvf emg2pose_dataset_mini.tar"
        )

    datasets = {}
    for name in ("train", "val", "test"):
        datasets[name] = CalibratedEmgDataset(
            data_dir=data_dir,
            session_names=split[name],
            window_length=window_length,
            stride=stride,
            jitter=(name == "train"),
            calibration_k=calibration_k,
            min_calibration_k=min_calibration_k,
        )

    loaders = tuple(
        DataLoader(
            datasets[name],
            batch_size=batch_size,
            shuffle=(name == "train"),
            num_workers=num_workers,
            pin_memory=True,
        )
        for name in ("train", "val", "test")
    )
    return loaders  # type: ignore[return-value]


# -----------------------------------------------------------------------------
# Small Utilities
# -----------------------------------------------------------------------------


def create_dummy_batch(
    batch_size: int = 4,
    emg_channels: int = 16,
    num_joints: int = 20,
    seq_length: int = 10_000,
    calibration_k: int = 5,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a dummy batch for quick testing (no real data required)."""
    return {
        "emg": torch.randn(batch_size, emg_channels, seq_length, device=device),
        "joint_angles": torch.randn(
            batch_size, num_joints, seq_length, device=device
        ),
        "calibration_emg": torch.randn(
            batch_size, calibration_k, emg_channels, seq_length, device=device
        ),
        "calibration_k": torch.full((batch_size,), calibration_k, device=device),
        "no_ik_failure": torch.ones(
            batch_size, seq_length, dtype=torch.bool, device=device
        ),
    }


def check_dataset_available(data_dir: Optional[Path] = None) -> bool:
    """Check if emg2pose_dataset_mini is available."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return data_dir.exists() and any(data_dir.glob("*.hdf5"))


def download_dataset_instructions() -> str:
    """Return instructions for downloading the dataset."""
    return (
        "emg2pose_dataset_mini not found. To download:\n\n"
        "    cd ~\n"
        f'    curl "{DATASET_URL}" -o emg2pose_dataset_mini.tar\n'
        "    tar -xvf emg2pose_dataset_mini.tar\n\n"
        "This will create ~/emg2pose_dataset_mini/ with HDF5 session files.\n"
    )
