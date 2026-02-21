# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Data utilities for REACT-EMG using emg2pose_dataset_mini.

Wraps the emg2pose dataset with calibration sampling for user adaptation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Import emg2pose data classes
try:
    from emg2pose.data import Emg2PoseSessionData, WindowedEmgDataset
    from emg2pose.transforms import ExtractToTensor
except ImportError:
    raise ImportError(
        "emg2pose not found. Install with: pip install -e emg2pose/"
    )


# Default dataset location
DEFAULT_DATA_DIR = Path.home() / "emg2pose_dataset_mini"
DATASET_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_dir: Path = DEFAULT_DATA_DIR
    window_length: int = 10_000  # 5 seconds at 2kHz
    stride: int = 2_000  # 1 second stride
    jitter: bool = True  # Random window offset during training
    skip_ik_failures: bool = True
    num_workers: int = 4
    batch_size: int = 32


# Mini split for quick testing
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


def get_session_path(data_dir: Path, session_name: str) -> Path:
    """Get full path to session HDF5 file."""
    return data_dir / f"{session_name}.hdf5"


def get_user_from_session(session_name: str) -> str:
    """Extract user ID from session name."""
    # Session names follow pattern: date-timestamp-user-...-recording-N_side
    # e.g., "2022-12-06-1670313600-e3096-cv-emg-pose-train@2-recording-1_left"
    parts = session_name.split("-")
    # User ID is typically the 4th part (e.g., "e3096")
    if len(parts) >= 4:
        return parts[3]
    return "unknown"


class CalibratedEmgDataset(Dataset):
    """EMG dataset with calibration samples for user adaptation.
    
    Wraps WindowedEmgDataset and adds calibration sample retrieval.
    """
    
    def __init__(
        self,
        data_dir: Path,
        session_names: List[str],
        window_length: int = 10_000,
        stride: int = 2_000,
        jitter: bool = False,
        skip_ik_failures: bool = True,
        calibration_k: int = 5,
        calibration_pool_size: int = 50,
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to emg2pose_dataset_mini directory.
            session_names: List of session names to include.
            window_length: Window size in samples.
            stride: Stride between windows.
            jitter: Random offset during training.
            skip_ik_failures: Skip windows with IK failures.
            calibration_k: Number of calibration samples per batch item.
            calibration_pool_size: Size of calibration pool per user.
        """
        self.data_dir = Path(data_dir)
        self.session_names = session_names
        self.calibration_k = calibration_k
        self.calibration_pool_size = calibration_pool_size
        
        # Create windowed datasets for each session
        self.datasets: List[WindowedEmgDataset] = []
        self.session_to_user: Dict[str, str] = {}
        
        for session_name in session_names:
            hdf5_path = get_session_path(self.data_dir, session_name)
            if not hdf5_path.exists():
                print(f"Warning: Session file not found: {hdf5_path}")
                continue
                
            dataset = WindowedEmgDataset(
                hdf5_path=hdf5_path,
                window_length=window_length,
                stride=stride,
                jitter=jitter,
                skip_ik_failures=skip_ik_failures,
            )
            self.datasets.append(dataset)
            
            # Map session to user
            user_id = get_user_from_session(session_name)
            self.session_to_user[session_name] = user_id
        
        # Build combined dataset and index mapping
        self._build_index_mapping()
        
        # Build calibration pools per user
        self.calibration_pools = self._build_calibration_pools()
    
    def _build_index_mapping(self):
        """Build mapping from global index to (dataset_idx, local_idx)."""
        self.index_map: List[Tuple[int, int, str]] = []
        
        for ds_idx, (dataset, session_name) in enumerate(
            zip(self.datasets, self.session_names)
        ):
            for local_idx in range(len(dataset)):
                user_id = self.session_to_user.get(session_name, "unknown")
                self.index_map.append((ds_idx, local_idx, user_id))
    
    def _build_calibration_pools(self) -> Dict[str, List[torch.Tensor]]:
        """Build calibration sample pools per user."""
        pools: Dict[str, List[torch.Tensor]] = {}
        user_counts: Dict[str, int] = {}
        
        for ds_idx, (dataset, session_name) in enumerate(
            zip(self.datasets, self.session_names)
        ):
            user_id = self.session_to_user.get(session_name, "unknown")
            
            if user_id not in pools:
                pools[user_id] = []
                user_counts[user_id] = 0
            
            # Sample from this dataset for calibration pool
            n_samples = min(
                self.calibration_pool_size - user_counts[user_id],
                len(dataset),
            )
            
            if n_samples > 0:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
                for idx in indices:
                    sample = dataset[idx]
                    pools[user_id].append(sample["emg"])
                user_counts[user_id] += n_samples
        
        return pools
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx, user_id = self.index_map[idx]
        
        # Get base sample
        sample = self.datasets[ds_idx][local_idx]
        
        # Add user info
        sample["user_id"] = user_id
        sample["session_name"] = self.session_names[ds_idx]
        
        # Sample calibration data for this user
        cal_pool = self.calibration_pools.get(user_id, [])
        if len(cal_pool) >= self.calibration_k:
            cal_indices = np.random.choice(len(cal_pool), self.calibration_k, replace=False)
            calibration_emg = torch.stack([cal_pool[i] for i in cal_indices])
        else:
            # Fallback: duplicate samples if pool too small
            calibration_emg = torch.stack([sample["emg"]] * self.calibration_k)
        
        sample["calibration_emg"] = calibration_emg
        sample["calibration_k"] = self.calibration_k
        
        return sample


def create_dataloaders(
    data_dir: Optional[Path] = None,
    split: Optional[Dict[str, List[str]]] = None,
    window_length: int = 10_000,
    stride: int = 2_000,
    batch_size: int = 32,
    num_workers: int = 4,
    calibration_k: int = 5,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders.
    
    Args:
        data_dir: Path to emg2pose_dataset_mini. Defaults to ~/emg2pose_dataset_mini.
        split: Dictionary with train/val/test session lists. Defaults to MINI_SPLIT.
        window_length: Window size in samples.
        stride: Stride between windows.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        calibration_k: Number of calibration samples.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    split = split or MINI_SPLIT
    
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_dir}. Download with:\n"
            f"  cd ~ && curl '{DATASET_URL}' -o emg2pose_dataset_mini.tar\n"
            f"  tar -xvf emg2pose_dataset_mini.tar"
        )
    
    # Create datasets
    train_dataset = CalibratedEmgDataset(
        data_dir=data_dir,
        session_names=split["train"],
        window_length=window_length,
        stride=stride,
        jitter=True,
        calibration_k=calibration_k,
    )
    
    val_dataset = CalibratedEmgDataset(
        data_dir=data_dir,
        session_names=split["val"],
        window_length=window_length,
        stride=stride,
        jitter=False,
        calibration_k=calibration_k,
    )
    
    test_dataset = CalibratedEmgDataset(
        data_dir=data_dir,
        session_names=split["test"],
        window_length=window_length,
        stride=stride,
        jitter=False,
        calibration_k=calibration_k,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


def create_dummy_batch(
    batch_size: int = 4,
    emg_channels: int = 16,
    num_joints: int = 20,
    seq_length: int = 10_000,
    calibration_k: int = 5,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a single dummy batch for quick testing (without real data).
    
    Useful for testing model forward pass without loading actual data.
    
    Args:
        batch_size: Batch size.
        emg_channels: Number of EMG channels.
        num_joints: Number of pose joints.
        seq_length: Sequence length.
        calibration_k: Number of calibration samples.
        device: Target device.
    
    Returns:
        Batch dictionary matching real data format.
    """
    batch = {
        "emg": torch.randn(batch_size, emg_channels, seq_length, device=device),
        "joint_angles": torch.randn(batch_size, num_joints, seq_length, device=device),
        "calibration_emg": torch.randn(batch_size, calibration_k, emg_channels, seq_length, device=device),
        "calibration_k": torch.full((batch_size,), calibration_k, device=device),
        "no_ik_failure": torch.ones(batch_size, seq_length, dtype=torch.bool, device=device),
    }
    return batch


def check_dataset_available(data_dir: Optional[Path] = None) -> bool:
    """Check if emg2pose_dataset_mini is available."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    return data_dir.exists() and any(data_dir.glob("*.hdf5"))


def download_dataset_instructions() -> str:
    """Return instructions for downloading the dataset."""
    return f"""
emg2pose_dataset_mini not found. To download:

    cd ~
    curl "{DATASET_URL}" -o emg2pose_dataset_mini.tar
    tar -xvf emg2pose_dataset_mini.tar

This will create ~/emg2pose_dataset_mini/ with HDF5 session files.
"""

