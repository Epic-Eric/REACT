# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
User-Aware EMG Dataset.

Extends emg2pose's data loading with user tracking and calibration support.
Each sample includes user ID and session information for calibration sampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from .calibration import (
    CalibrationSampler,
    CalibrationConfig,
    UserSessionRegistry,
    BatchCalibrationSampler,
)


@dataclass
class UserAwareDatasetConfig:
    """Configuration for user-aware EMG dataset.
    
    Attributes:
        window_length: Length of each EMG window in samples.
        stride: Stride between consecutive windows.
        padding: (left, right) contextual padding.
        jitter: Whether to randomly jitter window offset.
        calibration: Calibration sampling configuration.
        skip_ik_failures: Skip windows with inverse kinematics failures.
        transform: Optional transform to apply to data.
    """
    window_length: int = 10000
    stride: int = 5000
    padding: tuple = (1790, 0)  # Match vemg2pose left context
    jitter: bool = True
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    skip_ik_failures: bool = False
    transform: Optional[Callable] = None


class UserAwareEmgDataset(Dataset):
    """EMG dataset with user tracking for calibration.
    
    Extends the standard windowed dataset to include user and session
    information needed for calibration sampling.
    """
    
    def __init__(
        self,
        hdf5_path: Path,
        config: UserAwareDatasetConfig,
        registry: Optional[UserSessionRegistry] = None,
    ):
        """Initialize dataset.
        
        Args:
            hdf5_path: Path to HDF5 session file.
            config: Dataset configuration.
            registry: User session registry for calibration.
        """
        self.hdf5_path = Path(hdf5_path)
        self.config = config
        self.registry = registry
        
        # Open file and load metadata
        self._file = h5py.File(self.hdf5_path, 'r')
        self._group = self._file['emg2pose']
        
        # Extract metadata
        self.user_id = self._get_attr('user', f'user_{hash(hdf5_path)}')
        self.session_name = self._get_attr('session', self.hdf5_path.stem)
        
        # Load timeseries
        self.timeseries = self._group['timeseries']
        self.session_length = len(self.timeseries)
        
        # Compute windows
        self.window_length = config.window_length
        self.stride = config.stride or config.window_length
        self.left_padding, self.right_padding = config.padding
        
        # Precompute valid windows
        self._blocks = self._compute_blocks()
        self._windows = self._precompute_windows()
        
        # Register with registry if provided
        if registry is not None:
            registry.register(
                session_name=self.session_name,
                user_id=self.user_id,
                hdf5_path=self.hdf5_path,
                dataset_idx=0,  # Will be updated by ConcatDataset
            )
    
    def _get_attr(self, key: str, default: str) -> str:
        """Get attribute from HDF5, handling bytes."""
        val = self._group.attrs.get(key, default)
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        return val
    
    def _compute_blocks(self) -> List[tuple]:
        """Compute valid time blocks for windowing."""
        if not self.config.skip_ik_failures:
            return [(0, self.session_length)]
        
        # Load IK failure mask
        joint_angles = self.timeseries['joint_angles'][:]
        no_ik_failure = ~np.any(joint_angles == 0, axis=1)
        
        # Find contiguous blocks of valid data
        blocks = []
        start = None
        for i, valid in enumerate(no_ik_failure):
            if valid and start is None:
                start = i
            elif not valid and start is not None:
                if i - start >= self.window_length:
                    blocks.append((start, i))
                start = None
        
        if start is not None and self.session_length - start >= self.window_length:
            blocks.append((start, self.session_length))
        
        return blocks if blocks else [(0, self.session_length)]
    
    def _get_block_len(self, block: tuple) -> int:
        """Get number of windows in a block."""
        return max(0, (block[1] - block[0] - self.window_length) // self.stride + 1)
    
    def _precompute_windows(self) -> List[tuple]:
        """Precompute window start positions."""
        windows = []
        cumsum = np.cumsum([0] + [self._get_block_len(b) for b in self._blocks])
        
        total_len = int(cumsum[-1])
        for idx in range(total_len):
            block_idx = np.searchsorted(cumsum, idx, 'right') - 1
            start_idx, end_idx = self._blocks[block_idx]
            relative_idx = idx - cumsum[block_idx]
            windows.append((start_idx + relative_idx * self.stride, end_idx))
        
        return windows
    
    def __len__(self) -> int:
        return len(self._windows)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a windowed sample.
        
        Returns dictionary with:
        - emg: (16, L) EMG tensor
        - joint_angles: (20, L) joint angle tensor
        - no_ik_failure: (L,) boolean mask
        - user_id: User identifier string
        - session_name: Session identifier string
        - window_idx: Window index in session
        """
        # Get window boundaries
        offset, end_idx = self._windows[idx]
        leftover = end_idx - (offset + self.window_length)
        
        # Apply jitter
        if leftover > 0 and self.config.jitter:
            offset += np.random.randint(0, min(self.stride, leftover))
        
        # Expand for padding
        window_start = max(offset - self.left_padding, 0)
        window_end = min(offset + self.window_length + self.right_padding, self.session_length)
        
        # Load data
        window = self.timeseries[window_start:window_end]
        
        emg = window['emg'].T.astype(np.float32)  # (16, L)
        joint_angles = window['joint_angles'].T.astype(np.float32)  # (20, L)
        
        # IK failure mask
        no_ik_failure = ~np.any(joint_angles == 0, axis=0)
        
        # Convert to tensors
        emg = torch.from_numpy(emg)
        joint_angles = torch.from_numpy(joint_angles)
        no_ik_failure = torch.from_numpy(no_ik_failure)
        
        # Apply transform if any
        if self.config.transform is not None:
            emg = self.config.transform(emg)
        
        return {
            'emg': emg,
            'joint_angles': joint_angles,
            'no_ik_failure': no_ik_failure,
            'user_id': self.user_id,
            'session_name': self.session_name,
            'window_idx': idx,
        }
    
    def close(self):
        """Close the HDF5 file."""
        self._file.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


class CalibratedEmgDataset(Dataset):
    """Wraps UserAwareEmgDataset to include calibration data.
    
    Each sample includes calibration recordings from the same user.
    """
    
    def __init__(
        self,
        base_dataset: UserAwareEmgDataset,
        calibration_sampler: CalibrationSampler,
        precompute: bool = False,
    ):
        """Initialize calibrated dataset.
        
        Args:
            base_dataset: Base user-aware dataset.
            calibration_sampler: Sampler for calibration data.
            precompute: Whether to precompute calibration assignments.
        """
        self.base_dataset = base_dataset
        self.sampler = calibration_sampler
        self.precompute = precompute
        
        # Optionally precompute calibration assignments
        self._cal_cache = {}
        if precompute:
            self._precompute_calibration()
    
    def _precompute_calibration(self):
        """Precompute which calibration sessions to use for each sample."""
        for idx in range(len(self.base_dataset)):
            session_name = self.base_dataset.session_name
            k = self.sampler.sample_k()
            sessions = self.sampler.sample_calibration_sessions(session_name, k)
            self._cal_cache[idx] = sessions
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample with calibration data."""
        # Get base sample
        sample = self.base_dataset[idx]
        
        # Get calibration data
        session_name = sample['session_name']
        
        if self.precompute and idx in self._cal_cache:
            cal_sessions = self._cal_cache[idx]
            k = len(cal_sessions)
        else:
            k = None  # Random k
        
        cal_emg, num_samples, lengths = self.sampler.get_calibration_batch(
            session_name, k=k
        )
        
        sample['calibration_emg'] = cal_emg
        sample['num_calibration_samples'] = num_samples
        sample['calibration_lengths'] = lengths
        
        return sample


def collate_with_calibration(
    batch: List[Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """Collate function that handles variable calibration data.
    
    Pads calibration tensors to uniform size across the batch.
    """
    batch_size = len(batch)
    
    # Find max calibration dimensions
    k_max = max(b['calibration_emg'].shape[0] for b in batch)
    l_max_main = max(b['emg'].shape[1] for b in batch)
    l_max_cal = max(b['calibration_emg'].shape[2] for b in batch)
    
    # Prepare output tensors
    emg = torch.zeros(batch_size, 16, l_max_main)
    joint_angles = torch.zeros(batch_size, 20, l_max_main)
    no_ik_failure = torch.zeros(batch_size, l_max_main, dtype=torch.bool)
    
    cal_emg = torch.zeros(batch_size, k_max, 16, l_max_cal)
    num_cal_samples = torch.zeros(batch_size, dtype=torch.long)
    cal_lengths = torch.zeros(batch_size, k_max, dtype=torch.long)
    
    user_ids = []
    session_names = []
    
    for i, sample in enumerate(batch):
        # Main data
        l_main = sample['emg'].shape[1]
        emg[i, :, :l_main] = sample['emg']
        joint_angles[i, :, :l_main] = sample['joint_angles']
        no_ik_failure[i, :l_main] = sample['no_ik_failure']
        
        # Calibration data
        k = sample['calibration_emg'].shape[0]
        l_cal = sample['calibration_emg'].shape[2]
        cal_emg[i, :k, :, :l_cal] = sample['calibration_emg']
        num_cal_samples[i] = sample['num_calibration_samples']
        
        k_actual = sample['calibration_lengths'].shape[0]
        cal_lengths[i, :k_actual] = sample['calibration_lengths']
        
        user_ids.append(sample['user_id'])
        session_names.append(sample['session_name'])
    
    return {
        'emg': emg,
        'joint_angles': joint_angles,
        'no_ik_failure': no_ik_failure,
        'calibration_emg': cal_emg,
        'num_calibration_samples': num_cal_samples,
        'calibration_lengths': cal_lengths,
        'user_ids': user_ids,
        'session_names': session_names,
    }


def create_user_aware_dataloaders(
    train_paths: List[Path],
    val_paths: List[Path],
    test_paths: List[Path],
    config: UserAwareDatasetConfig,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple:
    """Create dataloaders with user tracking and calibration.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, registry).
    """
    # Build registry
    all_paths = train_paths + val_paths + test_paths
    registry = UserSessionRegistry.from_session_files(all_paths)
    
    # Create calibration sampler
    cal_sampler = CalibrationSampler(config.calibration, registry)
    
    def make_dataset(paths: List[Path], jitter: bool) -> Dataset:
        datasets = []
        cfg = UserAwareDatasetConfig(
            window_length=config.window_length,
            stride=config.stride,
            padding=config.padding,
            jitter=jitter,
            calibration=config.calibration,
            skip_ik_failures=config.skip_ik_failures,
            transform=config.transform,
        )
        
        for path in paths:
            base_ds = UserAwareEmgDataset(path, cfg, registry)
            cal_ds = CalibratedEmgDataset(base_ds, cal_sampler)
            datasets.append(cal_ds)
        
        return ConcatDataset(datasets) if datasets else []
    
    train_dataset = make_dataset(train_paths, jitter=True)
    val_dataset = make_dataset(val_paths, jitter=False)
    test_dataset = make_dataset(test_paths, jitter=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_with_calibration,
        pin_memory=True,
    ) if len(train_dataset) > 0 else None
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_calibration,
        pin_memory=True,
    ) if len(val_dataset) > 0 else None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_with_calibration,
        pin_memory=True,
    ) if len(test_dataset) > 0 else None
    
    return train_loader, val_loader, test_loader, registry
