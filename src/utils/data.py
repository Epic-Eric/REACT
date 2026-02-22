# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Data utilities for REACT-EMG using emg2pose_dataset_mini.

Wraps the emg2pose dataset with calibration sampling for user adaptation.
Includes lazy loading support for large datasets (25k+ files).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
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


# Default dataset location - look in src/data/ relative to this file
DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "emg2pose_dataset_mini"
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
    Each sample gets a random number of calibration samples (1 to max_k).
    """
    
    def __init__(
        self,
        data_dir: Path,
        session_names: List[str],
        window_length: int = 10_000,
        stride: int = 2_000,
        jitter: bool = False,
        skip_ik_failures: bool = True,
        calibration_k: int = 5,  # Max K (actual K sampled randomly per item)
        calibration_pool_size: int = 50,
        min_calibration_k: int = 1,  # Minimum K to sample
    ):
        """Initialize dataset.
        
        Args:
            data_dir: Path to emg2pose_dataset_mini directory.
            session_names: List of session names to include.
            window_length: Window size in samples.
            stride: Stride between windows.
            jitter: Random offset during training.
            skip_ik_failures: Skip windows with IK failures.
            calibration_k: Maximum calibration samples per batch item.
            calibration_pool_size: Size of calibration pool per user.
            min_calibration_k: Minimum calibration samples to sample.
        """
        self.data_dir = Path(data_dir)
        self.session_names = session_names
        self.max_calibration_k = calibration_k  # Max K for padding
        self.min_calibration_k = min_calibration_k
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
        
        # Sample RANDOM K for this item (between min and max)
        cal_pool = self.calibration_pools.get(user_id, [])
        pool_size = len(cal_pool)
        
        # Determine actual K for this sample (random within bounds)
        max_k = min(self.max_calibration_k, pool_size) if pool_size > 0 else self.max_calibration_k
        min_k = min(self.min_calibration_k, max_k)
        actual_k = np.random.randint(min_k, max_k + 1)  # Random K for THIS sample
        
        # Sample calibration recordings for this user
        if pool_size >= actual_k:
            cal_indices = np.random.choice(pool_size, actual_k, replace=False)
            cal_samples = [cal_pool[i] for i in cal_indices]
        else:
            # Fallback: use current sample if pool too small
            cal_samples = [sample["emg"]] * actual_k
        
        # Pad to max_calibration_k with zeros (for batching)
        emg_shape = sample["emg"].shape  # (16, L)
        padded_calibration = torch.zeros(self.max_calibration_k, *emg_shape)
        for i, cal_emg in enumerate(cal_samples):
            padded_calibration[i] = cal_emg
        
        sample["calibration_emg"] = padded_calibration  # (max_k, 16, L)
        sample["calibration_k"] = actual_k  # Actual number of valid samples (varies per item!)
        
        return sample


class LazyCalibratedEmgDataset(Dataset):
    """Lazy-loading EMG dataset for large datasets (25k+ files).
    
    Unlike CalibratedEmgDataset, this class:
    1. Uses metadata as a manifest (doesn't scan filesystem)
    2. Loads HDF5 files only on demand with LRU caching
    3. Builds calibration pools lazily
    
    This prevents heartbeat timeouts when working with large datasets.
    
    Args:
        data_dir: Path to dataset directory containing HDF5 files.
        session_infos: List of (filename, user_id) tuples from metadata.
        window_length: Window size in samples (default: 10000 = 5s at 2kHz).
        stride: Stride between windows.
        jitter: Random window offset during training.
        skip_ik_failures: Skip windows with IK failures.
        calibration_k: Maximum calibration samples per batch item.
        min_calibration_k: Minimum calibration samples per batch item.
        estimated_windows_per_session: Approximate windows per session for sizing.
        cache_size: Number of sessions to keep in LRU cache.
    """
    
    def __init__(
        self,
        data_dir: Path,
        session_infos: List[Tuple[str, str]],  # (filename, user_id) tuples
        window_length: int = 10_000,
        stride: int = 2_000,
        jitter: bool = False,
        skip_ik_failures: bool = True,
        calibration_k: int = 5,
        min_calibration_k: int = 1,
        estimated_windows_per_session: int = 50,  # Typical session ~100k samples
        cache_size: int = 100,
        validate_files: bool = False,  # Skip file existence checks for speed
    ):
        self.data_dir = Path(data_dir)
        self.window_length = window_length
        self.stride = stride
        self.jitter = jitter
        self.skip_ik_failures = skip_ik_failures
        self.max_calibration_k = calibration_k
        self.min_calibration_k = min_calibration_k
        
        # Build session infos - trust metadata by default (fast!)
        # File existence is checked lazily when loading
        self.session_infos: List[Tuple[str, str]] = []  # (filename, user_id)
        self.user_sessions: Dict[str, List[str]] = {}  # user -> [filenames]
        self._failed_files: set = set()  # Track files that failed to load
        
        for filename, user_id in session_infos:
            if validate_files:
                hdf5_path = self.data_dir / f"{filename}.hdf5"
                if not hdf5_path.exists():
                    continue
            
            self.session_infos.append((filename, user_id))
            if user_id not in self.user_sessions:
                self.user_sessions[user_id] = []
            self.user_sessions[user_id].append(filename)
        
        if len(self.session_infos) == 0:
            raise ValueError(f"No valid sessions found (got {len(session_infos)} from metadata)")
        
        print(f"LazyCalibratedEmgDataset: {len(self.session_infos)} sessions, "
              f"{len(self.user_sessions)} users")
        
        # Build index mapping: global_idx -> (session_idx, local_window_idx)
        # We estimate windows per session to avoid opening files
        self.estimated_windows = estimated_windows_per_session
        self._total_samples = len(self.session_infos) * self.estimated_windows
        
        # Setup LRU cache for session loading
        self._cache_size = cache_size
        self._session_cache: Dict[str, WindowedEmgDataset] = {}
        self._cache_order: List[str] = []  # LRU order
        
        # Calibration pool: lazily populated per user
        self._calibration_pools: Dict[str, List[torch.Tensor]] = {}
    
    def _get_session_dataset(self, filename: str) -> WindowedEmgDataset:
        """Get or create WindowedEmgDataset for a session (with LRU caching)."""
        if filename in self._failed_files:
            raise FileNotFoundError(f"Previously failed: {filename}")
        
        if filename in self._session_cache:
            # Move to end of LRU order
            self._cache_order.remove(filename)
            self._cache_order.append(filename)
            return self._session_cache[filename]
        
        # Load new session
        hdf5_path = self.data_dir / f"{filename}.hdf5"
        
        if not hdf5_path.exists():
            self._failed_files.add(filename)
            raise FileNotFoundError(f"File not found: {hdf5_path}")
        
        dataset = WindowedEmgDataset(
            hdf5_path=hdf5_path,
            window_length=self.window_length,
            stride=self.stride,
            jitter=self.jitter,
            skip_ik_failures=self.skip_ik_failures,
        )
        
        # Add to cache
        self._session_cache[filename] = dataset
        self._cache_order.append(filename)
        
        # Evict oldest if cache full
        while len(self._cache_order) > self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._session_cache[oldest]
        
        return dataset
    
    def _load_full_recording(self, filename: str) -> torch.Tensor:
        """Load full EMG recording (not windowed) for calibration.
        
        Returns:
            EMG tensor of shape (16, L) where L is full recording length.
        """
        hdf5_path = self.data_dir / f"{filename}.hdf5"
        
        if not hdf5_path.exists():
            raise FileNotFoundError(f"File not found: {hdf5_path}")
        
        try:
            # Load full recording directly from HDF5
            session = Emg2PoseSessionData(hdf5_path)
            emg = session.timeseries[Emg2PoseSessionData.EMG]  # Shape: (T, 16)
            
            if len(emg) == 0:
                raise ValueError("Empty EMG recording")
            
            emg_tensor = torch.as_tensor(emg, dtype=torch.float32).T  # Shape: (16, T)
            return emg_tensor
        except Exception as e:
            raise RuntimeError(f"Failed to load {filename}: {e}")
    
    def _get_calibration_recordings(self, user_id: str, k: int) -> List[torch.Tensor]:
        """Get k FULL calibration recordings for a user (variable lengths).
        
        Each recording is the entire EMG sequence, not windowed.
        
        Returns:
            List of k tensors, each with shape (16, Li) where Li varies.
        """
        user_sessions = self.user_sessions.get(user_id, [])
        
        if len(user_sessions) == 0:
            return []
        
        # Select k sessions (or fewer if not enough)
        if len(user_sessions) >= k:
            selected = np.random.choice(user_sessions, k, replace=False).tolist()
        else:
            # Use what we have, potentially repeat
            selected = np.random.choice(user_sessions, k, replace=True).tolist()
        
        recordings = []
        for sess_filename in selected:
            try:
                emg = self._load_full_recording(sess_filename)
                recordings.append(emg)
            except Exception as e:
                # Skip failed recordings
                continue
        
        return recordings
    
    def __len__(self) -> int:
        return self._total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Map global index to session
        session_idx = idx // self.estimated_windows
        local_idx = idx % self.estimated_windows
        
        # Handle wraparound if idx > actual samples
        session_idx = session_idx % len(self.session_infos)
        
        filename, user_id = self.session_infos[session_idx]
        
        try:
            dataset = self._get_session_dataset(filename)
            
            # Handle local_idx overflow
            if local_idx >= len(dataset):
                local_idx = local_idx % max(1, len(dataset))
            
            sample = dataset[local_idx]
        except Exception as e:
            # Fallback: return zeros if file can't be loaded
            print(f"Warning: Failed to load {filename}: {e}")
            sample = {
                "emg": torch.zeros(16, self.window_length),
                "joint_angles": torch.zeros(20, self.window_length),
                "no_ik_failure": torch.ones(self.window_length, dtype=torch.bool),
            }
        
        sample["user_id"] = user_id
        sample["session_name"] = filename
        
        # Sample K full calibration recordings (variable lengths!)
        max_k = self.max_calibration_k
        min_k = self.min_calibration_k
        actual_k = np.random.randint(min_k, max_k + 1) if max_k > min_k else max_k
        
        # Get full recordings (list of variable-length tensors)
        cal_recordings = self._get_calibration_recordings(user_id, actual_k)
        
        # Store as list (can't pad variable lengths into single tensor)
        sample["calibration_recordings"] = cal_recordings  # List of (16, Li) tensors
        sample["calibration_k"] = len(cal_recordings)  # Actual number retrieved
        
        return sample


def collate_with_variable_calibration(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batches with variable-length calibration recordings.
    
    Standard fields (emg, joint_angles, etc.) are stacked normally.
    calibration_recordings stays as a list of lists (one per batch item).
    """
    # Standard collation for fixed-size tensors
    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])
    calibration_k = torch.tensor([item["calibration_k"] for item in batch])
    
    # Keep calibration recordings as list of lists (variable lengths)
    calibration_recordings = [item["calibration_recordings"] for item in batch]
    
    return {
        "emg": emg,
        "joint_angles": joint_angles,
        "no_ik_failure": no_ik_failure,
        "calibration_recordings": calibration_recordings,  # List[List[Tensor]]
        "calibration_k": calibration_k,
        "user_id": [item["user_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
    }


def create_lazy_datasets_from_metadata(
    data_dir: Path,
    metadata_df: "pd.DataFrame",  # pandas DataFrame with 'filename' and 'user' columns
    train_users: set,
    val_users: set,
    test_users: set,
    window_length: int = 10_000,
    stride: int = 2_000,
    calibration_k: int = 5,
    min_calibration_k: int = 1,
    cache_size: int = 100,
) -> Tuple["LazyCalibratedEmgDataset", "LazyCalibratedEmgDataset", "LazyCalibratedEmgDataset"]:
    """Create lazy train/val/test datasets from metadata DataFrame.
    
    This is the recommended way to create datasets for large (25k+ files) datasets.
    Uses metadata.csv as a manifest instead of scanning the filesystem.
    
    Args:
        data_dir: Path to directory containing HDF5 files.
        metadata_df: pandas DataFrame with 'filename' and 'user' columns.
        train_users: Set of user IDs for training.
        val_users: Set of user IDs for validation.
        test_users: Set of user IDs for testing.
        window_length: Window size in samples.
        stride: Stride between windows.
        calibration_k: Maximum calibration samples.
        min_calibration_k: Minimum calibration samples.
        cache_size: LRU cache size for session datasets.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    def get_session_infos(users: set) -> List[Tuple[str, str]]:
        filtered = metadata_df[metadata_df["user"].isin(users)]
        return [(row["filename"], row["user"]) for _, row in filtered.iterrows()]
    
    train_infos = get_session_infos(train_users)
    val_infos = get_session_infos(val_users)
    test_infos = get_session_infos(test_users)
    
    print(f"Creating lazy datasets: train={len(train_infos)}, val={len(val_infos)}, test={len(test_infos)}")
    
    train_dataset = LazyCalibratedEmgDataset(
        data_dir=data_dir,
        session_infos=train_infos,
        window_length=window_length,
        stride=stride,
        jitter=True,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        cache_size=cache_size,
        validate_files=False,  # Trust metadata, check lazily
    )
    
    val_dataset = LazyCalibratedEmgDataset(
        data_dir=data_dir,
        session_infos=val_infos,
        window_length=window_length,
        stride=stride * 2,  # Larger stride for validation
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        cache_size=cache_size // 2,
        validate_files=False,
    )
    
    test_dataset = LazyCalibratedEmgDataset(
        data_dir=data_dir,
        session_infos=test_infos,
        window_length=window_length,
        stride=stride * 2,
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
        cache_size=cache_size // 2,
        validate_files=False,
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
    """Create train, validation, and test DataLoaders.
    
    Args:
        data_dir: Path to emg2pose_dataset_mini. Defaults to ~/emg2pose_dataset_mini.
        split: Dictionary with train/val/test session lists. Defaults to MINI_SPLIT.
        window_length: Window size in samples.
        stride: Stride between windows.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        calibration_k: Max number of calibration samples.
        min_calibration_k: Min number of calibration samples.
    
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
        min_calibration_k=min_calibration_k,
    )
    
    val_dataset = CalibratedEmgDataset(
        data_dir=data_dir,
        session_names=split["val"],
        window_length=window_length,
        stride=stride,
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
    )
    
    test_dataset = CalibratedEmgDataset(
        data_dir=data_dir,
        session_names=split["test"],
        window_length=window_length,
        stride=stride,
        jitter=False,
        calibration_k=calibration_k,
        min_calibration_k=min_calibration_k,
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

