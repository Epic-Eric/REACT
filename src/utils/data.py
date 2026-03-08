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


# =============================================================================
# Validation Cache
# =============================================================================

def _cache_key(window_length: int, stride: int, skip_ik_failures: bool) -> str:
    """Compute a deterministic cache key from data parameters."""
    import hashlib
    raw = f"wl={window_length}_st={stride}_ik={skip_ik_failures}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _load_validation_cache(
    cache_dir: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
) -> Optional[Dict[str, int]]:
    """Load cached validation manifest mapping filename -> num_windows.
    
    Returns None if cache doesn't exist or is invalid.
    """
    import json
    key = _cache_key(window_length, stride, skip_ik_failures)
    cache_file = Path(cache_dir) / f"validation_manifest_{key}.json"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        # Sanity check
        if data.get("params") != {"window_length": window_length, "stride": stride, "skip_ik_failures": skip_ik_failures}:
            return None
        return data["sessions"]  # {filename: num_windows}
    except Exception:
        return None


def _save_validation_cache(
    cache_dir: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    valid_sessions: List[Tuple[str, str, int]],
) -> None:
    """Save validation results to cache.
    
    Stores a manifest mapping filename -> num_windows.
    """
    import json
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(window_length, stride, skip_ik_failures)
    cache_file = cache_dir / f"validation_manifest_{key}.json"
    data = {
        "params": {
            "window_length": window_length,
            "stride": stride,
            "skip_ik_failures": skip_ik_failures,
        },
        "sessions": {filename: num_windows for filename, user_id, num_windows in valid_sessions},
    }
    with open(cache_file, "w") as f:
        json.dump(data, f)
    print(f"Saved validation cache ({len(data['sessions'])} sessions) to {cache_file}")


def _load_calibration_cache(
    cache_dir: Path,
    split_name: str,
    window_length: int,
    stride: int,
    calibration_k: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load cached calibration pools for a split."""
    key = _cache_key(window_length, stride, True)
    cache_file = Path(cache_dir) / f"calibration_pools_{split_name}_{key}_k{calibration_k}.pt"
    if not cache_file.exists():
        return None
    try:
        pools = torch.load(cache_file, weights_only=False)
        print(f"Loaded calibration cache ({len(pools)} users) from {cache_file}")
        return pools
    except Exception:
        return None


def _save_calibration_cache(
    cache_dir: Path,
    split_name: str,
    window_length: int,
    stride: int,
    calibration_k: int,
    pools: Dict[str, torch.Tensor],
) -> None:
    """Save calibration pools to cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(window_length, stride, True)
    cache_file = cache_dir / f"calibration_pools_{split_name}_{key}_k{calibration_k}.pt"
    torch.save(pools, cache_file)
    print(f"Saved calibration cache ({len(pools)} users) to {cache_file}")


# =============================================================================
# Parallel Session Validation
# =============================================================================

def _validate_single_session(args: Tuple) -> Optional[Tuple[str, str, int]]:
    """Validate a single session file. Worker function for parallel processing.
    
    Args:
        args: Tuple of (data_dir, filename, user_id, window_length, stride, skip_ik_failures)
    
    Returns:
        (filename, user_id, num_windows) if valid, None otherwise.
    """
    data_dir, filename, user_id, window_length, stride, skip_ik_failures = args
    
    hdf5_path = Path(data_dir) / f"{filename}.hdf5"
    
    if not hdf5_path.exists():
        return None
    
    try:
        # Create dataset to check validity
        dataset = WindowedEmgDataset(
            hdf5_path=hdf5_path,
            window_length=window_length,
            stride=stride,
            jitter=False,  # Don't need jitter for validation
            skip_ik_failures=skip_ik_failures,
        )
        
        num_windows = len(dataset)
        if num_windows == 0:
            return None
        
        # Close the HDF5 file handle
        if hasattr(dataset, '_session') and hasattr(dataset._session, '_file'):
            dataset._session._file.close()
        
        return (filename, user_id, num_windows)
    except Exception:
        return None


def _parallel_validate_sessions(
    data_dir: Path,
    session_infos: List[Tuple[str, str]],
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    num_workers: int = 8,
    show_progress: bool = True,
) -> List[Tuple[str, str, int]]:
    """Validate sessions in parallel to filter out invalid ones.
    
    Args:
        data_dir: Path to dataset directory.
        session_infos: List of (filename, user_id) tuples.
        window_length: Window size in samples.
        stride: Stride between windows.
        skip_ik_failures: Whether to skip IK failure regions.
        num_workers: Number of parallel workers.
        show_progress: Whether to show progress bar.
    
    Returns:
        List of (filename, user_id, num_windows) for valid sessions.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm
    
    # Prepare arguments for workers
    args_list = [
        (str(data_dir), filename, user_id, window_length, stride, skip_ik_failures)
        for filename, user_id in session_infos
    ]
    
    valid_sessions = []
    
    # Use ProcessPoolExecutor for true parallelism (HDF5 is I/O bound)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_validate_single_session, args): args[1]  # filename
            for args in args_list
        }
        
        # Collect results with progress bar
        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating sessions",
            disable=not show_progress,
        )
        
        for future in iterator:
            try:
                result = future.result(timeout=30)  # 30s timeout per file
                if result is not None:
                    valid_sessions.append(result)
            except Exception:
                continue
    
    return valid_sessions


class PrebuiltCalibratedDataset(Dataset):
    """Pre-built EMG dataset with fixed-size calibration windows.
    
    Unlike LazyCalibratedEmgDataset, this class:
    1. Pre-builds all WindowedEmgDataset instances upfront (like emg2pose)
    2. Pre-filters sessions with no valid windows during setup
    3. Pre-computes calibration pools as fixed-size windows per user
    
    This is much faster for training since all HDF5 files are validated once
    during setup, not repeatedly during each epoch.
    
    Args:
        data_dir: Path to dataset directory containing HDF5 files.
        session_infos: List of (filename, user_id) tuples from metadata.
        window_length: Window size in samples (default: 10000 = 5s at 2kHz).
        stride: Stride between windows.
        jitter: Random window offset during training.
        skip_ik_failures: Skip windows with IK failures.
        calibration_k: Maximum calibration windows per user.
        min_calibration_k: Minimum calibration windows to sample per batch item.
        calibration_window_length: Length of each calibration window (defaults to window_length).
    """
    
    def __init__(
        self,
        data_dir: Path,
        session_infos: List[Tuple[str, str]],  # (filename, user_id) tuples
        window_length: int = 10_000,
        stride: int = 2_000,
        jitter: bool = False,
        skip_ik_failures: bool = True,
        calibration_k: int = 30,
        min_calibration_k: int = 0,
        calibration_window_length: Optional[int] = None,
        show_progress: bool = True,
        num_workers: int = 8,  # Parallel workers for loading
        cache_dir: Optional[Path] = None,  # Directory for validation/calibration caches
        split_name: str = "",  # e.g. "train", "val", "test" — used for calibration cache key
    ):
        from tqdm import tqdm
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        
        self.data_dir = Path(data_dir)
        self.window_length = window_length
        self.stride = stride
        self.jitter = jitter
        self.skip_ik_failures = skip_ik_failures
        self.max_calibration_k = calibration_k
        self.min_calibration_k = min_calibration_k
        self.calibration_window_length = calibration_window_length or window_length
        
        # Group sessions by user
        user_to_sessions: Dict[str, List[str]] = {}
        for filename, user_id in session_infos:
            if user_id not in user_to_sessions:
                user_to_sessions[user_id] = []
            user_to_sessions[user_id].append(filename)
        
        # Build WindowedEmgDataset for each session using parallel workers
        # This is done ONCE during setup, not during training
        self.datasets: List[WindowedEmgDataset] = []
        self.dataset_info: List[Tuple[str, str]] = []  # (filename, user_id)
        self.user_sessions: Dict[str, List[str]] = {}  # user -> valid session filenames
        
        # Step 1: Try cached validation, else run parallel validation
        cached_manifest = None
        if cache_dir is not None:
            cached_manifest = _load_validation_cache(
                cache_dir, window_length, stride, skip_ik_failures
            )
        
        if cached_manifest is not None:
            # Filter to only sessions in this split and look up num_windows from cache
            valid_sessions = []
            for filename, user_id in session_infos:
                num_windows = cached_manifest.get(filename)
                if num_windows is not None and num_windows > 0:
                    valid_sessions.append((filename, user_id, num_windows))
            print(f"Loaded validation cache: {len(valid_sessions)}/{len(session_infos)} sessions valid")
        else:
            print(f"Validating {len(session_infos)} sessions with {num_workers} workers...")
            valid_sessions = _parallel_validate_sessions(
                data_dir=self.data_dir,
                session_infos=session_infos,
                window_length=window_length,
                stride=stride,
                skip_ik_failures=skip_ik_failures,
                num_workers=num_workers,
                show_progress=show_progress,
            )
            # Save cache for next time (covers ALL sessions, not just this split)
            if cache_dir is not None:
                _save_validation_cache(
                    cache_dir, window_length, stride, skip_ik_failures, valid_sessions
                )
        
        failed_count = len(session_infos) - len(valid_sessions)
        
        # Step 2: Build datasets only for valid sessions (fast, sequential)
        print(f"Building {len(valid_sessions)} valid datasets...")
        iterator = tqdm(valid_sessions, desc="Creating datasets", disable=not show_progress)
        
        for filename, user_id, num_windows in iterator:
            hdf5_path = self.data_dir / f"{filename}.hdf5"
            
            try:
                dataset = WindowedEmgDataset(
                    hdf5_path=hdf5_path,
                    window_length=window_length,
                    stride=stride,
                    jitter=jitter,
                    skip_ik_failures=skip_ik_failures,
                )
                
                self.datasets.append(dataset)
                self.dataset_info.append((filename, user_id))
                
                if user_id not in self.user_sessions:
                    self.user_sessions[user_id] = []
                self.user_sessions[user_id].append(filename)
                
            except Exception:
                failed_count += 1
                continue
        
        if show_progress:
            print(f"Built {len(self.datasets)} valid datasets "
                  f"(skipped {failed_count} invalid)")
        
        if len(self.datasets) == 0:
            raise ValueError("No valid sessions found!")
        
        # Build global index mapping using ConcatDataset-style indexing
        self._build_index_mapping()
        
        # Pre-compute calibration pools per user (or load from cache)
        cached_pools = None
        if cache_dir is not None and split_name:
            cached_pools = _load_calibration_cache(
                cache_dir, split_name, window_length, stride, calibration_k
            )
        
        if cached_pools is not None:
            self.calibration_pools = cached_pools
            # Share in memory for DataLoader workers
            for pool in self.calibration_pools.values():
                pool.share_memory_()
            pool_sizes = [v.shape[0] for v in self.calibration_pools.values()]
            print(f"Calibration pools (cached): {len(self.calibration_pools)} users, "
                  f"avg {np.mean(pool_sizes):.1f} windows/user")
        else:
            self._build_calibration_pools(show_progress)
            # Save calibration cache for next time
            if cache_dir is not None and split_name:
                _save_calibration_cache(
                    cache_dir, split_name, window_length, stride,
                    calibration_k, self.calibration_pools
                )
        
        print(f"PrebuiltCalibratedDataset: {len(self.datasets)} sessions, "
              f"{len(self.user_sessions)} users, {len(self)} samples")
    
    def _build_index_mapping(self):
        """Build mapping from global index to (dataset_idx, local_idx, user_id)."""
        self.index_map: List[Tuple[int, int, str]] = []
        self.cumulative_sizes: List[int] = []
        
        cumsum = 0
        for ds_idx, (dataset, (filename, user_id)) in enumerate(
            zip(self.datasets, self.dataset_info)
        ):
            self.cumulative_sizes.append(cumsum)
            for local_idx in range(len(dataset)):
                self.index_map.append((ds_idx, local_idx, user_id))
            cumsum += len(dataset)
    
    def _build_calibration_pools(self, show_progress: bool = True):
        """Build calibration pools as fixed-size windows per user.
        
        Each user gets a pool of calibration windows of shape (pool_size, 16, L)
        where pool_size >= calibration_k and L = calibration_window_length.
        
        Windows are sampled from valid (no IK failure) windows across user's sessions.
        """
        from tqdm import tqdm
        
        self.calibration_pools: Dict[str, torch.Tensor] = {}
        
        # Target pool size per user (at least max_calibration_k)
        target_pool_size = max(self.max_calibration_k * 2, 50)
        
        # Build lookup: (filename, user_id) -> dataset index for O(1) lookup
        dataset_lookup: Dict[Tuple[str, str], int] = {
            (fn, uid): i for i, (fn, uid) in enumerate(self.dataset_info)
        }
        
        iterator = tqdm(
            self.user_sessions.items(), 
            desc="Building calibration pools",
            disable=not show_progress
        )
        
        for user_id, session_filenames in iterator:
            user_windows: List[torch.Tensor] = []
            
            # Shuffle sessions to get diversity
            shuffled_sessions = list(session_filenames)
            np.random.shuffle(shuffled_sessions)
            
            # Sample at least 1 window per session (up to 5) until we hit target
            for filename in shuffled_sessions:
                ds_idx = dataset_lookup.get((filename, user_id))
                
                if ds_idx is None:
                    continue
                
                dataset = self.datasets[ds_idx]
                n_windows = len(dataset)
                
                if n_windows == 0:
                    continue
                
                # Sample 1-5 windows per session (at least 1!)
                n_to_sample = min(max(1, 5), n_windows)
                
                indices = np.random.choice(n_windows, n_to_sample, replace=False)
                for idx in indices:
                    try:
                        sample = dataset[int(idx)]
                        emg = sample["emg"]  # Shape: (16, L)
                        
                        # Handle length mismatch if calibration_window_length differs
                        if emg.shape[1] >= self.calibration_window_length:
                            # Truncate to calibration length
                            emg = emg[:, :self.calibration_window_length]
                        else:
                            # Pad if too short (shouldn't happen normally)
                            pad_len = self.calibration_window_length - emg.shape[1]
                            emg = torch.nn.functional.pad(emg, (0, pad_len))
                        
                        user_windows.append(emg)
                    except Exception:
                        continue
                
                if len(user_windows) >= target_pool_size:
                    break
            
            # Stack into tensor or create empty pool
            if len(user_windows) > 0:
                self.calibration_pools[user_id] = torch.stack(user_windows)  # (N, 16, L)
            else:
                # Empty pool - will use zeros as fallback
                self.calibration_pools[user_id] = torch.zeros(
                    1, 16, self.calibration_window_length
                )
        
        # Share pool tensors in OS shared memory so DataLoader workers can
        # read them without pickling/copying the data on each __getitem__ call.
        for pool in self.calibration_pools.values():
            pool.share_memory_()
        
        if show_progress:
            pool_sizes = [v.shape[0] for v in self.calibration_pools.values()]
            print(f"Calibration pools: {len(self.calibration_pools)} users, "
                  f"avg {np.mean(pool_sizes):.1f} windows/user")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx, user_id = self.index_map[idx]
        
        # Get base sample from pre-built dataset
        sample = self.datasets[ds_idx][local_idx]
        filename = self.dataset_info[ds_idx][0]
        
        sample["user_id"] = user_id
        sample["session_name"] = filename
        
        # Always return max_k calibration windows — k is chosen once per batch
        # in the collate function, so no per-sample k sampling or padding needed.
        max_k = self.max_calibration_k
        cal_pool = self.calibration_pools.get(user_id)
        
        if cal_pool is not None and cal_pool.shape[0] > 0:
            pool_size = cal_pool.shape[0]
            if pool_size >= max_k:
                # Sample without replacement using torch.randperm (faster than np.random.choice)
                indices = torch.randperm(pool_size)[:max_k]
            else:
                # Pool smaller than max_k — sample with replacement
                indices = torch.randint(pool_size, (max_k,))
            cal_windows = cal_pool[indices]  # (max_k, 16, L) — always full, no padding
        else:
            cal_windows = torch.zeros(max_k, 16, self.calibration_window_length)
        
        sample["calibration_emg"] = cal_windows  # (max_k, 16, L)
        # No calibration_k field — k is uniform across the batch (set in collate)
        
        return sample


# Keep old name as alias for backwards compatibility
LazyCalibratedEmgDataset = PrebuiltCalibratedDataset


def collate_calibrated_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for PrebuiltCalibratedDataset.

    Each item already carries max_k calibration windows (no padding).  A single
    random k is chosen here for the whole batch and the calibration tensor is
    sliced to that k.  This eliminates wasted encoder compute on zero-padded
    slots and keeps batch dimensions uniform without any masking overhead.
    """
    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])
    calibration_emg = torch.stack([item["calibration_emg"] for item in batch])  # (B, max_k, 16, L)

    # Pick ONE k for this iteration — all samples use the same k, no padding.
    max_k = calibration_emg.shape[1]
    k = int(np.random.randint(1, max_k + 1)) if max_k > 1 else max_k
    calibration_emg = calibration_emg[:, :k, :, :]  # (B, k, 16, L)

    return {
        "emg": emg,  # (B, 16, L)
        "joint_angles": joint_angles,  # (B, 20, L)
        "no_ik_failure": no_ik_failure,  # (B, L)
        "calibration_emg": calibration_emg,  # (B, k, 16, L)  — k uniform across batch
        "calibration_k": torch.full((len(batch),), k, dtype=torch.long),  # (B,) — all equal k
        "user_id": [item["user_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
    }


def collate_with_variable_calibration(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """DEPRECATED: Use collate_calibrated_batch instead.
    
    Custom collate function for batches with variable-length calibration recordings.
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


def precompute_dataset_cache(
    data_dir: Path,
    metadata_df: "pd.DataFrame",
    cache_dir: Path,
    window_length: int = 10_000,
    stride: int = 2_000,
    skip_ik_failures: bool = True,
    num_workers: int = 8,
) -> None:
    """Pre-validate ALL sessions and save cache to persistent storage.
    
    Run this once (e.g. as a separate Modal function) to build the validation
    manifest.  Subsequent training runs will load the cache and skip the
    ~5-minute validation step entirely.
    
    Args:
        data_dir: Path to directory containing HDF5 files.
        metadata_df: pandas DataFrame with 'filename' and 'user' columns.
        cache_dir: Directory to write cache files.
        window_length: Window size in samples.
        stride: Stride between windows.
        skip_ik_failures: Skip windows with IK failures.
        num_workers: Number of parallel workers.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    all_infos = [
        (row["filename"], row["user"]) for _, row in metadata_df.iterrows()
    ]
    print(f"Pre-validating {len(all_infos)} sessions with {num_workers} workers...")
    
    valid_sessions = _parallel_validate_sessions(
        data_dir=data_dir,
        session_infos=all_infos,
        window_length=window_length,
        stride=stride,
        skip_ik_failures=skip_ik_failures,
        num_workers=num_workers,
        show_progress=True,
    )
    
    _save_validation_cache(
        cache_dir, window_length, stride, skip_ik_failures, valid_sessions
    )
    print(f"Done! {len(valid_sessions)}/{len(all_infos)} sessions valid.")


def create_lazy_datasets_from_metadata(
    data_dir: Path,
    metadata_df: "pd.DataFrame",  # pandas DataFrame with 'filename' and 'user' columns
    train_users: set,
    val_users: set,
    test_users: set,
    window_length: int = 10_000,
    stride: int = 2_000,
    calibration_k: int = 30,
    min_calibration_k: int = 0,
    cache_size: int = 100,  # Ignored, kept for backwards compatibility
    num_workers: int = 8,  # Parallel workers for loading
    cache_dir: Optional[Path] = None,  # Directory for validation/calibration caches
) -> Tuple["PrebuiltCalibratedDataset", "PrebuiltCalibratedDataset", "PrebuiltCalibratedDataset"]:
    """Create pre-built train/val/test datasets from metadata DataFrame.
    
    This is the recommended way to create datasets for large (25k+ files) datasets.
    Uses metadata.csv as a manifest to identify files, then pre-builds all datasets
    upfront (like emg2pose's ConcatDataset approach).
    
    If ``cache_dir`` is provided and a validation manifest exists there, the
    expensive per-file validation is skipped entirely.  Calibration pools are
    also cached per-split so the second run is almost instant.
    
    To build the cache for the first time, either:
      1. Pass ``cache_dir`` on the first training run (cache is written
         automatically after validation), or
      2. Call ``precompute_dataset_cache()`` separately (e.g. as a
         dedicated Modal function).
    
    Args:
        data_dir: Path to directory containing HDF5 files.
        metadata_df: pandas DataFrame with 'filename' and 'user' columns.
        train_users: Set of user IDs for training.
        val_users: Set of user IDs for validation.
        test_users: Set of user IDs for testing.
        window_length: Window size in samples.
        stride: Stride between windows.
        calibration_k: Maximum calibration windows per user.
        min_calibration_k: Minimum calibration windows to sample per batch item.
        cache_size: Ignored (kept for backwards compatibility).
        num_workers: Parallel workers for loading.
        cache_dir: Optional directory for validation/calibration caches.
                   Pass a path on the Modal persistent volume to persist across runs.
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    def get_session_infos(users: set) -> List[Tuple[str, str]]:
        filtered = metadata_df[metadata_df["user"].isin(users)]
        return [(row["filename"], row["user"]) for _, row in filtered.iterrows()]
    
    train_infos = get_session_infos(train_users)
    val_infos = get_session_infos(val_users)
    test_infos = get_session_infos(test_users)
    
    print(f"Creating datasets: train={len(train_infos)}, val={len(val_infos)}, test={len(test_infos)}")
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
        stride=stride * 2,  # Larger stride for validation
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

