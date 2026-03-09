# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
EMG dataset classes for REACT-EMG.

* ``CalibratedEmgDataset``  – lightweight wrapper for the mini dataset.
* ``PrebuiltCalibratedDataset`` – production dataset for the full 25 k-file
  emg2pose dataset, with validation caching, calibration pools, and
  shared-memory tensors for fast DataLoader access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from emg2pose.data import WindowedEmgDataset

from .cache import (
    load_calibration_cache,
    load_validation_cache,
    save_calibration_cache,
    save_validation_cache,
)
from .validation import parallel_validate_sessions


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_session_path(data_dir: Path, session_name: str) -> Path:
    """Get full path to session HDF5 file."""
    return data_dir / f"{session_name}.hdf5"


def get_user_from_session(session_name: str) -> str:
    """Extract user ID from session name (4th dash-separated segment)."""
    parts = session_name.split("-")
    return parts[3] if len(parts) >= 4 else "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# CalibratedEmgDataset  (mini / local)
# ─────────────────────────────────────────────────────────────────────────────

class CalibratedEmgDataset(Dataset):
    """EMG dataset with calibration samples for user adaptation.

    Wraps ``WindowedEmgDataset`` and adds calibration sample retrieval.
    Each sample gets a random number of calibration samples (1 to ``max_k``).

    Designed for the *mini* dataset or local experiments with few sessions.
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
        min_calibration_k: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.session_names = session_names
        self.max_calibration_k = calibration_k
        self.min_calibration_k = min_calibration_k
        self.calibration_pool_size = calibration_pool_size

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
            self.session_to_user[session_name] = get_user_from_session(session_name)

        self._build_index_mapping()
        self.calibration_pools = self._build_calibration_pools()

    # ─── internals ───────────────────────────────────────────────────────

    def _build_index_mapping(self):
        self.index_map: List[Tuple[int, int, str]] = []
        for ds_idx, (dataset, session_name) in enumerate(
            zip(self.datasets, self.session_names)
        ):
            user_id = self.session_to_user.get(session_name, "unknown")
            for local_idx in range(len(dataset)):
                self.index_map.append((ds_idx, local_idx, user_id))

    def _build_calibration_pools(self) -> Dict[str, List[torch.Tensor]]:
        pools: Dict[str, List[torch.Tensor]] = {}
        user_counts: Dict[str, int] = {}

        for ds_idx, (dataset, session_name) in enumerate(
            zip(self.datasets, self.session_names)
        ):
            user_id = self.session_to_user.get(session_name, "unknown")
            if user_id not in pools:
                pools[user_id] = []
                user_counts[user_id] = 0

            n_samples = min(
                self.calibration_pool_size - user_counts[user_id],
                len(dataset),
            )
            if n_samples > 0:
                indices = np.random.choice(len(dataset), n_samples, replace=False)
                for idx in indices:
                    pools[user_id].append(dataset[idx]["emg"])
                user_counts[user_id] += n_samples
        return pools

    # ─── public API ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx, user_id = self.index_map[idx]
        sample = self.datasets[ds_idx][local_idx]
        sample["user_id"] = user_id
        sample["session_name"] = self.session_names[ds_idx]

        cal_pool = self.calibration_pools.get(user_id, [])
        pool_size = len(cal_pool)

        max_k = min(self.max_calibration_k, pool_size) if pool_size > 0 else self.max_calibration_k
        min_k = min(self.min_calibration_k, max_k)
        actual_k = np.random.randint(min_k, max_k + 1)

        if pool_size >= actual_k:
            cal_indices = np.random.choice(pool_size, actual_k, replace=False)
            cal_samples = [cal_pool[i] for i in cal_indices]
        else:
            cal_samples = [sample["emg"]] * actual_k

        emg_shape = sample["emg"].shape
        padded_calibration = torch.zeros(self.max_calibration_k, *emg_shape)
        for i, cal_emg in enumerate(cal_samples):
            padded_calibration[i] = cal_emg

        sample["calibration_emg"] = padded_calibration
        sample["calibration_k"] = actual_k
        return sample


# ─────────────────────────────────────────────────────────────────────────────
# PrebuiltCalibratedDataset  (full dataset / Modal)
# ─────────────────────────────────────────────────────────────────────────────

class PrebuiltCalibratedDataset(Dataset):
    """Pre-built EMG dataset with fixed-size calibration windows.

    1. Pre-builds all ``WindowedEmgDataset`` instances upfront.
    2. Pre-filters sessions with no valid windows during setup.
    3. Pre-computes calibration pools as fixed-size tensors per user.

    Validation results and calibration pools are cached to disk when
    ``cache_dir`` is provided, so subsequent runs skip the expensive I/O.
    """

    def __init__(
        self,
        data_dir: Path,
        session_infos: List[Tuple[str, str]],
        window_length: int = 10_000,
        stride: int = 2_000,
        jitter: bool = False,
        skip_ik_failures: bool = True,
        calibration_k: int = 30,
        min_calibration_k: int = 0,
        calibration_window_length: Optional[int] = None,
        show_progress: bool = True,
        num_workers: int = 8,
        cache_dir: Optional[Path] = None,
        split_name: str = "",
    ):
        from tqdm import tqdm

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
            user_to_sessions.setdefault(user_id, []).append(filename)

        self.datasets: List[WindowedEmgDataset] = []
        self.dataset_info: List[Tuple[str, str]] = []
        self.user_sessions: Dict[str, List[str]] = {}

        # ── Step 1: validate sessions (cached or parallel) ───────────────
        cached_manifest = None
        if cache_dir is not None:
            cached_manifest = load_validation_cache(
                cache_dir, window_length, stride, skip_ik_failures,
                split_name=split_name,
            )

        if cached_manifest is not None:
            valid_sessions = []
            for filename, user_id in session_infos:
                num_windows = cached_manifest.get(filename)
                if num_windows is not None and num_windows > 0:
                    valid_sessions.append((filename, user_id, num_windows))
            print(
                f"Loaded validation cache: "
                f"{len(valid_sessions)}/{len(session_infos)} sessions valid"
            )
        else:
            print(f"Validating {len(session_infos)} sessions with {num_workers} workers...")
            valid_sessions = parallel_validate_sessions(
                data_dir=self.data_dir,
                session_infos=session_infos,
                window_length=window_length,
                stride=stride,
                skip_ik_failures=skip_ik_failures,
                num_workers=num_workers,
                show_progress=show_progress,
            )
            if cache_dir is not None:
                save_validation_cache(
                    cache_dir, window_length, stride, skip_ik_failures,
                    valid_sessions, split_name=split_name,
                )

        failed_count = len(session_infos) - len(valid_sessions)

        # ── Step 2: build datasets ───────────────────────────────────────
        print(f"Building {len(valid_sessions)} valid datasets...")
        iterator = tqdm(
            valid_sessions, desc="Creating datasets", disable=not show_progress
        )

        for filename, user_id, _num_windows in iterator:
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
                self.user_sessions.setdefault(user_id, []).append(filename)
            except Exception:
                failed_count += 1

        if show_progress:
            print(
                f"Built {len(self.datasets)} valid datasets "
                f"(skipped {failed_count} invalid)"
            )

        if not self.datasets:
            raise ValueError("No valid sessions found!")

        self._build_index_mapping()

        # ── Step 3: calibration pools (cached or built) ──────────────────
        cached_pools = None
        if cache_dir is not None and split_name:
            cached_pools = load_calibration_cache(
                cache_dir, split_name, window_length, stride, calibration_k
            )

        if cached_pools is not None:
            self.calibration_pools = cached_pools
            pool_sizes = [v.shape[0] for v in self.calibration_pools.values()]
            print(
                f"Calibration pools (cached): {len(self.calibration_pools)} users, "
                f"avg {np.mean(pool_sizes):.1f} windows/user"
            )
        else:
            self._build_calibration_pools(show_progress)
            if cache_dir is not None and split_name:
                save_calibration_cache(
                    cache_dir, split_name, window_length, stride,
                    calibration_k, self.calibration_pools,
                )

        print(
            f"PrebuiltCalibratedDataset: {len(self.datasets)} sessions, "
            f"{len(self.user_sessions)} users, {len(self)} samples"
        )

    # ─── internals ───────────────────────────────────────────────────────

    def _build_index_mapping(self):
        self.index_map: List[Tuple[int, int, str]] = []
        self.cumulative_sizes: List[int] = []
        cumsum = 0
        for ds_idx, (dataset, (_, user_id)) in enumerate(
            zip(self.datasets, self.dataset_info)
        ):
            self.cumulative_sizes.append(cumsum)
            for local_idx in range(len(dataset)):
                self.index_map.append((ds_idx, local_idx, user_id))
            cumsum += len(dataset)

    def _build_calibration_pools(self, show_progress: bool = True):
        from tqdm import tqdm

        self.calibration_pools: Dict[str, torch.Tensor] = {}
        target_pool_size = max(self.max_calibration_k * 2, 50)

        dataset_lookup: Dict[Tuple[str, str], int] = {
            (fn, uid): i for i, (fn, uid) in enumerate(self.dataset_info)
        }

        iterator = tqdm(
            self.user_sessions.items(),
            desc="Building calibration pools",
            disable=not show_progress,
        )

        for user_id, session_filenames in iterator:
            user_windows: List[torch.Tensor] = []
            shuffled = list(session_filenames)
            np.random.shuffle(shuffled)

            for filename in shuffled:
                ds_idx = dataset_lookup.get((filename, user_id))
                if ds_idx is None:
                    continue
                dataset = self.datasets[ds_idx]
                n_windows = len(dataset)
                if n_windows == 0:
                    continue

                n_to_sample = min(5, n_windows)
                indices = np.random.choice(n_windows, n_to_sample, replace=False)
                for idx in indices:
                    try:
                        emg = dataset[int(idx)]["emg"]
                        if emg.shape[1] >= self.calibration_window_length:
                            emg = emg[:, : self.calibration_window_length]
                        else:
                            pad = self.calibration_window_length - emg.shape[1]
                            emg = torch.nn.functional.pad(emg, (0, pad))
                        user_windows.append(emg)
                    except Exception:
                        continue

                if len(user_windows) >= target_pool_size:
                    break

            if user_windows:
                self.calibration_pools[user_id] = torch.stack(user_windows)
            else:
                self.calibration_pools[user_id] = torch.zeros(
                    1, 16, self.calibration_window_length
                )

        if show_progress:
            pool_sizes = [v.shape[0] for v in self.calibration_pools.values()]
            print(
                f"Calibration pools: {len(self.calibration_pools)} users, "
                f"avg {np.mean(pool_sizes):.1f} windows/user"
            )

    # ─── public API ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ds_idx, local_idx, user_id = self.index_map[idx]
        sample = self.datasets[ds_idx][local_idx]
        filename = self.dataset_info[ds_idx][0]

        sample["user_id"] = user_id
        sample["session_name"] = filename

        max_k = self.max_calibration_k
        cal_pool = self.calibration_pools.get(user_id)

        if cal_pool is not None and cal_pool.shape[0] > 0:
            pool_size = cal_pool.shape[0]
            if pool_size >= max_k:
                indices = torch.randperm(pool_size)[:max_k]
            else:
                indices = torch.randint(pool_size, (max_k,))
            cal_windows = cal_pool[indices]
        else:
            # Use encoded feature shape if pools have been pre-encoded,
            # otherwise fall back to raw EMG shape.
            if hasattr(self, '_encoded_cal_shape') and self._encoded_cal_shape is not None:
                cal_windows = torch.zeros(max_k, *self._encoded_cal_shape)
            else:
                cal_windows = torch.zeros(
                    max_k, 16, self.calibration_window_length
                )

        sample["calibration_emg"] = cal_windows
        return sample


# Backwards-compatible alias
LazyCalibratedEmgDataset = PrebuiltCalibratedDataset
