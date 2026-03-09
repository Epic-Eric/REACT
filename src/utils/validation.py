# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Parallel HDF5 session validation for REACT-EMG.

Uses ``ProcessPoolExecutor`` to validate many session files concurrently,
checking that each file exists and produces at least one valid window.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from emg2pose.data import WindowedEmgDataset


def _validate_single_session(args: Tuple) -> Optional[Tuple[str, str, int]]:
    """Validate a single session file.  Worker function for parallel processing.

    Args:
        args: ``(data_dir, filename, user_id, window_length, stride, skip_ik_failures)``

    Returns:
        ``(filename, user_id, num_windows)`` if valid, ``None`` otherwise.
    """
    data_dir, filename, user_id, window_length, stride, skip_ik_failures = args

    hdf5_path = Path(data_dir) / f"{filename}.hdf5"
    if not hdf5_path.exists():
        return None

    try:
        dataset = WindowedEmgDataset(
            hdf5_path=hdf5_path,
            window_length=window_length,
            stride=stride,
            jitter=False,
            skip_ik_failures=skip_ik_failures,
        )

        num_windows = len(dataset)
        if num_windows == 0:
            return None

        # Release the HDF5 file handle
        if hasattr(dataset, "_session") and hasattr(dataset._session, "_file"):
            dataset._session._file.close()

        return (filename, user_id, num_windows)
    except Exception:
        return None


def parallel_validate_sessions(
    data_dir: Path,
    session_infos: List[Tuple[str, str]],
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    num_workers: int = 8,
    show_progress: bool = True,
) -> List[Tuple[str, str, int]]:
    """Validate sessions in parallel and return only the valid ones.

    Args:
        data_dir: Path to dataset directory containing HDF5 files.
        session_infos: List of ``(filename, user_id)`` tuples.
        window_length: Window size in samples.
        stride: Stride between windows.
        skip_ik_failures: Whether to skip IK failure regions.
        num_workers: Number of parallel workers.
        show_progress: Whether to show a ``tqdm`` progress bar.

    Returns:
        List of ``(filename, user_id, num_windows)`` for valid sessions.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    args_list = [
        (str(data_dir), filename, user_id, window_length, stride, skip_ik_failures)
        for filename, user_id in session_infos
    ]

    valid_sessions: List[Tuple[str, str, int]] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_validate_single_session, args): args[1]
            for args in args_list
        }

        iterator = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating sessions",
            disable=not show_progress,
        )

        for future in iterator:
            try:
                result = future.result(timeout=30)
                if result is not None:
                    valid_sessions.append(result)
            except Exception:
                continue

    return valid_sessions
