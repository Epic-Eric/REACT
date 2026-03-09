# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Validation and calibration cache I/O for REACT-EMG datasets.

Caches are keyed by (window_length, stride, skip_ik_failures) and stored as
JSON (validation manifests) or .pt files (calibration pools).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def cache_key(window_length: int, stride: int, skip_ik_failures: bool) -> str:
    """Compute a deterministic cache key from data parameters."""
    raw = f"wl={window_length}_st={stride}_ik={skip_ik_failures}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ── Validation manifest ─────────────────────────────────────────────────────

def load_validation_cache(
    cache_dir: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    split_name: str = "",
) -> Optional[Dict[str, int]]:
    """Load cached validation manifest mapping filename -> num_windows.

    Returns ``None`` if the cache file doesn't exist or fails sanity checks.
    """
    key = cache_key(window_length, stride, skip_ik_failures)
    prefix = f"validation_manifest_{split_name}_" if split_name else "validation_manifest_"
    cache_file = Path(cache_dir) / f"{prefix}{key}.json"

    if not cache_file.exists():
        return None
    try:
        with open(cache_file) as f:
            data = json.load(f)
        expected = {
            "window_length": window_length,
            "stride": stride,
            "skip_ik_failures": skip_ik_failures,
        }
        if data.get("params") != expected:
            return None
        return data["sessions"]  # {filename: num_windows}
    except Exception:
        return None


def save_validation_cache(
    cache_dir: Path,
    window_length: int,
    stride: int,
    skip_ik_failures: bool,
    valid_sessions: List[Tuple[str, str, int]],
    split_name: str = "",
) -> None:
    """Save validation results to a JSON cache file.

    ``valid_sessions`` is a list of ``(filename, user_id, num_windows)`` tuples.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(window_length, stride, skip_ik_failures)
    prefix = f"validation_manifest_{split_name}_" if split_name else "validation_manifest_"
    cache_file = cache_dir / f"{prefix}{key}.json"

    data = {
        "params": {
            "window_length": window_length,
            "stride": stride,
            "skip_ik_failures": skip_ik_failures,
        },
        "sessions": {
            filename: num_windows
            for filename, _user_id, num_windows in valid_sessions
        },
    }
    with open(cache_file, "w") as f:
        json.dump(data, f)
    print(f"Saved validation cache ({len(data['sessions'])} sessions) to {cache_file}")


# ── Calibration pools ────────────────────────────────────────────────────────

def load_calibration_cache(
    cache_dir: Path,
    split_name: str,
    window_length: int,
    stride: int,
    calibration_k: int,
) -> Optional[Dict[str, torch.Tensor]]:
    """Load cached per-user calibration pool tensors for a split."""
    key = cache_key(window_length, stride, True)
    cache_file = (
        Path(cache_dir) / f"calibration_pools_{split_name}_{key}_k{calibration_k}.pt"
    )
    if not cache_file.exists():
        return None
    try:
        pools = torch.load(cache_file, weights_only=False)
        print(f"Loaded calibration cache ({len(pools)} users) from {cache_file}")
        return pools
    except Exception:
        return None


def save_calibration_cache(
    cache_dir: Path,
    split_name: str,
    window_length: int,
    stride: int,
    calibration_k: int,
    pools: Dict[str, torch.Tensor],
) -> None:
    """Save per-user calibration pool tensors to a ``.pt`` file."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = cache_key(window_length, stride, True)
    cache_file = (
        cache_dir / f"calibration_pools_{split_name}_{key}_k{calibration_k}.pt"
    )
    torch.save(pools, cache_file)
    print(f"Saved calibration cache ({len(pools)} users) to {cache_file}")
