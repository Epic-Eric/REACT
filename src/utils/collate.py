# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Collate functions for REACT-EMG DataLoaders.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch


def collate_calibrated_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for ``PrebuiltCalibratedDataset``.

    Each item already carries ``max_k`` calibration windows (no padding).  A
    single random *k* is chosen here for the whole batch and the calibration
    tensor is sliced to that *k*.  This eliminates wasted encoder compute on
    zero-padded slots and keeps batch dimensions uniform without masking.
    """
    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])
    calibration_emg = torch.stack(
        [item["calibration_emg"] for item in batch]
    )  # (B, max_k, 16, L)

    # Pick ONE k for this iteration — all samples use the same k, no padding.
    max_k = calibration_emg.shape[1]
    k = int(np.random.randint(1, max_k + 1)) if max_k > 1 else max_k
    calibration_emg = calibration_emg[:, :k, :, :]  # (B, k, 16, L)

    return {
        "emg": emg,  # (B, 16, L)
        "joint_angles": joint_angles,  # (B, 20, L)
        "no_ik_failure": no_ik_failure,  # (B, L)
        "calibration_emg": calibration_emg,  # (B, k, 16, L)
        "calibration_k": torch.full(
            (len(batch),), k, dtype=torch.long
        ),  # (B,) all equal k
        "user_id": [item["user_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
    }


# Backwards-compatible alias
collate_with_variable_calibration = collate_calibrated_batch
