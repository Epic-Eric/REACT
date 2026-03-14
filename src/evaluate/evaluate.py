# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Evaluation utilities for REACT-EMG.

Computes the same set of metrics as emg2pose (AngleMAE, AngularDerivatives,
PerFingerAngleMAE, PDAngleMAE, LandmarkDistances) on REACT model predictions.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def interpolate_predictions(
    pred: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Upsample predictions to match target temporal resolution.

    Same as ``emg2pose.pose_modules.BasePoseModule.align_predictions``.

    Args:
        pred: Model predictions (B, C, L').
        target_length: Desired temporal length.

    Returns:
        Interpolated predictions (B, C, target_length).
    """
    return F.interpolate(pred, size=target_length, mode="linear", align_corners=False)


def align_mask(
    mask: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Align IK-failure mask to match target length.

    Same as ``emg2pose.pose_modules.BasePoseModule.align_mask``.

    Args:
        mask: Boolean mask (B, T).
        target_length: Desired temporal length.

    Returns:
        Aligned boolean mask (B, target_length).
    """
    mask = mask[:, None].to(torch.float32)
    aligned = F.interpolate(mask, size=target_length, mode="nearest")
    return aligned.squeeze(1).to(torch.bool)


def pre_encode_calibration_pools(
    dataset,
    encoder: torch.nn.Module,
    device: torch.device,
    batch_size: int = 128,
) -> None:
    """Pre-encode all calibration pool windows with the frozen encoder.

    Replaces raw EMG tensors with encoded features in-place.
    """
    encoder.eval()
    total_windows = sum(p.shape[0] for p in dataset.calibration_pools.values())
    encoded_pools = {}
    with torch.no_grad():
        for user_id, pool in dataset.calibration_pools.items():
            encoded_chunks = []
            for i in range(0, pool.shape[0], batch_size):
                chunk = pool[i : i + batch_size].to(device)
                enc = encoder(chunk)
                encoded_chunks.append(enc.cpu())
            encoded_pools[user_id] = torch.cat(encoded_chunks, dim=0)
    dataset.calibration_pools = encoded_pools
    sample_pool = next(iter(encoded_pools.values()))
    dataset._encoded_cal_shape = tuple(sample_pool.shape[1:])
    print(
        f"  Pre-encoded {total_windows} calibration windows "
        f"-> feature shape {dataset._encoded_cal_shape}"
    )


def collate_fixed_k(batch: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """Collate function with a *fixed* k for evaluation.

    Unlike the training collate which samples k randomly, this always
    uses exactly ``k`` calibration windows per sample.
    """
    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])
    calibration_emg = torch.stack([item["calibration_emg"] for item in batch])

    max_k = calibration_emg.shape[1]
    actual_k = min(k, max_k)
    calibration_emg = calibration_emg[:, :actual_k, :, :]

    return {
        "emg": emg,
        "joint_angles": joint_angles,
        "no_ik_failure": no_ik_failure,
        "calibration_emg": calibration_emg,
        "calibration_k": torch.full((len(batch),), actual_k, dtype=torch.long),
        "user_id": [item["user_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
    }


def evaluate_react_emg(
    model,
    test_loader,
    device: torch.device,
    left_context: int = 0,
    right_context: int = 0,
    use_amp: bool = True,
    baseline_mode: bool = False,
    provide_initial_pos: bool = False,
) -> Dict[str, Any]:
    """Run evaluation computing full emg2pose metrics.

    Args:
        model: REACT FiLMConditionedModel or emg2pose StatePoseModule (eval mode, on device).
        test_loader: DataLoader producing batches.
        device: Compute device.
        left_context: Encoder left context (samples to trim from targets).
        right_context: Encoder right context.
        use_amp: Use automatic mixed precision.
        baseline_mode: If True, run emg2pose StatePoseModule forward pass
            (no calibration). Batches only need emg/joint_angles/no_ik_failure.
        provide_initial_pos: If True, extract initial position from
            joint_angles[:, :, left_context] and pass to model.

    Returns:
        Dictionary with aggregate and per-user metric results.
    """
    from emg2pose.metrics import get_default_metrics

    metrics_list = get_default_metrics()
    stage = "test"

    # Accumulators: list of (metric_dict, n_valid) per batch
    batch_metrics: List[Dict[str, float]] = []
    batch_counts: List[int] = []

    # Per-user accumulators
    user_batch_metrics: Dict[str, List[Dict[str, float]]] = {}
    user_batch_counts: Dict[str, List[int]] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            emg = batch["emg"].to(device, non_blocking=True)
            targets = batch["joint_angles"].to(device, non_blocking=True)
            no_ik_failure = batch["no_ik_failure"].to(device, non_blocking=True)
            user_ids = batch.get("user_id")

            try:
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    if baseline_mode:
                        # emg2pose BasePoseModule forward (StatePoseModule
                        # for tracking, VEMG2PoseWithInitialState for regression)
                        model_batch = {
                            "emg": emg,
                            "joint_angles": targets,
                            "no_ik_failure": no_ik_failure,
                        }
                        predictions, targets, no_ik_failure = model(
                            model_batch, provide_initial_pos=provide_initial_pos
                        )
                        # BasePoseModule.forward already trims targets/mask
                        # and aligns predictions, so skip the manual trim below
                        pred_aligned = predictions.float()
                        targets = targets.float()
                        mask_aligned = no_ik_failure
                    else:
                        cal_features = batch["calibration_emg"].to(device, non_blocking=True)
                        cal_k = batch["calibration_k"].to(device, non_blocking=True)
                        # Extract initial position for tracking mode
                        init_pos = None
                        if provide_initial_pos:
                            init_pos = targets[:, :, left_context]
                        predictions = model(
                            emg=emg,
                            calibration_features=cal_features,
                            num_calibration_samples=cal_k,
                            initial_pos=init_pos,
                        )
            except Exception as e:
                print(f"  Skipping batch due to error: {e}")
                continue

            if not baseline_mode:
                # Trim targets and mask for encoder context
                if left_context > 0 or right_context > 0:
                    start = left_context
                    end = -right_context if right_context > 0 else None
                    targets = targets[..., start:end]
                    no_ik_failure = no_ik_failure[..., start:end]

                # Interpolate predictions to match target temporal resolution
                target_len = targets.shape[-1]
                pred_aligned = interpolate_predictions(predictions, target_len)
                mask_aligned = no_ik_failure

                # Cast to float32 for metrics (AMP may produce float16;
                # LandmarkDistances forward_kinematics requires float32)
                pred_aligned = pred_aligned.float()
                targets = targets.float()

            # Number of valid time steps in this batch
            n_valid = int(mask_aligned.sum().item())
            if n_valid == 0:
                continue

            # Compute all emg2pose metrics for this batch
            m = {}
            for metric in metrics_list:
                try:
                    m.update(metric(pred_aligned, targets, mask_aligned, stage))
                except Exception as e:
                    print(f"  Metric {type(metric).__name__} failed: {e}")

            # Detach to CPU floats
            m_cpu = {k: v.detach().cpu().item() if torch.is_tensor(v) else float(v) for k, v in m.items()}
            batch_metrics.append(m_cpu)
            batch_counts.append(n_valid)

            # Per-user accumulation
            unique_users = set(user_ids)
            for uid in unique_users:
                mask_user = torch.tensor(
                    [u == uid for u in user_ids], dtype=torch.bool, device=device
                )
                n_user = int(mask_user.sum().item())
                if n_user == 0:
                    continue

                pred_u = pred_aligned[mask_user]
                tgt_u = targets[mask_user]
                ik_u = mask_aligned[mask_user]
                n_valid_u = int(ik_u.sum().item())
                if n_valid_u == 0:
                    continue

                m_u = {}
                for metric in metrics_list:
                    try:
                        m_u.update(metric(pred_u, tgt_u, ik_u, stage))
                    except Exception:
                        pass

                m_u_cpu = {k: v.detach().cpu().item() if torch.is_tensor(v) else float(v) for k, v in m_u.items()}

                if uid not in user_batch_metrics:
                    user_batch_metrics[uid] = []
                    user_batch_counts[uid] = []
                user_batch_metrics[uid].append(m_u_cpu)
                user_batch_counts[uid].append(n_valid_u)

    # ── Aggregate metrics (weighted mean by valid count) ──────────────
    if not batch_metrics:
        return {"error": "No valid batches processed"}

    all_keys = set()
    for m in batch_metrics:
        all_keys.update(m.keys())

    total_valid = sum(batch_counts)
    aggregate = {}
    for key in sorted(all_keys):
        weighted_sum = sum(
            m.get(key, 0.0) * c for m, c in zip(batch_metrics, batch_counts)
        )
        aggregate[key] = weighted_sum / total_valid

    # ── Per-user aggregate ───────────────────────────────────────────
    per_user = {}
    for uid in sorted(user_batch_metrics.keys()):
        u_metrics = user_batch_metrics[uid]
        u_counts = user_batch_counts[uid]
        u_total = sum(u_counts)
        u_agg = {}
        for key in sorted(all_keys):
            ws = sum(m.get(key, 0.0) * c for m, c in zip(u_metrics, u_counts))
            u_agg[key] = ws / u_total
        u_agg["num_valid_samples"] = u_total
        per_user[uid] = u_agg

    return {
        "aggregate": aggregate,
        "per_user": per_user,
        "num_batches": len(batch_metrics),
        "total_valid_timesteps": total_valid,
        "num_users": len(per_user),
    }
