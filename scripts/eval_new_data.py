#!/usr/bin/env python3
"""
Evaluate REACT and baseline models on NEW (held-out) data.

Uses the emg2pose_dataset_mini which contains 30 recordings from user
d387095792 — a held_out_user who was NEVER in the training set.
This satisfies the "evaluate on new data" requirement.

Evaluates:
  1. emg2pose baseline (tracking_vemg2pose) — no calibration
  2. emg2pose baseline (regression_vemg2pose) — no calibration
  3. REACT tracking model — with k calibration recordings
  4. REACT regression model — with k calibration recordings

Metrics computed:
  - AngleMAE (angular mean absolute error, in degrees)
  - Per-finger MAE
  - Fingertip landmark distance (mm)

Usage:
    python scripts/eval_new_data.py \
        --react-tracking checkpoints/best_model_tracking.pt \
        --react-regression checkpoints/best_model_regression.pt \
        --k 5

    # Baseline only (no REACT checkpoints yet):
    python scripts/eval_new_data.py --baseline-only

    # REACT tracking only:
    python scripts/eval_new_data.py \
        --react-tracking checkpoints/best_model_tracking.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "emg2pose"))
sys.path.insert(0, str(PROJECT_ROOT / "emg2pose" / "emg2pose" / "UmeTrack"))

from emg2pose.data import Emg2PoseSessionData, WindowedEmgDataset

# ─── Constants ───────────────────────────────────────────────────────────────

MINI_DATA_DIR = PROJECT_ROOT / "src" / "data" / "emg2pose_dataset_mini"
PRETRAINED_DIR = PROJECT_ROOT / "emg2pose_model_checkpoints"
OUTPUT_DIR = PROJECT_ROOT / "paper" / "final_report"
RAD2DEG = 180.0 / np.pi

# All 30 recordings in the mini dataset (user d387095792)
# We split: first 5 recordings (left+right) as calibration, rest as evaluation
WINDOW_LENGTH = 11790  # 10000 + 1790 (TDS left_context)
STRIDE = 4000


# ─── Metric computation ─────────────────────────────────────────────────────

def compute_angle_mae(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> float:
    """Compute mean absolute error in degrees over valid frames.

    Args:
        pred: (B, 20, T) in radians
        target: (B, 20, T) in radians
        mask: (B, T) boolean
    """
    mask_expanded = mask.unsqueeze(1).expand_as(pred)
    if not mask_expanded.any():
        return float("nan")
    diff = (pred - target).abs()
    mae_rad = diff[mask_expanded].mean().item()
    return mae_rad * RAD2DEG


def compute_per_finger_mae(pred: torch.Tensor, target: torch.Tensor,
                           mask: torch.Tensor) -> Dict[str, float]:
    """Compute MAE per finger in degrees.

    Joint mapping (20 DOF):
      Thumb: 0-3, Index: 4-7, Middle: 8-11, Ring: 12-15, Pinky: 16-19
    """
    finger_ranges = {
        "thumb": (0, 4), "index": (4, 8), "middle": (8, 12),
        "ring": (12, 16), "pinky": (16, 20),
    }
    mask_expanded = mask.unsqueeze(1)
    results = {}
    for finger, (start, end) in finger_ranges.items():
        finger_pred = pred[:, start:end, :]
        finger_tgt = target[:, start:end, :]
        finger_mask = mask_expanded.expand_as(finger_pred)
        if finger_mask.any():
            results[finger] = (finger_pred - finger_tgt).abs()[finger_mask].mean().item() * RAD2DEG
        else:
            results[finger] = float("nan")
    return results


def compute_landmark_distance(pred: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor) -> float:
    """Compute mean fingertip landmark distance in mm using forward kinematics."""
    try:
        from emg2pose.kinematics import forward_kinematics
    except ImportError:
        return float("nan")

    mask_expanded = mask.unsqueeze(1)
    # FK expects (B, 20, T) -> landmarks (B, T, 21, 3)
    with torch.no_grad():
        pred_landmarks = forward_kinematics(pred.float())
        tgt_landmarks = forward_kinematics(target.float())

    # Fingertip indices: 0-4
    pred_tips = pred_landmarks[:, :, :5, :]  # (B, T, 5, 3)
    tgt_tips = tgt_landmarks[:, :, :5, :]

    dist = (pred_tips - tgt_tips).norm(dim=-1)  # (B, T, 5)
    # Apply mask: (B, T) -> (B, T, 1)
    tip_mask = mask.unsqueeze(-1).expand_as(dist)
    if tip_mask.any():
        return dist[tip_mask].mean().item()
    return float("nan")


# ─── Session loading helpers ─────────────────────────────────────────────────

def get_mini_sessions():
    """Get all session filenames from the mini dataset."""
    hdf5_files = sorted(MINI_DATA_DIR.glob("*.hdf5"))
    return [f.stem for f in hdf5_files]


def split_calibration_eval(session_names, num_cal_recordings=5):
    """Split sessions into calibration and evaluation sets.

    Takes the first `num_cal_recordings` recording numbers for calibration,
    rest for evaluation. Each recording has left+right variants.
    """
    # Group by recording number
    recordings = {}
    for name in session_names:
        # e.g. "...-recording-10_left" -> recording number "10"
        parts = name.rsplit("-recording-", 1)
        if len(parts) == 2:
            rec_num = parts[1].split("_")[0]
            if rec_num not in recordings:
                recordings[rec_num] = []
            recordings[rec_num].append(name)

    sorted_recs = sorted(recordings.keys(), key=int)
    cal_recs = sorted_recs[:num_cal_recordings]
    eval_recs = sorted_recs[num_cal_recordings:]

    cal_sessions = [s for r in cal_recs for s in recordings[r]]
    eval_sessions = [s for r in eval_recs for s in recordings[r]]

    return cal_sessions, eval_sessions


# ─── Baseline evaluation ────────────────────────────────────────────────────

def evaluate_baseline(checkpoint_name: str, eval_sessions: list[str]) -> Dict[str, Any]:
    """Evaluate an emg2pose baseline model on eval sessions."""
    from emg2pose.utils import generate_hydra_config_from_overrides
    from emg2pose.lightning import Emg2PoseModule

    ckpt_path = str(PRETRAINED_DIR / checkpoint_name)
    experiment = "tracking_vemg2pose" if "tracking" in checkpoint_name else "regression_vemg2pose"

    config = generate_hydra_config_from_overrides(
        overrides=[f"experiment={experiment}", f"checkpoint={ckpt_path}"]
    )
    module = Emg2PoseModule.load_from_checkpoint(
        config.checkpoint,
        network=config.network,
        optimizer=config.optimizer,
        lr_scheduler=config.lr_scheduler,
    )
    module.eval()

    all_mae = []
    all_finger_mae = {f: [] for f in ["thumb", "index", "middle", "ring", "pinky"]}
    all_landmark_dist = []
    total_frames = 0

    for session_name in tqdm(eval_sessions, desc=f"Baseline ({checkpoint_name})"):
        hdf5_path = MINI_DATA_DIR / f"{session_name}.hdf5"
        session = Emg2PoseSessionData(hdf5_path=hdf5_path)

        # Process in windows
        ds = WindowedEmgDataset(
            hdf5_path=hdf5_path,
            window_length=WINDOW_LENGTH,
            stride=STRIDE,
            jitter=False,
            skip_ik_failures=True,
        )

        for i in range(len(ds)):
            sample = ds[i]
            emg = sample["emg"].unsqueeze(0)  # (1, 16, L)
            targets = sample["joint_angles"].unsqueeze(0)  # (1, 20, L)
            no_ik = sample["no_ik_failure"].unsqueeze(0)  # (1, L)

            batch = {"emg": emg, "joint_angles": targets, "no_ik_failure": no_ik}

            with torch.no_grad():
                preds, targets_out, mask_out = module.forward(batch)

            preds = preds.float()
            targets_out = targets_out.float()

            n_valid = mask_out.sum().item()
            if n_valid == 0:
                continue

            mae = compute_angle_mae(preds, targets_out, mask_out)
            finger_mae = compute_per_finger_mae(preds, targets_out, mask_out)
            landmark = compute_landmark_distance(preds, targets_out, mask_out)

            all_mae.append((mae, n_valid))
            for f, v in finger_mae.items():
                all_finger_mae[f].append((v, n_valid))
            all_landmark_dist.append((landmark, n_valid))
            total_frames += n_valid

    # Weighted average
    def weighted_avg(pairs):
        vals = [(v, n) for v, n in pairs if not np.isnan(v)]
        if not vals:
            return float("nan")
        total_n = sum(n for _, n in vals)
        return sum(v * n for v, n in vals) / total_n

    return {
        "angle_mae_deg": weighted_avg(all_mae),
        "per_finger_mae_deg": {f: weighted_avg(vs) for f, vs in all_finger_mae.items()},
        "fingertip_distance_mm": weighted_avg(all_landmark_dist),
        "total_valid_frames": total_frames,
        "num_eval_sessions": len(eval_sessions),
    }


# ─── REACT evaluation ───────────────────────────────────────────────────────

def load_react_model(checkpoint_path: str, device: torch.device):
    """Load a REACT model from checkpoint."""
    from src.models.hybrid_model import (
        FiLMConditionedModel,
        FiLMConditionedModelConfig,
        load_pretrained_encoder,
        load_pretrained_decoder,
    )
    from src.models.user_encoder import UserEncoderConfig
    from src.models.blocks.gru_pooling import GRUTemporalPoolingConfig

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # Resolve encoder architecture checkpoint
    raw_path = model_cfg.get("pretrained_checkpoint",
                              "emg2pose_model_checkpoints/regression_vemg2pose.ckpt")
    encoder_path = str(PRETRAINED_DIR / Path(raw_path).name)

    encoder = load_pretrained_encoder(encoder_path, device=str(device))
    decoder = load_pretrained_decoder(encoder_path, device=str(device))

    # User encoder config
    ue_yaml = model_cfg.get("user_encoder", {})
    pooling = ue_yaml.get("pooling_method", "attention")
    ue_config = UserEncoderConfig(pooling_method=pooling)
    if pooling == "gru":
        gru_yaml = ue_yaml.get("gru_pooling", {})
        if gru_yaml:
            ue_config.gru_pooling = GRUTemporalPoolingConfig(**gru_yaml)

    config_kwargs = dict(
        feature_dim=model_cfg.get("feature_dim", 64),
        user_embedding_dim=model_cfg.get("user_embedding_dim", 128),
        freeze_encoder=False,
        freeze_decoder=False,
        predict_vel=model_cfg.get("predict_vel", False),
        provide_initial_pos=model_cfg.get("provide_initial_pos", False),
        state_condition=model_cfg.get("state_condition", True),
        user_encoder=ue_config,
    )

    # Cross-attention support (dev branch only)
    try:
        from src.models.blocks.cross_attention import CrossAttentionConfig
        cond_method = model_cfg.get("conditioning_method", "film")
        ca_yaml = model_cfg.get("cross_attention", {})
        ca_config = CrossAttentionConfig(
            feature_dim=model_cfg.get("feature_dim", 64),
            **{k: type(getattr(CrossAttentionConfig, k, 0))(v)
               for k, v in ca_yaml.items() if k != "feature_dim"}
        ) if ca_yaml else CrossAttentionConfig(feature_dim=model_cfg.get("feature_dim", 64))
        config_kwargs["conditioning_method"] = cond_method
        config_kwargs["cross_attention"] = ca_config
        config_kwargs["contrastive_dim"] = int(model_cfg.get("contrastive_dim", 0))
    except ImportError:
        pass

    model_config = FiLMConditionedModelConfig(**config_kwargs)
    model = FiLMConditionedModel(model_config, pretrained_encoder=encoder,
                                  pretrained_decoder=decoder)

    raw_sd = ckpt["model_state_dict"]
    cleaned_sd = {re.sub(r"\._orig_mod\.", ".", k): v for k, v in raw_sd.items()}
    model.load_state_dict(cleaned_sd)
    model.to(device).eval()

    left_context = getattr(model.encoder, "left_context", 0)
    right_context = getattr(model.encoder, "right_context", 0)

    return model, model_config, left_context, right_context


def pre_encode_calibration(model, cal_sessions: list[str],
                           device: torch.device, k: int):
    """Encode calibration windows from calibration sessions.

    Returns:
        cal_features: (1, k, C, L') pre-encoded calibration tensor
    """
    # Collect windows from calibration sessions
    all_windows = []
    for session_name in cal_sessions:
        hdf5_path = MINI_DATA_DIR / f"{session_name}.hdf5"
        ds = WindowedEmgDataset(
            hdf5_path=hdf5_path, window_length=WINDOW_LENGTH,
            stride=STRIDE, jitter=False, skip_ik_failures=True,
        )
        for i in range(len(ds)):
            all_windows.append(ds[i]["emg"])  # (16, L)

    # Sample k windows
    if len(all_windows) < k:
        print(f"  Warning: only {len(all_windows)} calibration windows available, using all")
        k = len(all_windows)

    indices = np.random.choice(len(all_windows), k, replace=False)
    cal_emg = torch.stack([all_windows[i] for i in indices])  # (k, 16, L)

    # Encode through frozen encoder
    with torch.no_grad():
        cal_emg_dev = cal_emg.to(device)
        encoded = model.encode(cal_emg_dev)  # (k, C, L')

    cal_features = encoded.unsqueeze(0)  # (1, k, C, L')
    print(f"  Calibration: {k} windows encoded -> {cal_features.shape}")
    return cal_features, k


def evaluate_react(checkpoint_path: str, cal_sessions: list[str],
                   eval_sessions: list[str], k: int,
                   device: torch.device) -> Dict[str, Any]:
    """Evaluate a REACT model on eval sessions with calibration."""
    print(f"Loading REACT model: {checkpoint_path}")
    model, model_config, left_context, right_context = load_react_model(
        checkpoint_path, device
    )
    print(f"  Mode: {'tracking' if model_config.predict_vel else 'regression'}")
    print(f"  Encoder context: left={left_context}, right={right_context}")

    # Pre-encode calibration
    cal_features, actual_k = pre_encode_calibration(model, cal_sessions, device, k)
    cal_k = torch.tensor([actual_k], dtype=torch.long, device=device)

    all_mae = []
    all_finger_mae = {f: [] for f in ["thumb", "index", "middle", "ring", "pinky"]}
    all_landmark_dist = []
    total_frames = 0

    for session_name in tqdm(eval_sessions, desc=f"REACT ({'tracking' if model_config.predict_vel else 'regression'})"):
        hdf5_path = MINI_DATA_DIR / f"{session_name}.hdf5"

        ds = WindowedEmgDataset(
            hdf5_path=hdf5_path, window_length=WINDOW_LENGTH,
            stride=STRIDE, jitter=False, skip_ik_failures=True,
        )

        for i in range(len(ds)):
            sample = ds[i]
            emg = sample["emg"].unsqueeze(0).to(device)        # (1, 16, L)
            targets = sample["joint_angles"].unsqueeze(0).to(device)  # (1, 20, L)
            no_ik = sample["no_ik_failure"].unsqueeze(0).to(device)   # (1, L)

            # Initial position for tracking mode
            init_pos = None
            if model_config.provide_initial_pos:
                init_pos = targets[:, :, left_context].detach()

            with torch.no_grad():
                preds = model(
                    emg=emg,
                    calibration_features=cal_features,
                    num_calibration_samples=cal_k,
                    initial_pos=init_pos,
                )

            # Trim targets for encoder context
            start = left_context
            end = -right_context if right_context > 0 else None
            targets_trimmed = targets[..., start:end]
            mask_trimmed = no_ik[..., start:end]

            # Align prediction length to target
            target_len = targets_trimmed.shape[-1]
            preds_aligned = F.interpolate(
                preds, size=target_len, mode="linear", align_corners=False,
            ).float()
            targets_trimmed = targets_trimmed.float()

            n_valid = mask_trimmed.sum().item()
            if n_valid == 0:
                continue

            mae = compute_angle_mae(preds_aligned, targets_trimmed, mask_trimmed)
            finger_mae = compute_per_finger_mae(preds_aligned, targets_trimmed, mask_trimmed)
            landmark = compute_landmark_distance(preds_aligned, targets_trimmed, mask_trimmed)

            all_mae.append((mae, n_valid))
            for f, v in finger_mae.items():
                all_finger_mae[f].append((v, n_valid))
            all_landmark_dist.append((landmark, n_valid))
            total_frames += n_valid

    def weighted_avg(pairs):
        vals = [(v, n) for v, n in pairs if not np.isnan(v)]
        if not vals:
            return float("nan")
        total_n = sum(n for _, n in vals)
        return sum(v * n for v, n in vals) / total_n

    return {
        "angle_mae_deg": weighted_avg(all_mae),
        "per_finger_mae_deg": {f: weighted_avg(vs) for f, vs in all_finger_mae.items()},
        "fingertip_distance_mm": weighted_avg(all_landmark_dist),
        "total_valid_frames": total_frames,
        "num_eval_sessions": len(eval_sessions),
        "calibration_k": actual_k,
        "num_cal_sessions": len(cal_sessions),
    }


# ─── Results display ────────────────────────────────────────────────────────

def print_results_table(results: Dict[str, Dict[str, Any]]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 75)
    print("EVALUATION ON NEW DATA — User d387095792 (held-out, never in training)")
    print("=" * 75)

    # Header
    models = list(results.keys())
    header = f"{'Metric':<30}" + "".join(f"{m:>20}" for m in models)
    print(header)
    print("-" * len(header))

    # AngleMAE
    row = f"{'AngleMAE (deg)':<30}"
    for m in models:
        v = results[m].get("angle_mae_deg", float("nan"))
        row += f"{v:>20.3f}"
    print(row)

    # Fingertip distance
    row = f"{'Fingertip Dist (mm)':<30}"
    for m in models:
        v = results[m].get("fingertip_distance_mm", float("nan"))
        row += f"{v:>20.3f}"
    print(row)

    # Per-finger MAE
    print("-" * len(header))
    for finger in ["thumb", "index", "middle", "ring", "pinky"]:
        row = f"  {finger.capitalize():<28}"
        for m in models:
            v = results[m].get("per_finger_mae_deg", {}).get(finger, float("nan"))
            row += f"{v:>20.3f}"
        print(row)

    # Metadata
    print("-" * len(header))
    row = f"{'Valid frames':<30}"
    for m in models:
        v = results[m].get("total_valid_frames", 0)
        row += f"{v:>20,}"
    print(row)

    row = f"{'Eval sessions':<30}"
    for m in models:
        v = results[m].get("num_eval_sessions", 0)
        row += f"{v:>20}"
    print(row)
    print("=" * 75)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on new (held-out) data from emg2pose_dataset_mini"
    )
    parser.add_argument("--react-tracking", type=str, default=None,
                        help="Path to REACT tracking checkpoint (.pt)")
    parser.add_argument("--react-regression", type=str, default=None,
                        help="Path to REACT regression checkpoint (.pt)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only evaluate baselines (no REACT)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of calibration windows for REACT")
    parser.add_argument("--num-cal-recordings", type=int, default=5,
                        help="Number of recordings (left+right) to use for calibration")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUT_DIR / "new_data_eval_results.json"),
                        help="Path to save JSON results")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Get sessions and split
    all_sessions = get_mini_sessions()
    cal_sessions, eval_sessions = split_calibration_eval(
        all_sessions, args.num_cal_recordings
    )
    print(f"Mini dataset: {len(all_sessions)} sessions (user d387095792, held-out)")
    print(f"  Calibration: {len(cal_sessions)} sessions ({args.num_cal_recordings} recordings)")
    print(f"  Evaluation:  {len(eval_sessions)} sessions")
    print(f"  Calibration k={args.k}")

    results = {}

    # ── Baselines ──
    print("\n--- Baseline: tracking_vemg2pose ---")
    t0 = time.time()
    results["Baseline\n(Tracking)"] = evaluate_baseline(
        "tracking_vemg2pose.ckpt", eval_sessions,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\n--- Baseline: regression_vemg2pose ---")
    t0 = time.time()
    results["Baseline\n(Regression)"] = evaluate_baseline(
        "regression_vemg2pose.ckpt", eval_sessions,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── REACT models ──
    if not args.baseline_only:
        if args.react_tracking:
            print(f"\n--- REACT Tracking (k={args.k}) ---")
            t0 = time.time()
            results[f"REACT\n(Tracking, k={args.k})"] = evaluate_react(
                args.react_tracking, cal_sessions, eval_sessions,
                args.k, device,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

        if args.react_regression:
            print(f"\n--- REACT Regression (k={args.k}) ---")
            t0 = time.time()
            results[f"REACT\n(Regression, k={args.k})"] = evaluate_react(
                args.react_regression, cal_sessions, eval_sessions,
                args.k, device,
            )
            print(f"  Done in {time.time()-t0:.1f}s")

    # ── Print results ──
    print_results_table(results)

    # ── Save JSON ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Clean up keys for JSON (remove newlines)
    json_results = {k.replace("\n", " "): v for k, v in results.items()}
    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
