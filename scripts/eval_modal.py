#!/usr/bin/env python3
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Modal evaluation script for REACT-EMG.

Evaluates a trained REACT-EMG model checkpoint on the full emg2pose test set,
computing the same metrics as emg2pose:
  - AngleMAE (angular mean absolute error)
  - AngularDerivatives (velocity, acceleration, jerk)
  - PerFingerAngleMAE (per-finger MAE)
  - PDAngleMAE (proximal/mid/distal MAE)
  - LandmarkDistances (fingertip + landmark Euclidean distance)

Results are returned to the local machine and saved under outputs/.

Usage:
    # With default k=15
    modal run scripts/eval_modal.py \\
        --checkpoint /persistent/react_outputs/run_20260309_042335/best_model.pt \\
        --k 15

    # Try a different k
    modal run scripts/eval_modal.py \\
        --checkpoint /persistent/react_outputs/run_20260309_042335/best_model.pt \\
        --k 5
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from functools import partial

import modal

# =============================================================================
# Modal Configuration
# =============================================================================

REMOTE_EMG2POSE = "/root/emg2pose"
REMOTE_SRC = "/root/src"
REMOTE_CONFIGS = "/root/configs"
VOLUME_MOUNT_PATH = "/persistent"
CHECKPOINTS_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"
)


def _find_local_dirs() -> tuple[Path, Path, Path]:
    """Find local emg2pose, src, and configs directories."""
    here = Path(__file__).resolve().parent.parent
    emg2pose_path = here / "emg2pose"
    src_path = here / "src"
    configs_path = here / "configs"

    if not (emg2pose_path / "setup.py").exists():
        raise FileNotFoundError(f"emg2pose not found at {emg2pose_path}")
    if not src_path.exists():
        raise FileNotFoundError(f"src not found at {src_path}")
    if not configs_path.exists():
        raise FileNotFoundError(f"configs not found at {configs_path}")

    return emg2pose_path, src_path, configs_path


if os.environ.get("MODAL_ENVIRONMENT") is None:
    LOCAL_EMG2POSE, LOCAL_SRC, LOCAL_CONFIGS = _find_local_dirs()
else:
    LOCAL_EMG2POSE = Path(REMOTE_EMG2POSE)
    LOCAL_SRC = Path(REMOTE_SRC)
    LOCAL_CONFIGS = Path(REMOTE_CONFIGS)

app = modal.App("react-emg-evaluation")
dataset_volume = modal.Volume.from_name(
    "emg2pose-full-dataset", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "tar")
    .pip_install(
        "torch==2.3.1",
        "pytorch-lightning==2.2.2",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "h5py==3.11.0",
        "pandas==2.2.2",
        "pyyaml==6.0.1",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "tqdm==4.66.4",
        "joblib==1.4.2",
    )
    .add_local_dir(str(LOCAL_EMG2POSE), remote_path=REMOTE_EMG2POSE)
    .add_local_dir(str(LOCAL_SRC), remote_path=REMOTE_SRC)
    .add_local_dir(str(LOCAL_CONFIGS), remote_path=REMOTE_CONFIGS)
)


# =============================================================================
# Collate (module-level for DataLoader pickling)
# =============================================================================

def _collate_fixed_k(batch, *, k: int):
    """Collate with a fixed k (used at module level for multiprocessing)."""
    import torch

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


def _collate_baseline(batch, *, user_id: str = "unknown"):
    """Collate for baseline vemg2pose (no calibration data)."""
    import torch

    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])

    return {
        "emg": emg,
        "joint_angles": joint_angles,
        "no_ik_failure": no_ik_failure,
        "user_id": [user_id] * len(batch),
    }


# =============================================================================
# Modal Evaluation Function
# =============================================================================

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    gpu="A10G",
    timeout=60 * 60 * 4,  # 4 hours
    cpu=10,
    memory=131072,  # 128 GB
)
def run_evaluation(
    checkpoint_path: str = "/persistent/react_outputs/run_20260309_042335/best_model.pt",
    k: int = 15,
    batch_size: int = 32,
    num_workers: int = 10,
    window_length: int = 11790,
    stride: int = 4000,
) -> dict:
    """Evaluate REACT-EMG model on the full test set with emg2pose metrics.

    Automatically detects tracking vs regression mode from the checkpoint
    config (predict_vel, provide_initial_pos, pretrained_checkpoint).

    Args:
        checkpoint_path: Path to REACT checkpoint on Modal volume.
        k: Number of calibration samples per user (fixed for all users).
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
        window_length: Windowing length (should match training; 11790 = 10000 + 1790).
        stride: Windowing stride (larger = faster but fewer windows).

    Returns:
        Dictionary with aggregate, per-user metrics, and metadata.
    """
    import subprocess
    import time

    import numpy as np
    import pandas as pd
    import torch
    from functools import partial

    # ── Setup paths & installs ───────────────────────────────────────
    persistent_root = Path(VOLUME_MOUNT_PATH)
    dataset_dir = persistent_root / "emg2pose_data"
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
    metadata_file = dataset_dir / "metadata.csv"

    home_dataset = Path("/root/emg2pose_data")
    home_checkpoints = Path("/root/emg2pose_model_checkpoints")

    def ensure_symlink(target: Path, link: Path):
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                import shutil
                shutil.rmtree(link)
        link.symlink_to(target)

    ensure_symlink(dataset_dir, home_dataset)
    ensure_symlink(checkpoints_dir, home_checkpoints)

    sys.path.insert(0, REMOTE_EMG2POSE)
    sys.path.insert(0, "/root")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", REMOTE_EMG2POSE],
        check=True,
    )

    # Validate inputs
    if not metadata_file.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_file}")
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Download pretrained encoder checkpoints if needed
    encoder_ckpt = checkpoints_dir / "regression_vemg2pose.ckpt"
    if not encoder_ckpt.exists():
        archive = persistent_root / "emg2pose_model_checkpoints.tar.gz"
        if not archive.exists():
            subprocess.run(
                ["curl", "-L", CHECKPOINTS_URL, "-o", str(archive)], check=True
            )
        subprocess.run(
            ["tar", "-xvzf", str(archive), "-C", str(persistent_root)], check=True
        )
        dataset_volume.commit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Imports (after path setup) ───────────────────────────────────
    from src.models.hybrid_model import (
        FiLMConditionedModel,
        FiLMConditionedModelConfig,
        load_pretrained_encoder,
        load_pretrained_decoder,
    )
    from src.utils.data import PrebuiltCalibratedDataset
    from src.evaluate.evaluate import (
        evaluate_react_emg,
        pre_encode_calibration_pools,
    )

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading REACT checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("config", {})

    feature_dim = cfg.get("model", {}).get("feature_dim", 64)
    user_embedding_dim = cfg.get("model", {}).get("user_embedding_dim", 128)

    # Resolve pretrained encoder path
    raw_encoder_path = cfg.get("model", {}).get(
        "pretrained_checkpoint",
        "emg2pose_model_checkpoints/regression_vemg2pose.ckpt",
    )
    if not Path(raw_encoder_path).is_absolute():
        encoder_path = str(home_checkpoints / Path(raw_encoder_path).name)
    else:
        encoder_path = raw_encoder_path

    print(f"Loading encoder architecture from: {encoder_path}")
    print("  (architecture only — all weights will be loaded from REACT checkpoint)")
    encoder = load_pretrained_encoder(encoder_path, device=str(device))

    print(f"Loading decoder architecture from: {encoder_path}")
    decoder = load_pretrained_decoder(encoder_path, device=str(device))

    model_cfg = cfg.get("model", {})
    is_tracking = model_cfg.get("predict_vel", False)
    print(f"Mode: {'tracking' if is_tracking else 'regression'}")
    model_config = FiLMConditionedModelConfig(
        feature_dim=feature_dim,
        user_embedding_dim=user_embedding_dim,
        freeze_encoder=False,   # irrelevant at eval time (torch.no_grad)
        freeze_decoder=False,
        predict_vel=model_cfg.get("predict_vel", False),
        provide_initial_pos=model_cfg.get("provide_initial_pos", False),
        state_condition=model_cfg.get("state_condition", True),
    )
    model = FiLMConditionedModel(
        model_config,
        pretrained_encoder=encoder,
        pretrained_decoder=decoder,
    )

    # Strip torch.compile _orig_mod. prefixes from state dict keys
    raw_sd = ckpt["model_state_dict"]
    import re
    cleaned_sd = {}
    for key, value in raw_sd.items():
        new_key = re.sub(r"\._orig_mod\.", ".", key)
        cleaned_sd[new_key] = value
    model.load_state_dict(cleaned_sd)

    model = model.to(device)
    model.eval()

    # Read context from the encoder directly (FiLMConditionedModel.__init__
    # doesn't copy it — only set_encoder() does)
    left_context = getattr(model.encoder, 'left_context', 0)
    right_context = getattr(model.encoder, 'right_context', 0)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {total_params:,} params")
    print(f"Encoder context: left={left_context}, right={right_context}")

    # ── Evaluate per generalization type (like emg2pose Table 5) ────
    RAD2DEG = 180.0 / np.pi
    metadata = pd.read_csv(metadata_file)
    test_df = metadata[metadata["split"] == "test"]
    print(f"\nFull test set: {len(test_df)} sessions, {test_df['user'].nunique()} users")

    cache_dir = persistent_root / "dataset_cache"
    generalization_types = ["user", "stage", "user_stage"]
    gen_labels = {"user": "User", "stage": "Stage", "user_stage": "User, Stage"}
    all_results = {}
    total_start = time.time()

    for gen_type in generalization_types:
        gen_df = test_df[test_df["generalization"] == gen_type]
        if gen_df.empty:
            print(f"\n  Skipping {gen_type}: no sessions")
            continue

        session_infos = [(row["filename"], row["user"]) for _, row in gen_df.iterrows()]
        n_users = gen_df["user"].nunique()
        print(f"\n{'='*60}")
        print(f"Generalization: {gen_labels[gen_type]} ({gen_type})")
        print(f"  {len(session_infos)} sessions, {n_users} users")
        print(f"{'='*60}")

        # Build dataset for this generalization split
        print(f"Building {gen_type} dataset...")
        gen_dataset = PrebuiltCalibratedDataset(
            data_dir=home_dataset,
            session_infos=session_infos,
            window_length=window_length,
            stride=stride,
            jitter=False,
            calibration_k=max(k, 1),
            min_calibration_k=max(k, 1),
            show_progress=True,
            num_workers=num_workers,
            cache_dir=cache_dir,
            split_name=f"test_{gen_type}",
        )

        # Pre-encode calibration pools
        print(f"Pre-encoding calibration pools for {gen_type}...")
        pre_encode_calibration_pools(gen_dataset, model.encoder, device)

        # Create DataLoader
        collate_fn = partial(_collate_fixed_k, k=k)
        gen_loader = torch.utils.data.DataLoader(
            gen_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=2,
            persistent_workers=False,
        )
        print(f"  Batches: {len(gen_loader)} (k={k})")

        # Run evaluation
        t0 = time.time()
        gen_results = evaluate_react_emg(
            model=model,
            test_loader=gen_loader,
            device=device,
            left_context=left_context,
            right_context=right_context,
            use_amp=True,
            provide_initial_pos=model_config.provide_initial_pos,
        )
        dt = time.time() - t0
        gen_results["eval_time_seconds"] = dt
        all_results[gen_type] = gen_results

        # Print per-generalization summary
        agg = gen_results.get("aggregate", {})
        print(f"\n  {gen_labels[gen_type]} results ({dt:.1f}s):")
        print(f"    Batches: {gen_results.get('num_batches', 0)}, "
              f"Valid timesteps: {gen_results.get('total_valid_timesteps', 0)}, "
              f"Users: {gen_results.get('num_users', 0)}")

        # Detailed metrics (in degrees for angles)
        metric_groups = {
            "Angle MAE": ["test_mae"],
            "Per-Finger MAE": [f"test_mae_{f}" for f in ["thumb", "index", "middle", "ring", "pinky"]],
            "Proximal-Distal MAE": [f"test_mae_{g}" for g in ["proximal", "mid", "distal"]],
            "Angular Derivatives": ["test_vel", "test_acc", "test_jerk"],
            "Landmark Distances": ["test_fingertip_distance", "test_landmark_distance"],
        }
        ANGLE_KEYS = {
            "test_mae", "test_vel", "test_acc", "test_jerk",
            "test_mae_thumb", "test_mae_index", "test_mae_middle",
            "test_mae_ring", "test_mae_pinky",
            "test_mae_proximal", "test_mae_mid", "test_mae_distal",
        }
        for group_name, keys in metric_groups.items():
            print(f"    {group_name}:")
            for mkey in keys:
                if mkey in agg:
                    label = mkey.replace("test_", "").replace("_", " ")
                    val = agg[mkey]
                    if mkey in ANGLE_KEYS:
                        unit = "°" if "mae" in mkey else "°/s^n"
                        print(f"      {label:28s} {val * RAD2DEG:10.4f} {unit}")
                    else:
                        print(f"      {label:28s} {val:10.4f} mm")
            print()

        # Free memory
        del gen_dataset, gen_loader
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Print Table 5 format summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TABLE 5 FORMAT: Test set results (k={k})")
    print(f"Mean and standard deviation reported across users")
    print(f"{'='*60}")
    header = f"{'Generalization':<16s} {'Angular Error (°)':>20s} {'Landmark Dist (mm)':>20s}"
    print(header)
    print("-" * len(header))

    for gen_type in generalization_types:
        if gen_type not in all_results:
            continue
        per_user = all_results[gen_type].get("per_user", {})
        if not per_user:
            continue
        mae_vals = [v.get("test_mae", float("nan")) * RAD2DEG for v in per_user.values()]
        dist_vals = [v.get("test_landmark_distance", float("nan")) for v in per_user.values()]

        mae_mean, mae_std = np.nanmean(mae_vals), np.nanstd(mae_vals)
        dist_mean, dist_std = np.nanmean(dist_vals), np.nanstd(dist_vals)

        label = gen_labels[gen_type]
        print(f"{label:<16s} {mae_mean:7.1f} ± {mae_std:<7.1f}   {dist_mean:7.1f} ± {dist_std:<7.1f}")

    print(f"\nTotal evaluation time: {total_time:.1f}s")

    # Add metadata
    all_results["metadata"] = {
        "checkpoint_path": checkpoint_path,
        "k": k,
        "batch_size": batch_size,
        "window_length": window_length,
        "stride": stride,
        "total_eval_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
        "epoch": ckpt.get("epoch", "unknown"),
        "train_config": cfg,
    }

    return all_results


# =============================================================================
# Baseline vemg2pose Evaluation (original pretrained weights, no REACT)
# =============================================================================

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    gpu="A10G",
    timeout=60 * 60 * 4,  # 4 hours
    cpu=10,
    memory=131072,  # 128 GB
)
def run_baseline_evaluation(
    baseline_checkpoint: str = "tracking_vemg2pose",
    batch_size: int = 32,
    num_workers: int = 10,
    window_length: int = 10_000,
    stride: int = 4000,
) -> dict:
    """Evaluate the original vemg2pose model (no REACT) as a baseline.

    Uses emg2pose's StatePoseModule with pretrained encoder + decoder
    weights directly from the emg2pose checkpoint.

    Args:
        baseline_checkpoint: Name of the emg2pose checkpoint to use.
            One of: tracking_vemg2pose, regression_vemg2pose,
            tracking_emg2pose, regression_emg2pose, regression_neuropose.
        batch_size: Evaluation batch size.
        num_workers: DataLoader workers.
        window_length: Content window length (10000 = 5s at 2kHz).
        stride: Windowing stride.

    Returns:
        Dictionary with per-generalization aggregate and per-user metrics.
    """
    import logging
    import subprocess
    import time

    import numpy as np
    import pandas as pd
    import torch
    from torch.utils.data import ConcatDataset, DataLoader

    # ── Setup paths & installs ───────────────────────────────────────
    persistent_root = Path(VOLUME_MOUNT_PATH)
    dataset_dir = persistent_root / "emg2pose_data"
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
    metadata_file = dataset_dir / "metadata.csv"

    home_dataset = Path("/root/emg2pose_data")
    home_checkpoints = Path("/root/emg2pose_model_checkpoints")

    def ensure_symlink(target: Path, link: Path):
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                import shutil
                shutil.rmtree(link)
        link.symlink_to(target)

    ensure_symlink(dataset_dir, home_dataset)
    ensure_symlink(checkpoints_dir, home_checkpoints)

    sys.path.insert(0, REMOTE_EMG2POSE)
    sys.path.insert(0, "/root")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", REMOTE_EMG2POSE],
        check=True,
    )

    # Validate inputs
    if not metadata_file.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_file}")

    # Download pretrained checkpoints if needed
    ckpt_file = checkpoints_dir / f"{baseline_checkpoint}.ckpt"
    if not ckpt_file.exists():
        archive = persistent_root / "emg2pose_model_checkpoints.tar.gz"
        if not archive.exists():
            subprocess.run(
                ["curl", "-L", CHECKPOINTS_URL, "-o", str(archive)], check=True
            )
        subprocess.run(
            ["tar", "-xvzf", str(archive), "-C", str(persistent_root)], check=True
        )
        dataset_volume.commit()
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Imports ──────────────────────────────────────────────────────
    from src.models.hybrid_model import load_pretrained_encoder, load_pretrained_decoder
    from src.evaluate.evaluate import evaluate_react_emg
    from emg2pose.data import WindowedEmgDataset
    from emg2pose.pose_modules import StatePoseModule, VEMG2PoseWithInitialState
    from emg2pose.constants import EMG_SAMPLE_RATE

    # ── Build baseline model with pretrained weights ─────────────────
    print(f"\nLoading baseline vemg2pose model: {baseline_checkpoint}")
    encoder_path = str(home_checkpoints / f"{baseline_checkpoint}.ckpt")

    encoder = load_pretrained_encoder(encoder_path, device=str(device))
    decoder = load_pretrained_decoder(encoder_path, device=str(device))

    left_context = getattr(encoder, 'left_context', 0)
    right_context = getattr(encoder, 'right_context', 0)

    # Determine mode from checkpoint name
    is_tracking = "tracking" in baseline_checkpoint
    rollout_freq = 50

    if is_tracking:
        # Tracking: StatePoseModule (predict_vel=True, state_condition=True)
        model = StatePoseModule(
            network=encoder,
            decoder=decoder,
            state_condition=True,
            predict_vel=True,
            rollout_freq=rollout_freq,
        )
        provide_initial_pos = True
        mode_label = "tracking"
    else:
        # Regression: VEMG2PoseWithInitialState (decoder out=40, pos+vel)
        model = VEMG2PoseWithInitialState(
            network=encoder,
            decoder=decoder,
            num_position_steps=500,  # 250ms at 2kHz (emg2pose default)
            state_condition=True,
            rollout_freq=rollout_freq,
        )
        provide_initial_pos = False
        mode_label = "regression"

    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Baseline model: {total_params:,} params")
    print(f"  mode={mode_label}, provide_initial_pos={provide_initial_pos}")
    print(f"  rollout_freq={rollout_freq}")
    print(f"  Encoder context: left={left_context}, right={right_context}")

    # ── Effective window length (content + context) ──────────────────
    context_length = left_context + right_context
    effective_window_length = window_length + context_length

    # ── Evaluate per generalization type ─────────────────────────────
    RAD2DEG = 180.0 / np.pi
    metadata = pd.read_csv(metadata_file)
    test_df = metadata[metadata["split"] == "test"]
    print(f"\nFull test set: {len(test_df)} sessions, {test_df['user'].nunique()} users")

    generalization_types = ["user", "stage", "user_stage"]
    gen_labels = {"user": "User", "stage": "Stage", "user_stage": "User, Stage"}
    all_results = {}
    total_start = time.time()

    for gen_type in generalization_types:
        gen_df = test_df[test_df["generalization"] == gen_type]
        if gen_df.empty:
            print(f"\n  Skipping {gen_type}: no sessions")
            continue

        n_users = gen_df["user"].nunique()
        print(f"\n{'='*60}")
        print(f"Generalization: {gen_labels[gen_type]} ({gen_type})")
        print(f"  {len(gen_df)} sessions, {n_users} users")
        print(f"{'='*60}")

        # Build one dataloader per user (for per-user metrics)
        # Matches emg2pose's test_analysis: groupby user, each gets
        # its own ConcatDataset, but we combine into one DataLoader
        # with user_id tracking via collate.
        print(f"Building {gen_type} dataloaders...")
        all_datasets = []
        user_for_sample = []  # track which user each sample belongs to

        for user_id, user_df in gen_df.groupby("user"):
            sessions = []
            for fn in user_df["filename"] + ".hdf5":
                full_path = str(home_dataset / fn)
                try:
                    ds = WindowedEmgDataset(
                        full_path,
                        window_length=effective_window_length,
                        stride=stride,
                        jitter=False,
                        skip_ik_failures=True,
                    )
                    sessions.append(ds)
                except OSError as e:
                    logging.warning(f"Skipping {full_path}: {e}")

            if sessions:
                user_ds = ConcatDataset(sessions)
                # Record user_id for each sample in this user's dataset
                user_for_sample.extend([user_id] * len(user_ds))
                all_datasets.append(user_ds)

        if not all_datasets:
            print(f"  No valid sessions for {gen_type}")
            continue

        combined = ConcatDataset(all_datasets)
        print(f"  {len(combined)} samples across {n_users} users")

        # Build a mapping from sample index to user_id
        _user_map = user_for_sample  # list aligned with combined dataset indices

        # Wrap in a dataset that injects user_id per sample
        class _UserTaggedDataset(torch.utils.data.Dataset):
            """Wraps a ConcatDataset and adds user_id to each sample."""
            def __init__(self, dataset, user_map):
                self._ds = dataset
                self._umap = user_map
            def __len__(self):
                return len(self._ds)
            def __getitem__(self, idx):
                sample = self._ds[idx]
                sample["user_id"] = self._umap[idx]
                return sample

        tagged = _UserTaggedDataset(combined, _user_map)

        def _baseline_collate(batch_items):
            """Collate that stacks tensors and gathers user_ids."""
            import torch as _torch
            emg = _torch.stack([item["emg"] for item in batch_items])
            joint_angles = _torch.stack([item["joint_angles"] for item in batch_items])
            no_ik_failure = _torch.stack([item["no_ik_failure"] for item in batch_items])
            uids = [item["user_id"] for item in batch_items]
            return {
                "emg": emg,
                "joint_angles": joint_angles,
                "no_ik_failure": no_ik_failure,
                "user_id": uids,
            }

        gen_loader = DataLoader(
            tagged,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_baseline_collate,
            pin_memory=True,
            persistent_workers=False,
        )
        print(f"  Batches: {len(gen_loader)}")

        # Run evaluation
        t0 = time.time()
        gen_results = evaluate_react_emg(
            model=model,
            test_loader=gen_loader,
            device=device,
            left_context=left_context,
            right_context=right_context,
            use_amp=False,  # baseline model is float32
            baseline_mode=True,
            provide_initial_pos=provide_initial_pos,
        )
        dt = time.time() - t0
        gen_results["eval_time_seconds"] = dt
        all_results[gen_type] = gen_results

        # Print per-generalization summary
        agg = gen_results.get("aggregate", {})
        print(f"\n  {gen_labels[gen_type]} results ({dt:.1f}s):")
        print(f"    Batches: {gen_results.get('num_batches', 0)}, "
              f"Valid timesteps: {gen_results.get('total_valid_timesteps', 0)}, "
              f"Users: {gen_results.get('num_users', 0)}")

        metric_groups = {
            "Angle MAE": ["test_mae"],
            "Per-Finger MAE": [f"test_mae_{f}" for f in ["thumb", "index", "middle", "ring", "pinky"]],
            "Proximal-Distal MAE": [f"test_mae_{g}" for g in ["proximal", "mid", "distal"]],
            "Angular Derivatives": ["test_vel", "test_acc", "test_jerk"],
            "Landmark Distances": ["test_fingertip_distance", "test_landmark_distance"],
        }
        ANGLE_KEYS = {
            "test_mae", "test_vel", "test_acc", "test_jerk",
            "test_mae_thumb", "test_mae_index", "test_mae_middle",
            "test_mae_ring", "test_mae_pinky",
            "test_mae_proximal", "test_mae_mid", "test_mae_distal",
        }
        for group_name, keys in metric_groups.items():
            print(f"    {group_name}:")
            for mkey in keys:
                if mkey in agg:
                    label = mkey.replace("test_", "").replace("_", " ")
                    val = agg[mkey]
                    if mkey in ANGLE_KEYS:
                        unit = "°" if "mae" in mkey else "°/s^n"
                        print(f"      {label:28s} {val * RAD2DEG:10.4f} {unit}")
                    else:
                        print(f"      {label:28s} {val:10.4f} mm")
            print()

        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # ── Print Table 5 format summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print(f"BASELINE TABLE 5: {baseline_checkpoint}")
    print(f"Mean and standard deviation reported across users")
    print(f"{'='*60}")
    header = f"{'Generalization':<16s} {'Angular Error (°)':>20s} {'Landmark Dist (mm)':>20s}"
    print(header)
    print("-" * len(header))

    for gen_type in generalization_types:
        if gen_type not in all_results:
            continue
        per_user = all_results[gen_type].get("per_user", {})
        if not per_user:
            continue
        mae_vals = [v.get("test_mae", float("nan")) * RAD2DEG for v in per_user.values()]
        dist_vals = [v.get("test_landmark_distance", float("nan")) for v in per_user.values()]

        mae_mean, mae_std = np.nanmean(mae_vals), np.nanstd(mae_vals)
        dist_mean, dist_std = np.nanmean(dist_vals), np.nanstd(dist_vals)

        label = gen_labels[gen_type]
        print(f"{label:<16s} {mae_mean:7.1f} ± {mae_std:<7.1f}   {dist_mean:7.1f} ± {dist_std:<7.1f}")

    print(f"\nTotal evaluation time: {total_time:.1f}s")

    all_results["metadata"] = {
        "baseline_checkpoint": baseline_checkpoint,
        "batch_size": batch_size,
        "window_length": window_length,
        "stride": stride,
        "total_eval_time_seconds": total_time,
        "timestamp": datetime.now().isoformat(),
    }

    return all_results


# =============================================================================
# Local Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    checkpoint: str = "/persistent/react_outputs/run_20260309_042335/best_model.pt",
    k: int = 15,
    batch_size: int = 32,
    stride: int = 4000,
    baseline: bool = False,
    baseline_checkpoint: str = "tracking_vemg2pose",
):
    """Evaluate REACT-EMG or baseline model on all 3 generalization splits.

    Matches the emg2pose paper Table 5 format:
      - User:       held-out users, seen stages
      - Stage:      seen users, held-out stages
      - User,Stage: held-out users AND held-out stages

    REACT mode auto-detects tracking vs regression from the checkpoint config.

    Results are saved locally under outputs/ with per-generalization CSVs.

    Args:
        checkpoint: Path to REACT model checkpoint on the Modal volume.
        k: Number of calibration samples per user.
        batch_size: Evaluation batch size.
        stride: Windowing stride (larger = faster, fewer windows).
        baseline: If True, evaluate the original vemg2pose model
            (pretrained weights, no REACT fine-tuning) as a baseline.
        baseline_checkpoint: Name of the emg2pose checkpoint for baseline
            evaluation (e.g., tracking_vemg2pose, regression_vemg2pose).

    Examples:
        # REACT tracking eval
        modal run scripts/eval_modal.py \\
            --checkpoint /persistent/react_outputs/run_.../best_model.pt --k 15

        # REACT regression eval
        modal run scripts/eval_modal.py \\
            --checkpoint /persistent/react_outputs/run_.../best_model.pt --k 15

        # Baseline tracking eval
        modal run scripts/eval_modal.py --baseline

        # Baseline regression eval
        modal run scripts/eval_modal.py --baseline --baseline-checkpoint regression_vemg2pose
    """
    import csv
    import math

    RAD2DEG = 180.0 / math.pi

    if baseline:
        print(f"Launching BASELINE evaluation on Modal...")
        print(f"  Baseline checkpoint: {baseline_checkpoint}")
        print(f"  Batch size: {batch_size}")
        print(f"  Stride: {stride}")

        results = run_baseline_evaluation.remote(
            baseline_checkpoint=baseline_checkpoint,
            batch_size=batch_size,
            stride=stride,
        )

        # Save results locally
        output_dir = Path("outputs") / f"eval_baseline_{baseline_checkpoint}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        table_title = f"BASELINE TABLE 5: {baseline_checkpoint}"

    else:
        print(f"Launching REACT evaluation on Modal...")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  k: {k}")
        print(f"  Batch size: {batch_size}")
        print(f"  Stride: {stride}")

        results = run_evaluation.remote(
            checkpoint_path=checkpoint,
            k=k,
            batch_size=batch_size,
            stride=stride,
        )

        # Save results locally
        output_dir = Path("outputs") / f"eval_k{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        table_title = f"TABLE 5 FORMAT: Test set results (k={k})"

    # Save full results as JSON
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {results_file}")

    # Metrics that should be converted from radians to degrees
    ANGLE_METRICS = {
        "test_mae", "test_vel", "test_acc", "test_jerk",
        "test_mae_thumb", "test_mae_index", "test_mae_middle",
        "test_mae_ring", "test_mae_pinky",
        "test_mae_proximal", "test_mae_mid", "test_mae_distal",
    }

    gen_labels = {"user": "User", "stage": "Stage", "user_stage": "User, Stage"}
    generalization_types = ["user", "stage", "user_stage"]

    for gen_type in generalization_types:
        if gen_type not in results:
            continue
        gen_results = results[gen_type]
        gen_dir = output_dir / gen_type
        gen_dir.mkdir(parents=True, exist_ok=True)

        # Save aggregate metrics as CSV (with degree conversion)
        agg = gen_results.get("aggregate", {})
        if agg:
            csv_file = gen_dir / "aggregate_metrics.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value", "unit"])
                for mkey in sorted(agg.keys()):
                    val = agg[mkey]
                    if mkey in ANGLE_METRICS:
                        if "mae" in mkey:
                            writer.writerow([mkey, val * RAD2DEG, "degrees"])
                        else:
                            writer.writerow([mkey, val * RAD2DEG, "deg/s^n"])
                    elif "distance" in mkey:
                        writer.writerow([mkey, val, "mm"])
                    else:
                        writer.writerow([mkey, val, ""])
            print(f"  {gen_labels[gen_type]} aggregate: {csv_file}")

        # Save per-user metrics as CSV (with degree conversion)
        per_user = gen_results.get("per_user", {})
        if per_user:
            csv_file = gen_dir / "per_user_metrics.csv"
            all_keys = set()
            for u_metrics in per_user.values():
                all_keys.update(u_metrics.keys())
            all_keys = sorted(all_keys)

            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                headers = ["user_id"]
                for col in all_keys:
                    if col in ANGLE_METRICS:
                        headers.append(f"{col} (deg)")
                    elif "distance" in col:
                        headers.append(f"{col} (mm)")
                    else:
                        headers.append(col)
                writer.writerow(headers)

                for uid in sorted(per_user.keys()):
                    row = [uid]
                    for col in all_keys:
                        val = per_user[uid].get(col, "")
                        if val != "" and col in ANGLE_METRICS:
                            val = val * RAD2DEG
                        row.append(val)
                    writer.writerow(row)
            print(f"  {gen_labels[gen_type]} per-user:  {csv_file}")

    # ── Print Table 5 format summary ─────────────────────────────────
    print(f"\n{'='*60}")
    print(table_title)
    print(f"Mean and standard deviation reported across users")
    print(f"{'='*60}")
    header = f"{'Generalization':<16s} {'Angular Error (°)':>20s} {'Landmark Dist (mm)':>20s}"
    print(header)
    print("-" * len(header))

    for gen_type in generalization_types:
        if gen_type not in results:
            continue
        per_user = results[gen_type].get("per_user", {})
        if not per_user:
            continue

        mae_vals = [v.get("test_mae", float("nan")) * RAD2DEG for v in per_user.values()]
        dist_vals = [v.get("test_landmark_distance", float("nan")) for v in per_user.values()]

        # Population mean and std (matching paper: mean ± std across users)
        n = len(mae_vals)
        mae_mean = sum(mae_vals) / n
        mae_std = (sum((x - mae_mean) ** 2 for x in mae_vals) / n) ** 0.5
        dist_mean = sum(dist_vals) / n
        dist_std = (sum((x - dist_mean) ** 2 for x in dist_vals) / n) ** 0.5

        label = gen_labels[gen_type]
        print(f"{label:<16s} {mae_mean:7.1f} ± {mae_std:<7.1f}   {dist_mean:7.1f} ± {dist_std:<7.1f}")

    print(f"\nResults directory: {output_dir}")
