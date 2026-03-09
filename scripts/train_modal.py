#!/usr/bin/env python3
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Modal cloud training script for REACT-EMG on full emg2pose dataset.

Supports both regression and tracking modes, aligned with emg2pose configs.

Usage:
    # Default (regression mode)
    modal run scripts/train_modal.py
    
    # Tracking mode
    modal run scripts/train_modal.py --config tracking
    
    # With overrides
    modal run scripts/train_modal.py --config regression --epochs 100
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal

# =============================================================================
# Modal Configuration
# =============================================================================

# Remote paths (used in container)
REMOTE_EMG2POSE = "/root/emg2pose"
REMOTE_SRC = "/root/src"
REMOTE_CONFIGS = "/root/configs"

# Volume paths
VOLUME_MOUNT_PATH = "/persistent"
FULL_DATASET_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
CHECKPOINTS_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"

# Find local directories for upload (only runs locally, not in container)
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


# Only find local dirs when running locally (not in Modal container)
# Modal containers have MODAL_ENVIRONMENT set
if os.environ.get("MODAL_ENVIRONMENT") is None:
    LOCAL_EMG2POSE, LOCAL_SRC, LOCAL_CONFIGS = _find_local_dirs()
else:
    # In container - use remote paths (already mounted)
    LOCAL_EMG2POSE = Path(REMOTE_EMG2POSE)
    LOCAL_SRC = Path(REMOTE_SRC)
    LOCAL_CONFIGS = Path(REMOTE_CONFIGS)

# Modal app and volume
app = modal.App("react-emg-training")
dataset_volume = modal.Volume.from_name("emg2pose-full-dataset", create_if_missing=True)

# GPU image with all dependencies
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
        "matplotlib==3.9.0",
        "joblib==1.4.2",  # Required by emg2pose
    )
    .dockerfile_commands(["RUN chmod 1777 /dev/shm"])
    # Add local code and configs
    .add_local_dir(str(LOCAL_EMG2POSE), remote_path=REMOTE_EMG2POSE)
    .add_local_dir(str(LOCAL_SRC), remote_path=REMOTE_SRC)
    .add_local_dir(str(LOCAL_CONFIGS), remote_path=REMOTE_CONFIGS)
)


# =============================================================================
# Collate Function (defined at module level for multiprocessing pickle)
# =============================================================================

def collate_calibrated_batch(batch):
    """Collate function for PrebuiltCalibratedDataset.

    Must be at module level so DataLoader workers can pickle it.

    Each item carries max_k calibration windows with no padding.  A single
    random k is sampled here for the whole batch and the calibration tensor is
    sliced to shape (B, k, 16, L).  Every sample in the batch uses exactly k
    calibration windows — no wasted encoder compute on padding, no masking.
    """
    import numpy as np
    import torch
    
    emg = torch.stack([item["emg"] for item in batch])
    joint_angles = torch.stack([item["joint_angles"] for item in batch])
    no_ik_failure = torch.stack([item["no_ik_failure"] for item in batch])
    calibration_emg = torch.stack([item["calibration_emg"] for item in batch])  # (B, max_k, 16, L)

    # Sample k for the whole batch.  Support k=0 (zero-embedding path)
    # with 10% probability to ensure the zero_embedding gets trained.
    max_k = calibration_emg.shape[1]
    if max_k > 1 and np.random.random() < 0.1:
        k = 0
    elif max_k > 1:
        k = int(np.random.randint(1, max_k + 1))
    else:
        k = max_k
    calibration_emg = calibration_emg[:, :max(k, 1), :, :]  # keep at least 1 for shape

    return {
        "emg": emg,  # (B, 16, L)
        "joint_angles": joint_angles,  # (B, 20, L)
        "no_ik_failure": no_ik_failure,  # (B, L)
        "calibration_emg": calibration_emg,  # (B, k, 16, L)
        "calibration_k": torch.full((len(batch),), k, dtype=torch.long),  # (B,) all equal k
        "user_id": [item["user_id"] for item in batch],
        "session_name": [item["session_name"] for item in batch],
    }


# Keep old name as alias for backwards compatibility
collate_with_variable_calibration = collate_calibrated_batch


def pre_encode_calibration_pools(dataset, encoder, device, batch_size=128):
    """Pre-encode all calibration pool windows with the frozen encoder.

    Replaces raw EMG tensors (pool_size, 16, L) with encoded features
    (pool_size, C, L') in-place, eliminating redundant encoder passes
    during training.  The encoded features are kept on CPU and moved
    to GPU per-batch by the DataLoader.

    Also sets ``dataset._encoded_cal_shape = (C, L')`` so that the
    ``__getitem__`` fallback path creates correctly-shaped zero tensors.
    """
    import torch
    encoder.eval()
    total_windows = sum(p.shape[0] for p in dataset.calibration_pools.values())
    encoded_count = 0
    encoded_pools = {}  # build new dict to avoid mutating during iteration
    with torch.no_grad():
        for user_id, pool in dataset.calibration_pools.items():
            # pool: (pool_size, 16, L)
            encoded_chunks = []
            for i in range(0, pool.shape[0], batch_size):
                chunk = pool[i:i + batch_size].to(device)
                enc = encoder(chunk)  # (chunk_size, C, L')
                encoded_chunks.append(enc.cpu())
            encoded_pools[user_id] = torch.cat(encoded_chunks, dim=0)
            encoded_count += pool.shape[0]
    # Replace all pools atomically and store encoded shape for fallback
    dataset.calibration_pools = encoded_pools
    sample_pool = next(iter(encoded_pools.values()))
    dataset._encoded_cal_shape = tuple(sample_pool.shape[1:])  # (C, L')
    # Verify all pools have the same feature shape
    for uid, pool in encoded_pools.items():
        assert pool.shape[1:] == sample_pool.shape[1:], (
            f"Shape mismatch for user {uid}: {pool.shape} vs expected {sample_pool.shape}"
        )
    print(f"  Pre-encoded {encoded_count}/{total_windows} calibration windows "
          f"-> feature shape {dataset._encoded_cal_shape}")


# =============================================================================
# Training Function
# =============================================================================

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    timeout=60 * 60,  # 1 hour for validation
    cpu=10,
)
def precompute_dataset_cache(
    window_length: int = 11790,
    stride: int = 2000,
    num_workers: int = 10,
):
    """Pre-validate all sessions per split and save caches to persistent volume.
    
    Run this ONCE before training to avoid the ~5-min validation step:
        modal run scripts/train_modal.py::precompute_cache
    
    Generates separate caches for train (stride), val (stride*2), and
    test (stride*2).  Stored at /persistent/dataset_cache/ and reused
    automatically by train_react_emg on subsequent runs.
    """
    import pandas as pd
    import subprocess

    persistent_root = Path(VOLUME_MOUNT_PATH)
    dataset_dir = persistent_root / "emg2pose_data"
    metadata_file = dataset_dir / "metadata.csv"
    cache_dir = persistent_root / "dataset_cache"

    # Symlink for emg2pose
    home_dataset = Path("/root/emg2pose_data")
    if not home_dataset.exists():
        home_dataset.symlink_to(dataset_dir)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Dataset metadata not found at {metadata_file}")

    sys.path.insert(0, REMOTE_EMG2POSE)
    sys.path.insert(0, "/root")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", REMOTE_EMG2POSE], check=True)

    from src.utils.data import precompute_dataset_cache as _precompute

    metadata = pd.read_csv(metadata_file)
    print(f"Total files in metadata: {len(metadata)}")
    for split_name in ["train", "val", "test"]:
        split_df = metadata[metadata["split"] == split_name]
        print(f"  {split_name}: {len(split_df)} sessions, {split_df['user'].nunique()} unique users")

    _precompute(
        data_dir=home_dataset,
        metadata_df=metadata,
        cache_dir=cache_dir,
        window_length=window_length,
        stride=stride,
        num_workers=num_workers,
    )

    dataset_volume.commit()
    print(f"\nCache saved to {cache_dir} and committed to volume.")
    print("Subsequent training runs will skip validation automatically.")


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    timeout=300,
)
def download_run_outputs(output_dir: str) -> dict:
    """Read all output files from a training run directory on the volume.

    Returns a dict mapping filename -> bytes for every file in output_dir.
    Called from the local entrypoint after training completes.
    """
    import os

    dataset_volume.reload()
    results = {}
    if not os.path.isdir(output_dir):
        print(f"Warning: output dir {output_dir} not found on volume")
        return results
    for fname in os.listdir(output_dir):
        fpath = os.path.join(output_dir, fname)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                results[fname] = f.read()
    return results


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    gpu="A100-80GB", 
    timeout=60 * 60 * 12,  # 12 hours max
    cpu=10,
    memory=262144,  # 256GB RAM
)
def train_react_emg(
    config_file: str = "modal_regression.yaml",
    config_overrides: dict | None = None,
    download_if_missing: bool = True,
    resume_checkpoint: str | None = None,
):
    """Train REACT-EMG model on full dataset.
    
    Args:
        config_file: Config filename in configs/experiment/ (default: modal_regression.yaml).
                     Options: modal_regression.yaml, modal_tracking.yaml
        config_overrides: Dictionary of config values to override.
        download_if_missing: Download dataset if not present.
        resume_checkpoint: Path to a Phase 1 checkpoint on the Modal volume
            (e.g. /persistent/react_outputs/run_.../best_model.pt).
            If provided, loads trained FiLM/head weights and skips Phase 1,
            jumping directly to Phase 2 encoder unfreezing.
    """
    import json
    import time
    from datetime import datetime
    
    import yaml
    import torch
    import torch.nn as nn
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    import subprocess
    matplotlib.use('Agg')

    try:
        shm_info = subprocess.check_output(['df', '-h', '/dev/shm']).decode()
        print(f"--- Shared Memory Status ---\n{shm_info}---------------------------")
    except Exception as e:
        print(f"Could not check /dev/shm: {e}")
    
    # Load config
    config_path = Path(REMOTE_CONFIGS) / "experiment" / config_file
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Log mode
    mode = "REGRESSION" if "regression" in config_file else "TRACKING"
    print(f"Mode: {mode}")
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    # Extract config values (ensure proper types)
    num_epochs = int(cfg["training"]["num_epochs"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["optimizer"]["lr"])
    weight_decay = float(cfg["training"]["optimizer"]["weight_decay"])
    grad_clip = float(cfg["training"]["gradient_clip_norm"])
    num_workers = int(cfg["training"]["num_workers"])
    
    k_min = int(cfg["data"]["calibration"]["k_min"])
    k_max = int(cfg["data"]["calibration"]["k_max"])
    
    feature_dim = int(cfg["model"]["feature_dim"])
    user_embedding_dim = int(cfg["model"]["user_embedding_dim"])
    freeze_encoder = bool(cfg["model"]["freeze_encoder"])
    
    seed = int(cfg.get("seed", 42))
    commit_every = int(cfg["output"].get("commit_every", 10))
    
    # Phase 2: encoder unfreezing config
    num_epochs_enc_unfreeze = int(cfg["training"].get("num_epochs_enc_unfreeze", 0))
    enc_unfreeze_cfg = cfg["training"].get("encoder_unfreeze", {})
    enc_base_lr = float(enc_unfreeze_cfg.get("base_lr", 1e-4))
    enc_lr_decay = float(enc_unfreeze_cfg.get("lr_decay", 0.5))
    
    total_epochs = num_epochs + num_epochs_enc_unfreeze
    
    train_sessions_limit = cfg["data"].get("train_sessions_limit")
    val_sessions_limit = cfg["data"].get("val_sessions_limit")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup paths
    persistent_root = Path(VOLUME_MOUNT_PATH)
    # Volume contents appear directly at mount point (not under volume name)
    dataset_dir = persistent_root / "emg2pose_data"
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = persistent_root / "react_outputs" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for emg2pose expectations
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
    
    # Check if dataset already exists (skip download)
    metadata_file = dataset_dir / "metadata.csv"
    if download_if_missing and not metadata_file.exists():
        # Check if data needs downloading
        print(f"Looking for dataset at: {dataset_dir}")
        print(f"Metadata file exists: {metadata_file.exists()}")
        # List what's in the volume
        print(f"Contents of {persistent_root}:")
        if persistent_root.exists():
            for item in persistent_root.iterdir():
                print(f"  {item}")
        raise FileNotFoundError(
            f"Dataset not found at {dataset_dir}. "
            f"Please ensure the data is uploaded to the Modal volume."
        )
    
    if not metadata_file.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")
    
    # Download checkpoints if needed
    checkpoint_path = checkpoints_dir / "regression_vemg2pose.ckpt"
    if not checkpoint_path.exists():
        print("Downloading pretrained checkpoints...")
        archive = persistent_root / "emg2pose_model_checkpoints.tar.gz"
        if not archive.exists():
            subprocess.run(["curl", "-L", CHECKPOINTS_URL, "-o", str(archive)], check=True)
        subprocess.run(["tar", "-xvzf", str(archive), "-C", str(persistent_root)], check=True)
        dataset_volume.commit()
    
    ensure_symlink(dataset_dir, home_dataset)
    ensure_symlink(checkpoints_dir, home_checkpoints)
    
    # Setup Python path
    sys.path.insert(0, REMOTE_EMG2POSE)
    sys.path.insert(0, "/root")
    
    # Install emg2pose
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", REMOTE_EMG2POSE], check=True)
    
    # Setup logging
    log_file = output_dir / "training.log"
    
    def log(msg):
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp_str}] {msg}"
        print(full_msg)
        with open(log_file, "a") as f:
            f.write(full_msg + "\n")
    
    # Import after path setup
    from src.models.hybrid_model import (
        FiLMConditionedModel,
        FiLMConditionedModelConfig,
        load_pretrained_encoder,
    )
    from src.utils.data import create_lazy_datasets_from_metadata
    
    log("=" * 60)
    log("REACT-EMG Full Dataset Training (Modal)")
    log("=" * 60)
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("")
    log("Training parameters (from config):")
    log(f"  Phase 1 epochs (frozen encoder): {num_epochs}")
    log(f"  Phase 2 epochs (encoder unfreeze): {num_epochs_enc_unfreeze}")
    log(f"  Total epochs: {total_epochs}")
    log(f"  Batch size: {batch_size}")
    log(f"  Learning rate: {lr}")
    log(f"  Weight decay: {weight_decay}")
    if num_epochs_enc_unfreeze > 0:
        log(f"  Encoder unfreeze base LR: {enc_base_lr}")
        log(f"  Encoder layerwise LR decay: {enc_lr_decay}")
    log("")
    log("Calibration parameters:")
    log(f"  K min: {k_min}")
    log(f"  K max: {k_max}")
    log("")
    log(f"Output dir: {output_dir}")
    log("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dataset cache directory on the persistent volume
    cache_dir = persistent_root / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log(f"Dataset cache dir: {cache_dir}")
    
    # Load metadata and create splits using official emg2pose split column
    import pandas as pd
    metadata = pd.read_csv(metadata_file)
    
    log(f"Total files: {len(metadata)}")
    for split_name in ["train", "val", "test"]:
        split_df = metadata[metadata["split"] == split_name]
        log(f"  {split_name}: {len(split_df)} sessions, {split_df['user'].nunique()} unique users")
    
    # Create datasets from metadata with parallel validation
    log("Building datasets with parallel workers...")
    train_dataset, val_dataset, test_dataset = create_lazy_datasets_from_metadata(
        data_dir=home_dataset,
        metadata_df=metadata,
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"],
        calibration_k=k_max,
        min_calibration_k=k_min,
        num_workers=num_workers,  # Parallel workers for faster loading
        cache_dir=cache_dir,  # Use persistent cache
    )
    log("Datasets built successfully (all sessions pre-validated)")
    
    # collate_calibrated_batch is defined at module level for multiprocessing
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_calibrated_batch,
        prefetch_factor=2,
        persistent_workers=False,  # Allow crashed workers to be replaced
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_calibrated_batch,
        prefetch_factor=2,
        persistent_workers=False,
    )
    
    log(f"Train batches: {len(train_loader)}")
    log(f"Val batches: {len(val_loader)}")
    
    # Load pretrained encoder
    log("\nLoading pretrained encoder...")
    pretrained_encoder = load_pretrained_encoder(
        checkpoint_path=str(checkpoint_path),
        device=str(device),
    )
    log("Encoder loaded successfully.")
    
    # Create model
    log("\nCreating model...")
    model_config = FiLMConditionedModelConfig(
        feature_dim=feature_dim,
        user_embedding_dim=user_embedding_dim,
        freeze_encoder=freeze_encoder,
    )
    model = FiLMConditionedModel(model_config, pretrained_encoder=pretrained_encoder).to(device)
    
    # Load checkpoint weights if resuming (skip Phase 1)
    skip_phase1 = False
    if resume_checkpoint:
        import re
        log(f"\nLoading checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        state_dict = ckpt["model_state_dict"]
        # Strip _orig_mod. prefixes from torch.compile if present
        state_dict = {re.sub(r'_orig_mod\.', '', k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        log(f"  Loaded weights from epoch {ckpt.get('epoch', '?')}, "
            f"phase {ckpt.get('phase', '?')}, "
            f"val_loss {ckpt.get('val_loss', '?')}")
        if num_epochs_enc_unfreeze > 0:
            skip_phase1 = True
            log("  Skipping Phase 1 — jumping directly to Phase 2 encoder unfreezing")
        else:
            log("  WARNING: resume_checkpoint provided but num_epochs_enc_unfreeze=0; "
                "no Phase 2 to run. Phase 1 will proceed from loaded weights.")
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {num_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")
    
    # Save raw calibration pools before encoding (needed for Phase 2 re-encoding)
    import copy
    raw_train_cal_pools = {uid: pool.clone() for uid, pool in train_dataset.calibration_pools.items()}
    raw_val_cal_pools = {uid: pool.clone() for uid, pool in val_dataset.calibration_pools.items()}
    log(f"Saved raw calibration pools ({len(raw_train_cal_pools)} train, {len(raw_val_cal_pools)} val users)")
    
    # Pre-encode calibration pools (eliminates B*K redundant encoder passes per batch)
    log("\nPre-encoding calibration pools (one-time cost)...")
    pre_encode_calibration_pools(train_dataset, model.encoder, device, batch_size=128)
    pre_encode_calibration_pools(val_dataset, model.encoder, device, batch_size=128)
    log("Calibration pools pre-encoded. Training will use model.forward() with cached features.")
    
    # Compile trainable sub-modules with torch.compile for faster execution.
    # Skip if Phase 2 unfreezing is planned — compiled graphs may cache
    # no-grad assumptions from Phase 1 that break when the encoder is
    # unfrozen and gradients flow through it.
    if hasattr(torch, 'compile') and num_epochs_enc_unfreeze == 0:
        log("Compiling trainable model components with torch.compile...")
        try:
            model.user_encoder = torch.compile(model.user_encoder)
            model.film_layer = torch.compile(model.film_layer)
            model.prediction_head = torch.compile(model.prediction_head)
            log("torch.compile applied to user_encoder, film_layer, prediction_head")
        except Exception as e:
            log(f"torch.compile failed (non-fatal): {e}")
    elif num_epochs_enc_unfreeze > 0:
        log("Skipping torch.compile (Phase 2 encoder unfreezing is planned)")
    
    # Training setup (ensure proper types for optimizer params)
    betas = tuple(float(b) for b in cfg["training"]["optimizer"]["betas"])
    eps = float(cfg["training"]["optimizer"]["eps"])
    min_lr = float(cfg["training"]["scheduler"]["min_lr"])
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=min_lr,
    )
    criterion = nn.L1Loss()  # MAE loss (matches emg2pose baseline)
    
    # Get encoder context for proper target alignment
    left_context = getattr(model.encoder, 'left_context', 0)
    right_context = getattr(model.encoder, 'right_context', 0)
    log(f"Encoder context: left={left_context}, right={right_context}")
    
    # Mixed precision training (AMP) – ~2x speedup on A100
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    log(f"Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    phase_labels = []  # Track which phase each epoch belongs to
    
    start_time = time.time()
    
    if skip_phase1:
        log("\nPhase 1 skipped (loaded from checkpoint).")
        # Still set best_val_loss from checkpoint so Phase 2 has a baseline
        if resume_checkpoint:
            ckpt_val = ckpt.get("val_loss")
            if ckpt_val is not None:
                best_val_loss = float(ckpt_val)
                log(f"  Using checkpoint val_loss as baseline: {best_val_loss:.4f}")
    else:
        log("\nStarting Phase 1: Frozen Encoder Training...")
    
    for epoch in range(0 if skip_phase1 else num_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        epoch_losses = []
        data_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):
            data_time = time.perf_counter() - data_start
            step_start = time.perf_counter()
            emg = batch["emg"].to(device, non_blocking=True)
            targets = batch["joint_angles"].to(device, non_blocking=True)
            no_ik_failure = batch["no_ik_failure"].to(device, non_blocking=True)
            # calibration_emg is now PRE-ENCODED features (B, K, C, L')
            calibration_features = batch["calibration_emg"].to(device, non_blocking=True)
            calibration_k = batch["calibration_k"].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            try:
                fwd_start = time.perf_counter()
                with torch.cuda.amp.autocast(enabled=use_amp):
                    predictions = model(
                        emg=emg,
                        calibration_features=calibration_features,
                        num_calibration_samples=calibration_k,
                    )
                    
                    # Trim targets for encoder left/right context
                    start = left_context
                    end = -right_context if right_context > 0 else None
                    targets_trimmed = targets[..., start:end]
                    mask_trimmed = no_ik_failure[..., start:end]
                    
                    # Interpolate predictions to match target temporal resolution
                    target_len = targets_trimmed.shape[-1]
                    predictions = nn.functional.interpolate(
                        predictions, size=target_len, mode='linear',
                        align_corners=False,
                    )
                    
                    # Apply IK failure mask (only train on valid frames)
                    mask = mask_trimmed.unsqueeze(1).expand_as(predictions)
                    if mask.any():
                        loss = criterion(predictions[mask], targets_trimmed[mask])
                    else:
                        continue  # skip batch with no valid frames
                
                fwd_time = time.perf_counter() - fwd_start
                bwd_start = time.perf_counter()
                scaler.scale(loss).backward()
                
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                bwd_time = time.perf_counter() - bwd_start
                step_total_time = time.perf_counter() - step_start
                epoch_losses.append(loss.item())

                if batch_idx % 10 == 0:
                    log(
                        f"Epoch {epoch+1} [{batch_idx:4d}/{len(train_loader)}] | "
                        f"Loss: {loss.item():.4f} | "
                        f"Data: {data_time:.3f}s | Fwd: {fwd_time:.3f}s | Bwd: {bwd_time:.3f}s | "
                        f"Total: {step_total_time:.3f}s"
                    )
                data_start = time.perf_counter()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_epoch_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                emg = batch["emg"].to(device, non_blocking=True)
                targets = batch["joint_angles"].to(device, non_blocking=True)
                no_ik_failure = batch["no_ik_failure"].to(device, non_blocking=True)
                calibration_features = batch["calibration_emg"].to(device, non_blocking=True)
                calibration_k = batch["calibration_k"].to(device, non_blocking=True)
                
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        predictions = model(
                            emg=emg,
                            calibration_features=calibration_features,
                            num_calibration_samples=calibration_k,
                        )
                        
                        start = left_context
                        end = -right_context if right_context > 0 else None
                        targets_trimmed = targets[..., start:end]
                        mask_trimmed = no_ik_failure[..., start:end]
                        
                        target_len = targets_trimmed.shape[-1]
                        predictions = nn.functional.interpolate(
                            predictions, size=target_len, mode='linear',
                            align_corners=False,
                        )
                        
                        mask = mask_trimmed.unsqueeze(1).expand_as(predictions)
                        if mask.any():
                            loss = criterion(predictions[mask], targets_trimmed[mask])
                        else:
                            continue
                    val_epoch_losses.append(loss.item())
                except Exception:
                    continue
        
        avg_val_loss = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        phase_labels.append("phase1")
        
        log(
            f"Epoch {epoch + 1:3d}/{total_epochs} [Phase 1] | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            torch.save({
                "epoch": epoch,
                "phase": "phase1",
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
            }, output_dir / "best_model.pt")
            log(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
        
        # Commit volume periodically
        if (epoch + 1) % commit_every == 0:
            dataset_volume.commit()
    
    log(f"\nPhase 1 complete. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    # =========================================================================
    # Phase 2: Gradual Encoder Unfreezing with Layerwise LR Decay
    # =========================================================================
    if num_epochs_enc_unfreeze > 0:
        log("\n" + "=" * 60)
        log("Phase 2: Gradual Encoder Unfreezing")
        log("=" * 60)
        
        # Save phase 1 checkpoint as fallback
        torch.save({
            "epoch": num_epochs,
            "phase": "end_phase1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": best_val_loss,
            "config": cfg,
        }, output_dir / "phase1_final.pt")
        log("Saved phase 1 final checkpoint as fallback.")
        
        # Get encoder layer groups in output→input order for gradual unfreezing.
        # TdsNetwork.layers is nn.Sequential with:
        #   [0] Conv1dBlock (input)    [1] Conv1dBlock
        #   [2] TdsStage 0             [3] TdsStage 1 (output)
        encoder_children = list(model.encoder.layers.children())
        num_enc_layers = len(encoder_children)
        # Reverse: output→input order
        layers_output_to_input = list(reversed(encoder_children))
        
        log(f"Encoder has {num_enc_layers} layer groups.")
        log(f"Unfreezing schedule: output→input over {num_epochs_enc_unfreeze} epochs")
        for i, layer in enumerate(layers_output_to_input):
            layer_lr = enc_base_lr * (enc_lr_decay ** i)
            log(f"  Group {i} (encoder.layers[{num_enc_layers - 1 - i}]): "
                f"LR = {layer_lr:.2e}  "
                f"({type(layer).__name__})")
        
        # Allow gradients through encoder in forward pass
        model.config.freeze_encoder = False
        
        # Distribute layer groups across unfreeze epochs
        groups_per_epoch = max(1, num_enc_layers // num_epochs_enc_unfreeze)
        
        # Build Phase 2 optimizer ONCE to preserve Adam momentum/variance
        # across epochs.  Start with non-encoder params; encoder param
        # groups are added as layers are unfrozen.
        non_encoder_params = [
            p for name, p in model.named_parameters()
            if not name.startswith("encoder.") and p.requires_grad
        ]
        p2_param_groups = [{"params": non_encoder_params, "lr": lr}]
        
        # Pre-add all encoder layer groups (initially empty; filled as
        # layers are unfrozen).  This lets us use stable group indices.
        enc_group_start_idx = len(p2_param_groups)  # 1
        for group_idx, layer in enumerate(layers_output_to_input):
            layer_lr = enc_base_lr * (enc_lr_decay ** group_idx)
            p2_param_groups.append({
                "params": [],  # populated when unfrozen
                "lr": layer_lr,
            })
        
        p2_optimizer = torch.optim.AdamW(
            p2_param_groups,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )
        # Transfer Phase 1 optimizer state for non-encoder params
        # by initializing from the Phase 1 optimizer's state for
        # matching parameters.  optimizer.state is keyed by tensor
        # objects, so build an id→state lookup first.
        p1_state_by_id = {id(k): v for k, v in optimizer.state.items()}
        transferred = 0
        for p in non_encoder_params:
            if id(p) in p1_state_by_id:
                p2_optimizer.state[p] = p1_state_by_id[id(p)]
                transferred += 1
        log(f"Phase 2 optimizer created (transferred {transferred}/{len(non_encoder_params)} param states from Phase 1)")
        
        for unfreeze_epoch in range(num_epochs_enc_unfreeze):
            global_epoch = num_epochs + unfreeze_epoch
            epoch_start = time.time()
            
            # Determine which layer groups to unfreeze this epoch
            start_idx = unfreeze_epoch * groups_per_epoch
            # Last epoch unfreezes all remaining layers
            if unfreeze_epoch == num_epochs_enc_unfreeze - 1:
                end_idx = num_enc_layers
            else:
                end_idx = min(start_idx + groups_per_epoch, num_enc_layers)
            
            # Unfreeze the scheduled layers and add params to optimizer
            for idx in range(start_idx, end_idx):
                layer = layers_output_to_input[idx]
                for param in layer.parameters():
                    param.requires_grad = True
                # Add newly unfrozen params to the appropriate optimizer group
                opt_group_idx = enc_group_start_idx + idx
                p2_optimizer.param_groups[opt_group_idx]["params"] = [
                    p for p in layer.parameters()
                ]
                orig_idx = num_enc_layers - 1 - idx
                n_params = sum(p.numel() for p in layer.parameters())
                log(f"  Unfroze encoder.layers[{orig_idx}] "
                    f"({type(layer).__name__}, {n_params:,} params)")
            
            # Use the persistent Phase 2 optimizer (not rebuilt each epoch)
            optimizer = p2_optimizer
            
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            log(f"  Trainable parameters: {trainable_params:,}")
            
            # Restore raw calibration pools and re-encode with updated encoder
            log(f"  Re-encoding calibration pools with updated encoder...")
            train_dataset.calibration_pools = {uid: pool.clone() for uid, pool in raw_train_cal_pools.items()}
            val_dataset.calibration_pools = {uid: pool.clone() for uid, pool in raw_val_cal_pools.items()}
            pre_encode_calibration_pools(train_dataset, model.encoder, device, batch_size=128)
            pre_encode_calibration_pools(val_dataset, model.encoder, device, batch_size=128)
            
            # Train one epoch
            model.train()
            epoch_losses = []
            data_start = time.perf_counter()
            
            for batch_idx, batch in enumerate(train_loader):
                data_time = time.perf_counter() - data_start
                step_start = time.perf_counter()
                emg = batch["emg"].to(device, non_blocking=True)
                targets = batch["joint_angles"].to(device, non_blocking=True)
                no_ik_failure = batch["no_ik_failure"].to(device, non_blocking=True)
                calibration_features = batch["calibration_emg"].to(device, non_blocking=True)
                calibration_k = batch["calibration_k"].to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                try:
                    fwd_start = time.perf_counter()
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        predictions = model(
                            emg=emg,
                            calibration_features=calibration_features,
                            num_calibration_samples=calibration_k,
                        )
                        
                        start = left_context
                        end = -right_context if right_context > 0 else None
                        targets_trimmed = targets[..., start:end]
                        mask_trimmed = no_ik_failure[..., start:end]
                        
                        target_len = targets_trimmed.shape[-1]
                        predictions = nn.functional.interpolate(
                            predictions, size=target_len, mode='linear',
                            align_corners=False,
                        )
                        
                        mask = mask_trimmed.unsqueeze(1).expand_as(predictions)
                        if mask.any():
                            loss = criterion(predictions[mask], targets_trimmed[mask])
                        else:
                            continue
                    
                    fwd_time = time.perf_counter() - fwd_start
                    bwd_start = time.perf_counter()
                    scaler.scale(loss).backward()
                    
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    bwd_time = time.perf_counter() - bwd_start
                    step_total_time = time.perf_counter() - step_start
                    epoch_losses.append(loss.item())
                    
                    if batch_idx % 10 == 0:
                        # Show LRs for first encoder group and FiLM layers
                        enc_lrs = [g['lr'] for g in optimizer.param_groups[1:]]
                        enc_lr_str = "/".join(f"{r:.1e}" for r in enc_lrs[:3])
                        log(
                            f"Epoch {global_epoch+1} [{batch_idx:4d}/{len(train_loader)}] | "
                            f"Loss: {loss.item():.4f} | "
                            f"Data: {data_time:.3f}s | Fwd: {fwd_time:.3f}s | Bwd: {bwd_time:.3f}s | "
                            f"Total: {step_total_time:.3f}s | "
                            f"Enc LRs: {enc_lr_str}"
                        )
                    data_start = time.perf_counter()
                    
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    import traceback; traceback.print_exc()
                    continue
            
            avg_train_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
            train_losses.append(avg_train_loss)
            
            # Validate
            model.eval()
            val_epoch_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    emg = batch["emg"].to(device, non_blocking=True)
                    targets = batch["joint_angles"].to(device, non_blocking=True)
                    no_ik_failure = batch["no_ik_failure"].to(device, non_blocking=True)
                    calibration_features = batch["calibration_emg"].to(device, non_blocking=True)
                    calibration_k = batch["calibration_k"].to(device, non_blocking=True)
                    
                    try:
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            predictions = model(
                                emg=emg,
                                calibration_features=calibration_features,
                                num_calibration_samples=calibration_k,
                            )
                            
                            start = left_context
                            end = -right_context if right_context > 0 else None
                            targets_trimmed = targets[..., start:end]
                            mask_trimmed = no_ik_failure[..., start:end]
                            
                            target_len = targets_trimmed.shape[-1]
                            predictions = nn.functional.interpolate(
                                predictions, size=target_len, mode='linear',
                                align_corners=False,
                            )
                            
                            mask = mask_trimmed.unsqueeze(1).expand_as(predictions)
                            if mask.any():
                                loss = criterion(predictions[mask], targets_trimmed[mask])
                            else:
                                continue
                        val_epoch_losses.append(loss.item())
                    except Exception:
                        continue
            
            avg_val_loss = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
            val_losses.append(avg_val_loss)
            phase_labels.append("phase2")
            
            epoch_time = time.time() - epoch_start
            
            log(
                f"Epoch {global_epoch + 1:3d}/{total_epochs} [Phase 2] | "
                f"Train: {avg_train_loss:.4f} | "
                f"Val: {avg_val_loss:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = global_epoch + 1
                torch.save({
                    "epoch": global_epoch,
                    "phase": "phase2",
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "config": cfg,
                }, output_dir / "best_model.pt")
                log(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
            
            # Commit volume every epoch during phase 2
            dataset_volume.commit()
        
        log(f"\nPhase 2 complete. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    
    total_time = time.time() - start_time
    
    # Save final model
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }, output_dir / "final_model.pt")
    
    # Save training curves with phase boundary
    plt.figure(figsize=(12, 6))
    epochs_axis = list(range(1, len(train_losses) + 1))
    plt.plot(epochs_axis, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs_axis, val_losses, 'r-', label='Val Loss')
    if num_epochs_enc_unfreeze > 0 and num_epochs > 0:
        plt.axvline(x=num_epochs + 0.5, color='green', linestyle='--',
                    alpha=0.7, label='Phase 1→2 (unfreeze encoder)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('REACT-EMG Training (Full Dataset)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close()
    
    # Save history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_time_minutes": total_time / 60,
        "num_epochs_phase1": num_epochs,
        "num_epochs_phase2": num_epochs_enc_unfreeze,
        "phase_labels": phase_labels,
        "config": cfg,
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    dataset_volume.commit()
    
    log("\n" + "=" * 60)
    log("Training Complete!")
    log("=" * 60)
    log(f"Total time: {total_time / 60:.1f} minutes")
    log(f"Phase 1 epochs: {num_epochs} (frozen encoder)")
    log(f"Phase 2 epochs: {num_epochs_enc_unfreeze} (encoder unfreezing)")
    log(f"Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    log(f"Outputs saved to: {output_dir}")
    
    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "output_dir": str(output_dir),
    }


# =============================================================================
# Entry Point
# =============================================================================

@app.local_entrypoint()
def main(
    config: str = "regression",
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    k_max: int = None,
    resume_from: str = None,
):
    """Launch REACT-EMG training on Modal.
    
    Args:
        config: Config mode - 'regression' or 'tracking' (default: regression).
                Maps to configs/experiment/modal_{config}.yaml
        epochs: Override number of training epochs.
        batch_size: Override batch size.
        lr: Override learning rate.
        k_max: Override max calibration samples.
        resume_from: Path to a Phase 1 checkpoint on the Modal volume.
            Loads trained FiLM/head weights and skips Phase 1, jumping
            directly to Phase 2 encoder unfreezing.
    
    Examples:
        # Train with regression mode (default)
        modal run scripts/train_modal.py
        
        # Train with tracking mode
        modal run scripts/train_modal.py --config tracking
        
        # Skip Phase 1 and finetune encoder from a checkpoint
        modal run scripts/train_modal.py --config tracking \\
            --resume-from /persistent/react_outputs/run_.../best_model.pt
        
        # Override epochs
        modal run scripts/train_modal.py --config regression --epochs 100
    """
    # Resolve config file
    config_file = f"modal_{config}.yaml"
    print(f"Using config: configs/experiment/{config_file}")
    
    # Build config overrides from CLI args
    overrides = {}
    if epochs is not None:
        overrides["training.num_epochs"] = epochs
    if batch_size is not None:
        overrides["training.batch_size"] = batch_size
    if lr is not None:
        overrides["training.optimizer.lr"] = lr
    if k_max is not None:
        overrides["data.calibration.k_max"] = k_max
    
    result = train_react_emg.remote(
        config_file=config_file,
        config_overrides=overrides if overrides else None,
        resume_checkpoint=resume_from,
    )
    print(f"\nTraining completed!")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Output dir: {result['output_dir']}")

    # Download outputs from Modal volume to local filesystem
    remote_output_dir = result["output_dir"]
    run_name = Path(remote_output_dir).name  # e.g. run_20260308_162810
    local_output_dir = Path("outputs") / run_name
    local_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading logs to {local_output_dir}...")
    files = download_run_outputs.remote(remote_output_dir)
    for fname, content in files.items():
        local_path = local_output_dir / fname
        with open(local_path, "wb") as f:
            f.write(content)
        size_kb = len(content) / 1024
        print(f"  {fname} ({size_kb:.1f} KB)")
    print(f"All outputs saved to {local_output_dir}")


@app.local_entrypoint(name="precompute_cache")
def precompute_cache_entrypoint(
    window_length: int = 11790,
    stride: int = 2000,
):
    """Pre-validate all sessions and cache results on the Modal volume.
    
    Run once before training to eliminate the ~5-min validation step:
        modal run scripts/train_modal.py::precompute_cache
    
    Subsequent `modal run scripts/train_modal.py` calls will
    automatically load the cached validation manifest and skip
    the expensive per-file HDF5 checks.
    """
    precompute_dataset_cache.remote(
        window_length=window_length,
        stride=stride,
    )
    print("\nCache precomputed and committed to volume!")
    print("Training runs will now skip session validation.")
