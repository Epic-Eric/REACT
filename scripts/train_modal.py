#!/usr/bin/env python3
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Modal cloud training script for REACT-EMG on full emg2pose dataset.

Loads configuration from configs/experiment/modal.yaml.
Trains FiLM-conditioned model with pretrained vemg2pose encoder on Modal cloud.

Usage:
    modal run scripts/train_modal.py
    modal run scripts/train_modal.py --epochs 50 --batch-size 16
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

# Find local directories for upload
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


LOCAL_EMG2POSE, LOCAL_SRC, LOCAL_CONFIGS = _find_local_dirs()
REMOTE_EMG2POSE = "/root/emg2pose"
REMOTE_SRC = "/root/src"
REMOTE_CONFIGS = "/root/configs"

# Volume paths
VOLUME_MOUNT_PATH = "/persistent"
FULL_DATASET_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset.tar"
CHECKPOINTS_URL = "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"

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
    )
    # Add local code and configs
    .add_local_dir(str(LOCAL_EMG2POSE), remote_path=REMOTE_EMG2POSE)
    .add_local_dir(str(LOCAL_SRC), remote_path=REMOTE_SRC)
    .add_local_dir(str(LOCAL_CONFIGS), remote_path=REMOTE_CONFIGS)
)


# =============================================================================
# Training Function
# =============================================================================

@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: dataset_volume},
    gpu="A10G",  # Can be overridden by config
    timeout=60 * 60 * 12,  # 12 hours max
    memory=32768,  # 32GB RAM
)
def train_react_emg(
    config_overrides: dict | None = None,
    download_if_missing: bool = True,
):
    """Train REACT-EMG model on full dataset.
    
    Args:
        config_overrides: Dictionary of config values to override.
        download_if_missing: Download dataset if not present.
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
    matplotlib.use('Agg')
    
    # Load config
    config_path = Path(REMOTE_CONFIGS) / "experiment" / "modal.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Apply overrides
    if config_overrides:
        for key, value in config_overrides.items():
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    
    # Extract config values
    num_epochs = cfg["training"]["num_epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["optimizer"]["lr"]
    weight_decay = cfg["training"]["optimizer"]["weight_decay"]
    grad_clip = cfg["training"]["gradient_clip_norm"]
    num_workers = cfg["training"]["num_workers"]
    
    k_min = cfg["data"]["calibration"]["k_min"]
    k_max = cfg["data"]["calibration"]["k_max"]
    
    feature_dim = cfg["model"]["feature_dim"]
    user_embedding_dim = cfg["model"]["user_embedding_dim"]
    freeze_encoder = cfg["model"]["freeze_encoder"]
    
    seed = cfg.get("seed", 42)
    commit_every = cfg["output"].get("commit_every", 10)
    
    train_sessions_limit = cfg["data"].get("train_sessions_limit")
    val_sessions_limit = cfg["data"].get("val_sessions_limit")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup paths
    persistent_root = Path(VOLUME_MOUNT_PATH)
    dataset_dir = persistent_root / "emg2pose_dataset"
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = persistent_root / "react_outputs" / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks for emg2pose expectations
    home_dataset = Path("/root/emg2pose_dataset")
    home_checkpoints = Path("/root/emg2pose_model_checkpoints")
    
    def ensure_symlink(target: Path, link: Path):
        if link.exists() or link.is_symlink():
            if link.is_symlink():
                link.unlink()
            elif link.is_dir():
                import shutil
                shutil.rmtree(link)
        link.symlink_to(target)
    
    # Download dataset if needed
    metadata_file = dataset_dir / "metadata.csv"
    if download_if_missing and not metadata_file.exists():
        print("Downloading full dataset (this may take a while)...")
        cmd = f"curl -L --retry 5 '{FULL_DATASET_URL}' | tar -xvkf - -C '{persistent_root}'"
        subprocess.run(cmd, shell=True, check=True)
        dataset_volume.commit()
    
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
    from src.utils.data import CalibratedEmgDataset
    
    log("=" * 60)
    log("REACT-EMG Full Dataset Training (Modal)")
    log("=" * 60)
    log(f"PyTorch version: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log("")
    log("Training parameters (from config):")
    log(f"  Epochs: {num_epochs}")
    log(f"  Batch size: {batch_size}")
    log(f"  Learning rate: {lr}")
    log(f"  Weight decay: {weight_decay}")
    log("")
    log("Calibration parameters:")
    log(f"  K min: {k_min}")
    log(f"  K max: {k_max}")
    log("")
    log(f"Output dir: {output_dir}")
    log("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load metadata and create splits
    import pandas as pd
    metadata = pd.read_csv(metadata_file)
    
    all_sessions = metadata["session_name"].unique().tolist()
    log(f"Total sessions: {len(all_sessions)}")
    
    # Split by user for proper generalization
    users = metadata["user_id"].unique().tolist()
    np.random.seed(seed)
    np.random.shuffle(users)
    
    n_train = int(0.7 * len(users))
    n_val = int(0.15 * len(users))
    
    train_users = set(users[:n_train])
    val_users = set(users[n_train:n_train + n_val])
    test_users = set(users[n_train + n_val:])
    
    train_sessions = metadata[metadata["user_id"].isin(train_users)]["session_name"].tolist()
    val_sessions = metadata[metadata["user_id"].isin(val_users)]["session_name"].tolist()
    test_sessions = metadata[metadata["user_id"].isin(test_users)]["session_name"].tolist()
    
    # Apply session limits
    if train_sessions_limit:
        train_sessions = train_sessions[:train_sessions_limit]
    if val_sessions_limit:
        val_sessions = val_sessions[:val_sessions_limit]
    
    log(f"Train sessions: {len(train_sessions)} ({len(train_users)} users)")
    log(f"Val sessions: {len(val_sessions)} ({len(val_users)} users)")
    log(f"Test sessions: {len(test_sessions)} ({len(test_users)} users)")
    
    # Create datasets
    train_dataset = CalibratedEmgDataset(
        data_dir=home_dataset,
        session_names=train_sessions,
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"],
        calibration_k=k_max,
        min_calibration_k=k_min,
    )
    
    val_dataset = CalibratedEmgDataset(
        data_dir=home_dataset,
        session_names=val_sessions,
        window_length=cfg["data"]["window_length"],
        stride=cfg["data"]["stride"] * 2,  # Larger stride for validation
        calibration_k=k_max,
        min_calibration_k=k_min,
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
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
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total parameters: {num_params:,}")
    log(f"Trainable parameters: {trainable_params:,}")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay,
        betas=tuple(cfg["training"]["optimizer"]["betas"]),
        eps=cfg["training"]["optimizer"]["eps"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=cfg["training"]["scheduler"]["min_lr"],
    )
    criterion = nn.MSELoss()
    
    # Training loop
    log("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Train
        model.train()
        epoch_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            emg = batch["emg"].to(device)
            targets = batch["joint_angles"].to(device)
            calibration_emg = batch["calibration_emg"].to(device)
            num_cal_samples = batch["calibration_k"].to(device)
            
            optimizer.zero_grad()
            
            try:
                predictions = model.forward_with_raw_calibration(
                    emg=emg,
                    calibration_emg=calibration_emg,
                    num_calibration_samples=num_cal_samples,
                )
                
                # Handle length mismatch
                pred_len = predictions.shape[-1]
                target_len = targets.shape[-1]
                min_len = min(pred_len, target_len)
                predictions = predictions[..., :min_len]
                targets_trimmed = targets[..., :min_len]
                
                loss = criterion(predictions, targets_trimmed)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                epoch_losses.append(loss.item())
                
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
                emg = batch["emg"].to(device)
                targets = batch["joint_angles"].to(device)
                calibration_emg = batch["calibration_emg"].to(device)
                num_cal_samples = batch["calibration_k"].to(device)
                
                try:
                    predictions = model.forward_with_raw_calibration(
                        emg=emg,
                        calibration_emg=calibration_emg,
                        num_calibration_samples=num_cal_samples,
                    )
                    
                    pred_len = predictions.shape[-1]
                    target_len = targets.shape[-1]
                    min_len = min(pred_len, target_len)
                    predictions = predictions[..., :min_len]
                    targets_trimmed = targets[..., :min_len]
                    
                    loss = criterion(predictions, targets_trimmed)
                    val_epoch_losses.append(loss.item())
                except:
                    continue
        
        avg_val_loss = sum(val_epoch_losses) / max(len(val_epoch_losses), 1)
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        epoch_time = time.time() - epoch_start
        
        log(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": cfg,
            }, output_dir / "best_model.pt")
            log(f"  -> New best! Saved to {output_dir / 'best_model.pt'}")
        
        # Commit volume periodically
        if (epoch + 1) % commit_every == 0:
            dataset_volume.commit()
    
    total_time = time.time() - start_time
    
    # Save final model
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "config": cfg,
    }, output_dir / "final_model.pt")
    
    # Save training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
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
        "config": cfg,
    }
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    dataset_volume.commit()
    
    log("\n" + "=" * 60)
    log("Training Complete!")
    log("=" * 60)
    log(f"Total time: {total_time / 60:.1f} minutes")
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
    epochs: int = None,
    batch_size: int = None,
    lr: float = None,
    k_max: int = None,
):
    """Launch REACT-EMG training on Modal.
    
    All parameters are optional - defaults come from configs/experiment/modal.yaml.
    
    Example:
        modal run scripts/train_modal.py
        modal run scripts/train_modal.py --epochs 50 --batch-size 16
    """
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
        config_overrides=overrides if overrides else None,
    )
    print(f"\nTraining completed!")
    print(f"Best val loss: {result['best_val_loss']:.4f}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Output dir: {result['output_dir']}")
