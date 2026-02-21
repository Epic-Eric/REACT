#!/usr/bin/env python
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Modal Training Script for REACT-EMG.

Trains FiLM-conditioned user-adaptive model on Modal cloud infrastructure.

Usage:
    modal run scripts/modal_train.py --epochs 100 --batch-size 32
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import modal


def _find_local_react_emg() -> Path:
    """Find local REACT package directory."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent,  # scripts/ -> REACT/
        here.parent.parent / "REACT",
        Path.cwd() / "REACT",
        Path.cwd(),  # Already in REACT directory
    ]
    for candidate in candidates:
        if (candidate / "src").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find REACT package. Run from the REACT directory."
    )


def _find_emg2pose_repo() -> Path:
    """Find emg2pose repository for pretrained encoder."""
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent / "emg2pose",  # REACT/emg2pose (submodule)
        here.parent.parent / "emg2pose",
        Path.cwd() / "emg2pose",
    ]
    for candidate in candidates:
        if (candidate / "setup.py").exists() and (candidate / "emg2pose").exists():
            return candidate
    raise FileNotFoundError("Could not find emg2pose repository at REACT/emg2pose.")


# Paths
LOCAL_REACT_EMG = _find_local_react_emg()
REMOTE_REACT_EMG = "/root/react_emg"

try:
    LOCAL_EMG2POSE = _find_emg2pose_repo()
    REMOTE_EMG2POSE = "/root/emg2pose"
except FileNotFoundError:
    LOCAL_EMG2POSE = None
    REMOTE_EMG2POSE = None

VOLUME_MOUNT_PATH = "/persistent"
CHECKPOINTS_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_model_checkpoints.tar.gz"
)
DATASET_URL = (
    "https://fb-ctrl-oss.s3.amazonaws.com/emg2pose/emg2pose_dataset_mini.tar"
)

# Create Modal app
app = modal.App("react-emg-training")
data_volume = modal.Volume.from_name("react-emg-data", create_if_missing=True)

# Build image with dependencies
image_builder = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "tar", "git")
    .pip_install(
        # Core ML
        "torch==2.3.1",
        "pytorch-lightning==2.2.2",
        # Data handling
        "numpy==1.26.4",
        "scipy==1.13.1",
        "h5py==3.11.0",
        "pandas==2.2.2",
        # Config & CLI
        "pyyaml==6.0.1",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "typer==0.12.3",
        "rich==13.7.1",
        # Visualization
        "matplotlib==3.8.4",
        "seaborn==0.13.2",
        # Progress
        "tqdm==4.66.4",
    )
    .add_local_dir(str(LOCAL_REACT_EMG), remote_path=REMOTE_REACT_EMG)
)

# Add emg2pose if available
if LOCAL_EMG2POSE:
    image_builder = image_builder.add_local_dir(
        str(LOCAL_EMG2POSE), remote_path=REMOTE_EMG2POSE
    )

image = image_builder


def _run(cmd: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    """Run command and check for errors."""
    print(f"$ {' '.join(cmd)}")
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=full_env, check=True)


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: data_volume},
    gpu="T4",  # Use T4 GPU, can be upgraded to A10G, A100, etc.
    timeout=60 * 60 * 12,  # 12 hour timeout
    memory=16384,  # 16GB RAM
)
def train_model(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    calibration_k: int = 10,
    experiment_name: str = "film_adaptive",
    use_pretrained: bool = True,
    pretrained_checkpoint: str = "tracking_vemg2pose.ckpt",
) -> dict:
    """Train FiLM-conditioned model on Modal.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        calibration_k: Number of calibration samples.
        experiment_name: Name for this experiment run.
        use_pretrained: Whether to load pretrained encoder.
        pretrained_checkpoint: Name of pretrained checkpoint file.
    
    Returns:
        Dictionary with training results.
    """
    import torch
    
    persistent_root = Path(VOLUME_MOUNT_PATH)
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
    dataset_dir = persistent_root / "emg2pose_dataset_mini"
    output_dir = persistent_root / "react_outputs" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup PYTHONPATH
    pythonpath = REMOTE_REACT_EMG
    if REMOTE_EMG2POSE:
        pythonpath = f"{REMOTE_REACT_EMG}:{REMOTE_EMG2POSE}"
    
    env = {"PYTHONPATH": pythonpath}
    
    print("=" * 60)
    print("REACT-EMG Modal Training")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Calibration K: {calibration_k}")
    print("=" * 60)
    
    # Download pretrained checkpoints if needed
    if use_pretrained:
        checkpoints_archive = persistent_root / "emg2pose_model_checkpoints.tar.gz"
        checkpoint_path = checkpoints_dir / pretrained_checkpoint
        
        if not checkpoint_path.exists():
            if not checkpoints_archive.exists():
                print("Downloading pretrained checkpoints...")
                _run(["curl", "-L", CHECKPOINTS_URL, "-o", str(checkpoints_archive)])
            print("Extracting checkpoints...")
            _run(["tar", "-xvzf", str(checkpoints_archive), "-C", str(persistent_root)])
            data_volume.commit()
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Using pretrained encoder: {checkpoint_path}")
    
    # Download dataset if needed
    dataset_archive = persistent_root / "emg2pose_dataset_mini.tar"
    if not dataset_dir.exists() or not any(dataset_dir.glob("*.hdf5")):
        if not dataset_archive.exists():
            print("Downloading emg2pose_dataset_mini...")
            _run(["curl", "-L", DATASET_URL, "-o", str(dataset_archive)])
        print("Extracting dataset...")
        _run(["tar", "-xvf", str(dataset_archive), "-C", str(persistent_root)])
        data_volume.commit()
    
    if not any(dataset_dir.glob("*.hdf5")):
        raise FileNotFoundError(f"Dataset not found at {dataset_dir}")
    
    print(f"Using dataset: {dataset_dir}")
    
    # Import training components
    sys.path.insert(0, REMOTE_REACT_EMG)
    if REMOTE_EMG2POSE:
        sys.path.insert(0, REMOTE_EMG2POSE)
    
    from src.utils.data import create_dataloaders
    from src.engine.trainer import Trainer, TrainerConfig
    from src.models.hybrid_model import FiLMConditionedModelConfig, FiLMConditionedModel
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=4,
        calibration_k=calibration_k,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = FiLMConditionedModelConfig(
        emg_channels=16,
        feature_dim=64,
        user_embedding_dim=128,
        num_joints=20,
    )
    model = FiLMConditionedModel(model_config).to(device)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Train
        model.train()
        epoch_losses = []
        for batch in train_loader:
            emg = batch["emg"].to(device)
            targets = batch["joint_angles"].to(device)
            calibration_emg = batch["calibration_emg"].to(device)
            
            optimizer.zero_grad()
            output = model(
                encoded_features=emg,
                calibration_features=calibration_emg,
            )
            loss = criterion(output["predictions"], targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_losses) / len(epoch_losses)
        train_losses.append(avg_train_loss)
        
        # Validate
        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                emg = batch["emg"].to(device)
                targets = batch["joint_angles"].to(device)
                calibration_emg = batch["calibration_emg"].to(device)
                
                output = model(
                    encoded_features=emg,
                    calibration_features=calibration_emg,
                )
                loss = criterion(output["predictions"], targets)
                val_epoch_losses.append(loss.item())
        
        avg_val_loss = sum(val_epoch_losses) / len(val_epoch_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = output_dir / f"{experiment_name}_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, best_checkpoint)
    
    # Save final checkpoint
    final_checkpoint = output_dir / f"{experiment_name}_final.pt"
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, final_checkpoint)
    data_volume.commit()
    
    results = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val_loss,
        "epochs_completed": epochs,
        "calibration_k": calibration_k,
    }
    
    print(f"\nTraining complete!")
    print(f"Final train loss: {results['final_train_loss']:.4f}")
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Checkpoints saved: {output_dir}")
    
    return results
        return {"status": "not_implemented"}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: data_volume},
    gpu="T4",
    timeout=60 * 60 * 2,
)
def evaluate_model(
    checkpoint_path: str,
    calibration_k: int = 10,
) -> dict:
    """Evaluate trained model.
    
    Args:
        checkpoint_path: Path to model checkpoint.
        calibration_k: Number of calibration samples.
    
    Returns:
        Evaluation metrics.
    """
    import torch
    
    persistent_root = Path(VOLUME_MOUNT_PATH)
    full_checkpoint_path = persistent_root / checkpoint_path
    
    if not full_checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {full_checkpoint_path}")
    
    print(f"Evaluating checkpoint: {full_checkpoint_path}")
    print(f"Calibration K: {calibration_k}")
    
    # Evaluation would be implemented here
    return {"status": "not_implemented"}


@app.function(
    image=image,
    volumes={VOLUME_MOUNT_PATH: data_volume},
)
def list_checkpoints() -> list[str]:
    """List available checkpoints in persistent storage."""
    persistent_root = Path(VOLUME_MOUNT_PATH)
    react_outputs = persistent_root / "react_outputs"
    
    checkpoints = []
    if react_outputs.exists():
        for ckpt in react_outputs.rglob("*.pt"):
            checkpoints.append(str(ckpt.relative_to(persistent_root)))
    
    # Also list emg2pose checkpoints
    emg2pose_ckpts = persistent_root / "emg2pose_model_checkpoints"
    if emg2pose_ckpts.exists():
        for ckpt in emg2pose_ckpts.glob("*.ckpt"):
            checkpoints.append(str(ckpt.relative_to(persistent_root)))
    
    return checkpoints


@app.local_entrypoint()
def main(
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    calibration_k: int = 10,
    experiment_name: str = "film_adaptive",
    list_ckpts: bool = False,
):
    """Main entry point for Modal training.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        calibration_k: Number of calibration samples.
        experiment_name: Name for this experiment.
        list_ckpts: List available checkpoints and exit.
    """
    if list_ckpts:
        checkpoints = list_checkpoints.remote()
        print("Available checkpoints:")
        for ckpt in checkpoints:
            print(f"  - {ckpt}")
        return
    
    print("Starting REACT-EMG training on Modal...")
    print(f"Experiment: {experiment_name}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}")
    print(f"Calibration K: {calibration_k}")
    
    results = train_model.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        calibration_k=calibration_k,
        experiment_name=experiment_name,
    )
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    for key, value in results.items():
        if key not in ("train_losses", "val_losses"):
            print(f"  {key}: {value}")
