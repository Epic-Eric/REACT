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
    use_dummy_data: bool = False,
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
        use_dummy_data: Use synthetic data for testing.
    
    Returns:
        Dictionary with training results.
    """
    import torch
    
    persistent_root = Path(VOLUME_MOUNT_PATH)
    checkpoints_dir = persistent_root / "emg2pose_model_checkpoints"
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
    if use_pretrained and not use_dummy_data:
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
    
    # Import training components
    sys.path.insert(0, REMOTE_REACT_EMG)
    if REMOTE_EMG2POSE:
        sys.path.insert(0, REMOTE_EMG2POSE)
    
    from src.engine.trainer import DummyTrainer
    from src.models.hybrid_model import FiLMConditionedModelConfig, FiLMConditionedModel
    
    if use_dummy_data:
        print("\nUsing dummy data for pipeline testing...")
        
        # Create and run dummy trainer
        trainer = DummyTrainer(
            emg_channels=16,
            num_joints=20,
            feature_dim=64,
            user_embedding_dim=128,
            calibration_k=calibration_k,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Training loop
        train_losses = []
        for epoch in range(epochs):
            loss = trainer.train_epoch()
            train_losses.append(loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}")
        
        # Save results
        results = {
            "final_loss": train_losses[-1],
            "train_losses": train_losses,
            "epochs_completed": epochs,
            "calibration_k": calibration_k,
        }
        
        # Save checkpoint
        checkpoint_out = output_dir / f"{experiment_name}_final.pt"
        trainer.save_checkpoint(checkpoint_out)
        data_volume.commit()
        
        print(f"\nTraining complete!")
        print(f"Final loss: {results['final_loss']:.4f}")
        print(f"Checkpoint saved: {checkpoint_out}")
        
        return results
    else:
        # Real training with actual data
        print("\nReal training not yet fully implemented.")
        print("Use --use-dummy-data for pipeline testing.")
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
    use_dummy_data: bool = True,
    list_ckpts: bool = False,
):
    """Main entry point for Modal training.
    
    Args:
        epochs: Number of training epochs.
        batch_size: Batch size.
        learning_rate: Learning rate.
        calibration_k: Number of calibration samples.
        experiment_name: Name for this experiment.
        use_dummy_data: Use synthetic data for testing.
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
    print(f"Dummy data: {use_dummy_data}")
    
    results = train_model.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        calibration_k=calibration_k,
        experiment_name=experiment_name,
        use_dummy_data=use_dummy_data,
    )
    
    print("\n" + "=" * 60)
    print("Training Results:")
    print("=" * 60)
    for key, value in results.items():
        if key != "train_losses":
            print(f"  {key}: {value}")
