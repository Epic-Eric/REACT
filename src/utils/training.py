#!/usr/bin/env python3
# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Shared training utilities for REACT-EMG.

Contains config loading, logging, plotting, and training helpers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Visualization
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# =============================================================================
# Config Loading
# =============================================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely get nested config value.
    
    Args:
        cfg: Configuration dictionary.
        *keys: Nested keys to access.
        default: Default value if key not found.
        
    Returns:
        Config value or default.
    """
    value = cfg
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


# =============================================================================
# Logging
# =============================================================================

class TrainingLogger:
    """Simple training logger that writes to file and stdout."""
    
    def __init__(self, output_dir: Path, filename: str = "training.log"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = output_dir / filename
    
    def log(self, message: str, also_print: bool = True):
        """Log a message with timestamp."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        with open(self.log_file, "a") as f:
            f.write(full_message + "\n")
        
        if also_print:
            print(full_message)
    
    def log_config(self, cfg: Dict[str, Any], title: str = "Configuration"):
        """Log configuration dictionary."""
        self.log(f"\n{title}:")
        self.log("-" * 40)
        self._log_dict(cfg, indent=2)
        self.log("-" * 40)
    
    def _log_dict(self, d: Dict[str, Any], indent: int = 0):
        """Recursively log dictionary."""
        prefix = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                self.log(f"{prefix}{key}:")
                self._log_dict(value, indent + 2)
            else:
                self.log(f"{prefix}{key}: {value}")


# =============================================================================
# Plotting
# =============================================================================

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    output_path: Path,
    title: str = "Training Progress",
):
    """Plot and save training curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add best val loss annotation
    if val_losses:
        best_epoch = int(np.argmin(val_losses)) + 1
        best_val = min(val_losses)
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
        ax.annotate(f'Best: {best_val:.4f} (ep {best_epoch})',
                   xy=(best_epoch, best_val),
                   xytext=(best_epoch + 2, best_val + 0.01),
                   fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_predictions(
    emg: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    output_path: Path,
    sample_idx: int = 0,
):
    """Plot EMG input, target poses, and predictions."""
    # Move to CPU and convert
    emg_np = emg[sample_idx].cpu().numpy()
    targets_np = targets[sample_idx].cpu().numpy()
    preds_np = predictions[sample_idx].cpu().detach().numpy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # EMG signal (first 4 channels)
    ax1 = axes[0]
    for ch in range(min(4, emg_np.shape[0])):
        ax1.plot(emg_np[ch, ::10], label=f'Ch {ch+1}', alpha=0.7)
    ax1.set_title('EMG Input (4 channels)', fontsize=12)
    ax1.set_xlabel('Time (samples / 10)')
    ax1.legend(loc='upper right', ncol=4, fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Target joint angles (first 5 joints)
    ax2 = axes[1]
    for j in range(min(5, targets_np.shape[0])):
        ax2.plot(targets_np[j, ::10], label=f'Joint {j+1}', alpha=0.7)
    ax2.set_title('Target Joint Angles (5 joints)', fontsize=12)
    ax2.set_xlabel('Time (samples / 10)')
    ax2.legend(loc='upper right', ncol=5, fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Predictions vs targets for one joint
    ax3 = axes[2]
    joint_idx = 0
    ax3.plot(targets_np[joint_idx, ::10], 'b-', label='Target', linewidth=2, alpha=0.7)
    ax3.plot(preds_np[joint_idx, ::10], 'r--', label='Prediction', linewidth=2, alpha=0.7)
    ax3.set_title(f'Joint {joint_idx + 1}: Target vs Prediction', fontsize=12)
    ax3.set_xlabel('Time (samples / 10)')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Device Selection
# =============================================================================

def get_device(device_str: str = "auto") -> torch.device:
    """Get torch device from string.
    
    Args:
        device_str: "auto", "cuda", "mps", or "cpu"
        
    Returns:
        torch.device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


# =============================================================================
# Training Helpers
# =============================================================================

def save_checkpoint(
    path: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
):
    """Save training checkpoint.
    
    Args:
        path: Output path.
        epoch: Current epoch.
        model: Model to save.
        optimizer: Optimizer to save.
        val_loss: Validation loss.
        config: Training config.
        extra: Additional data to save.
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "config": config,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)


def save_history(
    path: Path,
    train_losses: List[float],
    val_losses: List[float],
    best_val_loss: float,
    best_epoch: int,
    total_time: float,
    config: Dict[str, Any],
):
    """Save training history as JSON.
    
    Args:
        path: Output path.
        train_losses: List of training losses.
        val_losses: List of validation losses.
        best_val_loss: Best validation loss.
        best_epoch: Best epoch.
        total_time: Total training time in seconds.
        config: Training config.
    """
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_time_minutes": total_time / 60,
        "config": config,
    }
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def compute_loss_with_length_match(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    criterion: torch.nn.Module,
) -> torch.Tensor:
    """Compute loss handling length mismatch from encoder downsampling.
    
    Args:
        predictions: Model predictions (B, C, L1).
        targets: Ground truth targets (B, C, L2).
        criterion: Loss function.
        
    Returns:
        Loss tensor.
    """
    pred_len = predictions.shape[-1]
    target_len = targets.shape[-1]
    min_len = min(pred_len, target_len)
    return criterion(predictions[..., :min_len], targets[..., :min_len])
