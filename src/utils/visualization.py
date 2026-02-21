# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Visualization Utilities for REACT-EMG.

Provides plotting functions for training monitoring and result analysis.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import torch


def _check_matplotlib():
    """Check matplotlib availability."""
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    title: str = "Training Progress",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> Any:
    """Plot training and validation loss curves.
    
    Args:
        train_losses: Training loss per epoch.
        val_losses: Optional validation loss per epoch.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    
    if val_losses:
        val_epochs = range(1, len(val_losses) + 1)
        ax.plot(val_epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    joint_idx: int = 0,
    time_range: Optional[tuple] = None,
    title: str = "Pose Predictions",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
) -> Any:
    """Plot predicted vs ground truth joint angles.
    
    Args:
        predictions: Predictions (20, L) or (B, 20, L).
        targets: Ground truth (20, L) or (B, 20, L).
        joint_idx: Which joint to plot.
        time_range: Optional (start, end) time range.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    # Handle batch dimension
    if predictions.ndim == 3:
        predictions = predictions[0]
        targets = targets[0]
    
    pred_joint = predictions[joint_idx]
    target_joint = targets[joint_idx]
    
    if time_range:
        start, end = time_range
        pred_joint = pred_joint[start:end]
        target_joint = target_joint[start:end]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    time = np.arange(len(pred_joint))
    ax.plot(time, target_joint, 'b-', label='Ground Truth', linewidth=1.5, alpha=0.8)
    ax.plot(time, pred_joint, 'r-', label='Prediction', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel('Time (samples)', fontsize=12)
    ax.set_ylabel('Joint Angle', fontsize=12)
    ax.set_title(f'{title} - Joint {joint_idx}', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_all_joints(
    predictions: np.ndarray,
    targets: np.ndarray,
    ncols: int = 4,
    figsize_per_joint: tuple = (3, 2),
    save_path: Optional[Path] = None,
) -> Any:
    """Plot all joints in a grid.
    
    Args:
        predictions: Predictions (20, L).
        targets: Ground truth (20, L).
        ncols: Number of columns in grid.
        figsize_per_joint: Size per subplot.
        save_path: Path to save figure.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    num_joints = predictions.shape[0]
    nrows = (num_joints + ncols - 1) // ncols
    
    figsize = (figsize_per_joint[0] * ncols, figsize_per_joint[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(num_joints):
        ax = axes[i]
        ax.plot(targets[i], 'b-', label='GT', alpha=0.7, linewidth=1)
        ax.plot(predictions[i], 'r-', label='Pred', alpha=0.7, linewidth=1)
        ax.set_title(f'Joint {i}', fontsize=10)
        ax.set_xticks([])
        if i == 0:
            ax.legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(num_joints, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_user_embeddings(
    embeddings: np.ndarray,
    user_ids: List[str],
    method: str = "tsne",
    title: str = "User Embeddings",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
) -> Any:
    """Plot user embeddings using dimensionality reduction.
    
    Args:
        embeddings: User embeddings (N, D).
        user_ids: User IDs for each embedding.
        method: Reduction method ('tsne', 'pca', 'umap').
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    # Reduce to 2D
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, perplexity=min(30, len(embeddings)-1))
        reduced = reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2)
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            raise ImportError("umap-learn required. Install with: pip install umap-learn")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Assign colors to users
    unique_users = list(set(user_ids))
    color_map = {u: i for i, u in enumerate(unique_users)}
    colors = [color_map[u] for u in user_ids]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        reduced[:, 0], reduced[:, 1],
        c=colors, cmap='tab20', alpha=0.7, s=50
    )
    
    ax.set_xlabel(f'{method.upper()} 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Add colorbar with user labels
    if len(unique_users) <= 20:
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(unique_users)))
        cbar.set_ticklabels(unique_users)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_attention(
    attention_weights: np.ndarray,
    recording_names: Optional[List[str]] = None,
    title: str = "Calibration Attention",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 3),
) -> Any:
    """Plot attention weights over calibration recordings.
    
    Args:
        attention_weights: Attention weights (K,).
        recording_names: Optional names for recordings.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    k = len(attention_weights)
    
    if recording_names is None:
        recording_names = [f'R{i+1}' for i in range(k)]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(k), attention_weights, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Recording', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(range(k))
    ax.set_xticklabels(recording_names, rotation=45, ha='right')
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_film_parameters(
    gamma: np.ndarray,
    beta: np.ndarray,
    title: str = "FiLM Parameters",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 4),
) -> Any:
    """Plot FiLM gamma and beta parameters.
    
    Args:
        gamma: Scale parameters (C,).
        beta: Shift parameters (C,).
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    channels = range(len(gamma))
    
    ax1.bar(channels, gamma, color='coral', alpha=0.8)
    ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1)
    ax1.set_xlabel('Channel', fontsize=12)
    ax1.set_ylabel('Gamma (Scale)', fontsize=12)
    ax1.set_title('FiLM Gamma', fontsize=14)
    
    ax2.bar(channels, beta, color='teal', alpha=0.8)
    ax2.axhline(y=0.0, color='gray', linestyle='--', linewidth=1)
    ax2.set_xlabel('Channel', fontsize=12)
    ax2.set_ylabel('Beta (Shift)', fontsize=12)
    ax2.set_title('FiLM Beta', fontsize=14)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_calibration_effect(
    results: Dict[int, Dict[str, float]],
    metric: str = "mae",
    title: str = "Effect of Calibration Amount",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> Any:
    """Plot how metrics change with calibration k.
    
    Args:
        results: Dictionary mapping k values to metrics.
        metric: Which metric to plot.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
    
    Returns:
        Matplotlib figure.
    """
    plt = _check_matplotlib()
    
    k_values = sorted(results.keys())
    metric_values = [results[k][metric] for k in k_values]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_values, metric_values, 'bo-', linewidth=2, markersize=8)
    ax.fill_between(k_values, metric_values, alpha=0.2)
    
    ax.set_xlabel('Number of Calibration Samples (k)', fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
