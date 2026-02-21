# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Signal Processing Utilities for EMG data.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def normalize_emg(
    emg: np.ndarray,
    method: str = "zscore",
    axis: int = -1,
) -> np.ndarray:
    """Normalize EMG signal.
    
    Args:
        emg: EMG signal (channels, time) or (..., channels, time).
        method: Normalization method ('zscore', 'minmax', 'rms').
        axis: Axis for normalization.
    
    Returns:
        Normalized EMG signal.
    """
    if method == "zscore":
        mean = emg.mean(axis=axis, keepdims=True)
        std = emg.std(axis=axis, keepdims=True) + 1e-8
        return (emg - mean) / std
    
    elif method == "minmax":
        min_val = emg.min(axis=axis, keepdims=True)
        max_val = emg.max(axis=axis, keepdims=True)
        return (emg - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "rms":
        rms = np.sqrt(np.mean(emg ** 2, axis=axis, keepdims=True)) + 1e-8
        return emg / rms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def bandpass_filter(
    signal: np.ndarray,
    low_freq: float = 20.0,
    high_freq: float = 500.0,
    sample_rate: float = 2000.0,
    order: int = 4,
) -> np.ndarray:
    """Apply bandpass filter to signal.
    
    Args:
        signal: Input signal (channels, time).
        low_freq: Low cutoff frequency in Hz.
        high_freq: High cutoff frequency in Hz.
        sample_rate: Sampling rate in Hz.
        order: Filter order.
    
    Returns:
        Filtered signal.
    """
    try:
        from scipy.signal import butter, filtfilt
    except ImportError:
        raise ImportError("scipy required for filtering. Install with: pip install scipy")
    
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(order, [low, high], btype='band')
    
    # Apply filter along last axis
    if signal.ndim == 1:
        return filtfilt(b, a, signal)
    else:
        return np.apply_along_axis(lambda x: filtfilt(b, a, x), -1, signal)


def compute_rms(
    emg: np.ndarray,
    window_size: int = 100,
    stride: int = 50,
) -> np.ndarray:
    """Compute RMS envelope of EMG signal.
    
    Args:
        emg: EMG signal (channels, time).
        window_size: Window size in samples.
        stride: Stride between windows.
    
    Returns:
        RMS envelope (channels, time').
    """
    channels, length = emg.shape
    num_windows = (length - window_size) // stride + 1
    
    rms = np.zeros((channels, num_windows))
    
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        rms[:, i] = np.sqrt(np.mean(emg[:, start:end] ** 2, axis=1))
    
    return rms


def compute_envelope(
    emg: np.ndarray,
    method: str = "hilbert",
    lowpass_freq: float = 10.0,
    sample_rate: float = 2000.0,
) -> np.ndarray:
    """Compute EMG envelope.
    
    Args:
        emg: EMG signal (channels, time).
        method: Envelope method ('hilbert', 'rectify_lowpass').
        lowpass_freq: Lowpass cutoff for envelope.
        sample_rate: Sampling rate.
    
    Returns:
        EMG envelope.
    """
    try:
        from scipy.signal import hilbert, butter, filtfilt
    except ImportError:
        raise ImportError("scipy required. Install with: pip install scipy")
    
    if method == "hilbert":
        analytic = hilbert(emg, axis=-1)
        envelope = np.abs(analytic)
        
    elif method == "rectify_lowpass":
        rectified = np.abs(emg)
        
        nyquist = sample_rate / 2
        b, a = butter(4, lowpass_freq / nyquist, btype='low')
        envelope = filtfilt(b, a, rectified, axis=-1)
        
    else:
        raise ValueError(f"Unknown envelope method: {method}")
    
    return envelope


def downsample(
    signal: np.ndarray,
    factor: int,
    method: str = "decimate",
) -> np.ndarray:
    """Downsample signal.
    
    Args:
        signal: Input signal (channels, time).
        factor: Downsampling factor.
        method: Downsampling method ('decimate', 'average', 'subsample').
    
    Returns:
        Downsampled signal.
    """
    if method == "decimate":
        try:
            from scipy.signal import decimate
            return decimate(signal, factor, axis=-1)
        except ImportError:
            method = "average"
    
    if method == "average":
        length = signal.shape[-1]
        new_length = length // factor
        truncated = signal[..., :new_length * factor]
        reshaped = truncated.reshape(signal.shape[:-1] + (new_length, factor))
        return reshaped.mean(axis=-1)
    
    elif method == "subsample":
        return signal[..., ::factor]
    
    raise ValueError(f"Unknown downsampling method: {method}")


def torch_normalize_emg(
    emg: torch.Tensor,
    method: str = "zscore",
    dim: int = -1,
) -> torch.Tensor:
    """Normalize EMG signal (PyTorch version).
    
    Args:
        emg: EMG tensor (..., channels, time).
        method: Normalization method.
        dim: Dimension for normalization.
    
    Returns:
        Normalized tensor.
    """
    if method == "zscore":
        mean = emg.mean(dim=dim, keepdim=True)
        std = emg.std(dim=dim, keepdim=True) + 1e-8
        return (emg - mean) / std
    
    elif method == "minmax":
        min_val = emg.min(dim=dim, keepdim=True).values
        max_val = emg.max(dim=dim, keepdim=True).values
        return (emg - min_val) / (max_val - min_val + 1e-8)
    
    elif method == "rms":
        rms = torch.sqrt(torch.mean(emg ** 2, dim=dim, keepdim=True)) + 1e-8
        return emg / rms
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
