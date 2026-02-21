# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Dummy Data Generator for REACT-EMG.

Generates synthetic EMG and pose data for pipeline testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class DummyDataConfig:
    """Configuration for dummy data generation."""
    num_users: int = 5
    sessions_per_user: int = 4
    samples_per_session: int = 100
    emg_channels: int = 16
    num_joints: int = 20
    sample_rate: int = 2000
    window_size: int = 256
    stride: int = 64
    noise_level: float = 0.1


class DummyDataGenerator:
    """Generate synthetic EMG and pose data for testing."""
    
    def __init__(self, config: Optional[DummyDataConfig] = None, seed: int = 42):
        """Initialize generator.
        
        Args:
            config: Generation configuration.
            seed: Random seed.
        """
        self.config = config or DummyDataConfig()
        self.rng = np.random.default_rng(seed)
        
        # Generate user-specific biases (for user adaptation testing)
        self._user_biases = self._generate_user_characteristics()
    
    def _generate_user_characteristics(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate unique characteristics per user."""
        users = {}
        
        for i in range(self.config.num_users):
            user_id = f"user_{i:03d}"
            users[user_id] = {
                # EMG amplitude scaling per channel
                "emg_scale": 0.5 + self.rng.random(self.config.emg_channels),
                # EMG channel correlation pattern
                "emg_corr": self.rng.random((self.config.emg_channels, self.config.emg_channels)) * 0.2,
                # Joint angle bias
                "joint_bias": (self.rng.random(self.config.num_joints) - 0.5) * 0.3,
                # EMG-to-pose mapping weights
                "mapping_weights": self.rng.random((self.config.num_joints, self.config.emg_channels)),
            }
        
        return users
    
    def generate_emg(
        self,
        user_id: str,
        num_samples: int,
    ) -> np.ndarray:
        """Generate synthetic EMG signal.
        
        Args:
            user_id: User ID for user-specific characteristics.
            num_samples: Number of time samples.
        
        Returns:
            EMG data (channels, time).
        """
        user_chars = self._user_biases.get(user_id, self._user_biases[list(self._user_biases.keys())[0]])
        
        # Base signal: sum of sinusoids at different frequencies
        time = np.arange(num_samples) / self.config.sample_rate
        
        emg = np.zeros((self.config.emg_channels, num_samples))
        
        for ch in range(self.config.emg_channels):
            # Multiple frequency components
            freq1 = 20 + ch * 5
            freq2 = 100 + ch * 10
            
            signal = (
                0.5 * np.sin(2 * np.pi * freq1 * time) +
                0.3 * np.sin(2 * np.pi * freq2 * time) +
                0.2 * np.sin(2 * np.pi * (freq1 + freq2) * time)
            )
            
            # Apply user-specific scaling
            signal *= user_chars["emg_scale"][ch]
            
            # Add noise
            signal += self.rng.normal(0, self.config.noise_level, num_samples)
            
            emg[ch] = signal
        
        # Apply weak channel correlations
        emg = emg + user_chars["emg_corr"] @ emg * 0.1
        
        return emg.astype(np.float32)
    
    def generate_pose(
        self,
        user_id: str,
        emg: np.ndarray,
    ) -> np.ndarray:
        """Generate pose angles from EMG (deterministic mapping).
        
        Args:
            user_id: User ID.
            emg: EMG data (channels, time).
        
        Returns:
            Pose angles (joints, time).
        """
        user_chars = self._user_biases.get(user_id, self._user_biases[list(self._user_biases.keys())[0]])
        
        # Simple linear mapping with user-specific weights
        # Apply RMS envelope first
        window = 50
        emg_envelope = np.zeros_like(emg)
        for i in range(emg.shape[1]):
            start = max(0, i - window // 2)
            end = min(emg.shape[1], i + window // 2)
            emg_envelope[:, i] = np.sqrt(np.mean(emg[:, start:end] ** 2, axis=1))
        
        # Map to pose
        pose = user_chars["mapping_weights"] @ emg_envelope
        
        # Add user bias
        pose = pose + user_chars["joint_bias"][:, np.newaxis]
        
        # Normalize to reasonable range
        pose = np.tanh(pose) * np.pi  # [-pi, pi]
        
        return pose.astype(np.float32)
    
    def generate_session(
        self,
        user_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """Generate a complete session.
        
        Args:
            user_id: User ID.
            session_id: Session ID.
        
        Returns:
            Session dictionary with EMG, pose, and metadata.
        """
        num_samples = self.config.samples_per_session * self.config.window_size
        
        emg = self.generate_emg(user_id, num_samples)
        pose = self.generate_pose(user_id, emg)
        
        return {
            "user_id": user_id,
            "session_id": session_id,
            "emg": emg,
            "pose": pose,
            "sample_rate": self.config.sample_rate,
            "num_channels": self.config.emg_channels,
            "num_joints": self.config.num_joints,
        }
    
    def generate_dataset(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Generate complete dataset with all users and sessions.
        
        Returns:
            Nested dict: {user_id: {session_id: session_data}}.
        """
        dataset = {}
        
        for i in range(self.config.num_users):
            user_id = f"user_{i:03d}"
            dataset[user_id] = {}
            
            for j in range(self.config.sessions_per_user):
                session_id = f"session_{j:03d}"
                dataset[user_id][session_id] = self.generate_session(user_id, session_id)
        
        return dataset
    
    def generate_windowed_samples(
        self,
        user_id: Optional[str] = None,
        num_samples: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """Generate windowed samples for direct DataLoader use.
        
        Args:
            user_id: Specific user or None for random.
            num_samples: Number of windowed samples.
        
        Returns:
            List of sample dictionaries.
        """
        if user_id is None:
            user_id = f"user_{self.rng.integers(self.config.num_users):03d}"
        
        samples = []
        
        for i in range(num_samples):
            session_id = f"session_{i % self.config.sessions_per_user:03d}"
            
            # Generate one window
            emg = self.generate_emg(user_id, self.config.window_size)
            pose = self.generate_pose(user_id, emg)
            
            samples.append({
                "emg": torch.from_numpy(emg),
                "pose": torch.from_numpy(pose),
                "user_id": user_id,
                "session_id": session_id,
            })
        
        return samples


class DummyEmgDataset(Dataset):
    """PyTorch Dataset for dummy EMG data."""
    
    def __init__(
        self,
        config: Optional[DummyDataConfig] = None,
        num_samples: int = 1000,
        seed: int = 42,
    ):
        """Initialize dataset.
        
        Args:
            config: Data configuration.
            num_samples: Total samples in dataset.
            seed: Random seed.
        """
        self.config = config or DummyDataConfig()
        self.generator = DummyDataGenerator(self.config, seed)
        self.num_samples = num_samples
        
        # Pre-generate all samples
        self.samples = self._generate_all_samples()
    
    def _generate_all_samples(self) -> List[Dict[str, Any]]:
        """Pre-generate all samples."""
        samples = []
        
        samples_per_user = self.num_samples // self.config.num_users
        
        for i in range(self.config.num_users):
            user_id = f"user_{i:03d}"
            user_samples = self.generator.generate_windowed_samples(
                user_id=user_id,
                num_samples=samples_per_user,
            )
            samples.extend(user_samples)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class DummyCalibratedDataset(Dataset):
    """Dataset with calibration samples included."""
    
    def __init__(
        self,
        config: Optional[DummyDataConfig] = None,
        num_samples: int = 1000,
        calibration_k: int = 5,
        seed: int = 42,
    ):
        """Initialize dataset.
        
        Args:
            config: Data configuration.
            num_samples: Total samples.
            calibration_k: Number of calibration samples per item.
            seed: Random seed.
        """
        self.config = config or DummyDataConfig()
        self.generator = DummyDataGenerator(self.config, seed)
        self.num_samples = num_samples
        self.calibration_k = calibration_k
        
        # Generate base samples
        self.base_samples = self._generate_base_samples()
        
        # Generate calibration pool per user
        self.calibration_pool = self._generate_calibration_pool()
    
    def _generate_base_samples(self) -> List[Dict[str, Any]]:
        """Generate base samples."""
        samples = []
        samples_per_user = self.num_samples // self.config.num_users
        
        for i in range(self.config.num_users):
            user_id = f"user_{i:03d}"
            user_samples = self.generator.generate_windowed_samples(
                user_id=user_id,
                num_samples=samples_per_user,
            )
            samples.extend(user_samples)
        
        return samples
    
    def _generate_calibration_pool(self) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        """Generate calibration sample pool per user."""
        pool = {}
        
        for i in range(self.config.num_users):
            user_id = f"user_{i:03d}"
            pool[user_id] = self.generator.generate_windowed_samples(
                user_id=user_id,
                num_samples=self.calibration_k * 2,  # Extra for sampling
            )
        
        return pool
    
    def __len__(self) -> int:
        return len(self.base_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base_samples[idx].copy()
        user_id = sample["user_id"]
        
        # Sample calibration data
        pool = self.calibration_pool[user_id]
        cal_indices = torch.randperm(len(pool))[:self.calibration_k]
        
        calibration_emg = torch.stack([pool[i]["emg"] for i in cal_indices])
        
        sample["calibration_emg"] = calibration_emg
        sample["calibration_k"] = self.calibration_k
        
        return sample


def create_dummy_dataloaders(
    config: Optional[DummyDataConfig] = None,
    train_samples: int = 800,
    val_samples: int = 200,
    batch_size: int = 32,
    num_workers: int = 0,
    with_calibration: bool = False,
    calibration_k: int = 5,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders with dummy data.
    
    Args:
        config: Data configuration.
        train_samples: Number of training samples.
        val_samples: Number of validation samples.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        with_calibration: Include calibration samples.
        calibration_k: Number of calibration samples.
    
    Returns:
        Tuple of (train_loader, val_loader).
    """
    config = config or DummyDataConfig()
    
    if with_calibration:
        train_dataset = DummyCalibratedDataset(
            config=config,
            num_samples=train_samples,
            calibration_k=calibration_k,
            seed=42,
        )
        val_dataset = DummyCalibratedDataset(
            config=config,
            num_samples=val_samples,
            calibration_k=calibration_k,
            seed=123,
        )
    else:
        train_dataset = DummyEmgDataset(
            config=config,
            num_samples=train_samples,
            seed=42,
        )
        val_dataset = DummyEmgDataset(
            config=config,
            num_samples=val_samples,
            seed=123,
        )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


def create_dummy_batch(
    batch_size: int = 4,
    emg_channels: int = 16,
    num_joints: int = 20,
    seq_length: int = 256,
    calibration_k: int = 5,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Create a single dummy batch for quick testing.
    
    Args:
        batch_size: Batch size.
        emg_channels: Number of EMG channels.
        num_joints: Number of pose joints.
        seq_length: Sequence length.
        calibration_k: Number of calibration samples.
        device: Target device.
    
    Returns:
        Batch dictionary.
    """
    batch = {
        "emg": torch.randn(batch_size, emg_channels, seq_length, device=device),
        "pose": torch.randn(batch_size, num_joints, seq_length, device=device),
        "calibration_emg": torch.randn(batch_size, calibration_k, emg_channels, seq_length, device=device),
        "calibration_k": torch.full((batch_size,), calibration_k, device=device),
    }
    
    return batch
