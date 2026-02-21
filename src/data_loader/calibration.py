# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Calibration Sampling for User-Adaptive Training.

Manages selection of calibration recordings from the same user,
excluding the current training recording.

Key functionality:
- Build registry of user sessions
- Sample k recordings for calibration (k ∈ [0, max_k])
- Handle edge cases (users with few recordings)
- Support configurable sampling distributions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from pathlib import Path
from enum import Enum
import random

import numpy as np
import torch
import h5py


class KSamplingStrategy(Enum):
    """Strategy for sampling k (number of calibration samples)."""
    UNIFORM = "uniform"      # k ~ Uniform(min_k, max_k)
    TRUNCATED_NORMAL = "truncated_normal"  # k ~ TruncNorm centered at max_k/2
    FIXED = "fixed"          # k = fixed_k always
    WEIGHTED = "weighted"    # k sampled from custom weights


@dataclass
class CalibrationConfig:
    """Configuration for calibration sampling.
    
    Attributes:
        min_k: Minimum number of calibration samples (default 0).
        max_k: Maximum number of calibration samples (default 30).
        sampling_strategy: How to sample k.
        fixed_k: Fixed k value (for FIXED strategy).
        truncnorm_std: Standard deviation for truncated normal sampling.
        custom_weights: Custom sampling weights for k values.
        exclude_current: Whether to exclude current recording from samples.
        max_calibration_length: Maximum length for calibration recordings.
        random_crop: Whether to randomly crop calibration recordings.
    """
    min_k: int = 0
    max_k: int = 30
    sampling_strategy: KSamplingStrategy = KSamplingStrategy.UNIFORM
    fixed_k: int = 15
    truncnorm_std: float = 8.0
    custom_weights: Optional[List[float]] = None
    exclude_current: bool = True
    max_calibration_length: int = 20000  # ~10 seconds at 2kHz
    random_crop: bool = True


class UserSessionRegistry:
    """Registry mapping users to their recording sessions.
    
    Maintains an index of which sessions belong to each user
    for efficient calibration sampling.
    """
    
    def __init__(self):
        self.user_to_sessions: Dict[str, List[str]] = {}
        self.session_to_user: Dict[str, str] = {}
        self.session_to_path: Dict[str, Path] = {}
        self.session_to_idx: Dict[str, int] = {}
    
    def register(
        self,
        session_name: str,
        user_id: str,
        hdf5_path: Path,
        dataset_idx: int,
    ):
        """Register a session in the registry.
        
        Args:
            session_name: Unique session identifier.
            user_id: User ID this session belongs to.
            hdf5_path: Path to HDF5 file containing session data.
            dataset_idx: Index in the dataset for this session.
        """
        if user_id not in self.user_to_sessions:
            self.user_to_sessions[user_id] = []
        
        if session_name not in self.session_to_user:
            self.user_to_sessions[user_id].append(session_name)
        
        self.session_to_user[session_name] = user_id
        self.session_to_path[session_name] = hdf5_path
        self.session_to_idx[session_name] = dataset_idx
    
    def get_user_sessions(
        self,
        user_id: str,
        exclude: Optional[str] = None,
    ) -> List[str]:
        """Get all sessions for a user.
        
        Args:
            user_id: User ID to query.
            exclude: Optional session to exclude from results.
        
        Returns:
            List of session names.
        """
        sessions = self.user_to_sessions.get(user_id, [])
        if exclude:
            sessions = [s for s in sessions if s != exclude]
        return sessions
    
    def get_session_user(self, session_name: str) -> Optional[str]:
        """Get user ID for a session."""
        return self.session_to_user.get(session_name)
    
    def get_session_path(self, session_name: str) -> Optional[Path]:
        """Get HDF5 path for a session."""
        return self.session_to_path.get(session_name)
    
    def get_all_users(self) -> List[str]:
        """Get list of all users."""
        return list(self.user_to_sessions.keys())
    
    def get_user_recording_counts(self) -> Dict[str, int]:
        """Get number of recordings per user."""
        return {
            user: len(sessions) 
            for user, sessions in self.user_to_sessions.items()
        }
    
    @classmethod
    def from_session_files(
        cls,
        session_paths: List[Path],
        user_key: str = "user",
        session_key: str = "session",
    ) -> "UserSessionRegistry":
        """Build registry from list of HDF5 session files.
        
        Args:
            session_paths: List of paths to HDF5 session files.
            user_key: Attribute key for user ID in HDF5 metadata.
            session_key: Attribute key for session name in HDF5 metadata.
        
        Returns:
            Populated UserSessionRegistry.
        """
        registry = cls()
        
        for idx, path in enumerate(session_paths):
            try:
                with h5py.File(path, 'r') as f:
                    emg2pose_group = f['emg2pose']
                    user_id = emg2pose_group.attrs.get(user_key, f"user_{idx}")
                    session_name = emg2pose_group.attrs.get(session_key, f"session_{idx}")
                    
                    # Convert bytes to string if needed
                    if isinstance(user_id, bytes):
                        user_id = user_id.decode('utf-8')
                    if isinstance(session_name, bytes):
                        session_name = session_name.decode('utf-8')
                    
                    registry.register(
                        session_name=session_name,
                        user_id=user_id,
                        hdf5_path=path,
                        dataset_idx=idx,
                    )
            except Exception as e:
                print(f"Warning: Could not read {path}: {e}")
        
        return registry
    
    def __len__(self) -> int:
        return len(self.session_to_user)
    
    def __repr__(self) -> str:
        return (
            f"UserSessionRegistry("
            f"users={len(self.user_to_sessions)}, "
            f"sessions={len(self.session_to_user)})"
        )


class CalibrationSampler:
    """Samples calibration recordings for user-adaptive training.
    
    Given a recording, samples k other recordings from the same user
    to use as calibration context for FiLM conditioning.
    """
    
    def __init__(
        self,
        config: CalibrationConfig,
        registry: UserSessionRegistry,
        encoder: Optional[torch.nn.Module] = None,
    ):
        """Initialize sampler.
        
        Args:
            config: Calibration configuration.
            registry: User session registry.
            encoder: Optional pretrained encoder for pre-encoding samples.
        """
        self.config = config
        self.registry = registry
        self.encoder = encoder
        
        # Cache for pre-loaded/encoded calibration data
        self._cache: Dict[str, torch.Tensor] = {}
        self._loaded_sessions: Dict[str, np.ndarray] = {}
    
    def sample_k(self, max_available: Optional[int] = None) -> int:
        """Sample number of calibration recordings to use.
        
        Args:
            max_available: Maximum available recordings (caps the sample).
        
        Returns:
            Number of calibration samples k.
        """
        max_k = self.config.max_k
        if max_available is not None:
            max_k = min(max_k, max_available)
        
        if max_k <= self.config.min_k:
            return self.config.min_k
        
        strategy = self.config.sampling_strategy
        
        if strategy == KSamplingStrategy.FIXED:
            return min(self.config.fixed_k, max_k)
        
        elif strategy == KSamplingStrategy.UNIFORM:
            return random.randint(self.config.min_k, max_k)
        
        elif strategy == KSamplingStrategy.TRUNCATED_NORMAL:
            mean = (self.config.max_k + self.config.min_k) / 2
            std = self.config.truncnorm_std
            k = int(np.clip(
                np.random.normal(mean, std),
                self.config.min_k,
                max_k,
            ))
            return k
        
        elif strategy == KSamplingStrategy.WEIGHTED:
            if self.config.custom_weights is None:
                return random.randint(self.config.min_k, max_k)
            weights = self.config.custom_weights[:max_k - self.config.min_k + 1]
            weights = np.array(weights) / sum(weights)
            k = np.random.choice(
                range(self.config.min_k, max_k + 1),
                p=weights,
            )
            return k
        
        return self.config.min_k
    
    def sample_calibration_sessions(
        self,
        session_name: str,
        k: Optional[int] = None,
    ) -> List[str]:
        """Sample calibration sessions for a given session.
        
        Args:
            session_name: Current session name.
            k: Number of sessions to sample. If None, samples k randomly.
        
        Returns:
            List of calibration session names.
        """
        user_id = self.registry.get_session_user(session_name)
        if user_id is None:
            return []
        
        # Get available sessions
        exclude = session_name if self.config.exclude_current else None
        available_sessions = self.registry.get_user_sessions(user_id, exclude=exclude)
        
        if not available_sessions:
            return []
        
        # Sample k
        if k is None:
            k = self.sample_k(max_available=len(available_sessions))
        else:
            k = min(k, len(available_sessions))
        
        if k == 0:
            return []
        
        # Random sample without replacement
        return random.sample(available_sessions, k)
    
    def load_session_emg(
        self,
        session_name: str,
        max_length: Optional[int] = None,
        random_crop: bool = True,
    ) -> Optional[np.ndarray]:
        """Load EMG data for a session.
        
        Args:
            session_name: Session to load.
            max_length: Maximum length to load.
            random_crop: Whether to randomly crop if exceeding max_length.
        
        Returns:
            EMG data as numpy array (16, L) or None if failed.
        """
        if session_name in self._loaded_sessions:
            emg = self._loaded_sessions[session_name]
        else:
            path = self.registry.get_session_path(session_name)
            if path is None:
                return None
            
            try:
                with h5py.File(path, 'r') as f:
                    emg = f['emg2pose']['timeseries']['emg'][:]  # (L, 16)
                    emg = emg.T  # (16, L)
                    self._loaded_sessions[session_name] = emg
            except Exception as e:
                print(f"Warning: Could not load {session_name}: {e}")
                return None
        
        max_length = max_length or self.config.max_calibration_length
        
        if emg.shape[1] > max_length:
            if random_crop and self.config.random_crop:
                start = random.randint(0, emg.shape[1] - max_length)
            else:
                start = 0
            emg = emg[:, start:start + max_length]
        
        return emg
    
    def get_calibration_batch(
        self,
        session_name: str,
        k: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get padded calibration batch for a session.
        
        Args:
            session_name: Current session name.
            k: Number of calibration samples. None for random.
            device: Device for tensors.
        
        Returns:
            Tuple of:
            - calibration_emg: (K_actual, 16, L_max)
            - num_samples: scalar tensor with K_actual
            - lengths: (K_actual,) actual lengths before padding
        """
        sessions = self.sample_calibration_sessions(session_name, k)
        
        if not sessions:
            # Return empty batch
            dummy = torch.zeros(1, 16, 1)
            if device:
                dummy = dummy.to(device)
            return dummy, torch.tensor(0), torch.tensor([0])
        
        # Load EMG for each session
        emg_list = []
        lengths = []
        
        for sess in sessions:
            emg = self.load_session_emg(sess)
            if emg is not None:
                emg_list.append(emg)
                lengths.append(emg.shape[1])
        
        if not emg_list:
            dummy = torch.zeros(1, 16, 1)
            if device:
                dummy = dummy.to(device)
            return dummy, torch.tensor(0), torch.tensor([0])
        
        # Pad to uniform length
        max_len = max(lengths)
        padded = []
        for emg in emg_list:
            if emg.shape[1] < max_len:
                pad = np.zeros((16, max_len - emg.shape[1]))
                emg = np.concatenate([emg, pad], axis=1)
            padded.append(emg)
        
        calibration_emg = torch.tensor(np.stack(padded), dtype=torch.float32)
        num_samples = torch.tensor(len(emg_list))
        lengths_tensor = torch.tensor(lengths)
        
        if device:
            calibration_emg = calibration_emg.to(device)
            num_samples = num_samples.to(device)
            lengths_tensor = lengths_tensor.to(device)
        
        return calibration_emg, num_samples, lengths_tensor
    
    def clear_cache(self):
        """Clear loaded session cache."""
        self._loaded_sessions.clear()
        self._cache.clear()


class BatchCalibrationSampler:
    """Samples calibration data for entire batches efficiently.
    
    Optimized for batch training where multiple samples may come
    from the same user.
    """
    
    def __init__(
        self,
        config: CalibrationConfig,
        registry: UserSessionRegistry,
    ):
        self.config = config
        self.registry = registry
        self.sampler = CalibrationSampler(config, registry)
    
    def sample_batch(
        self,
        session_names: List[str],
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample calibration data for a batch.
        
        Args:
            session_names: List of B session names in the batch.
            device: Device for tensors.
        
        Returns:
            Dictionary with:
            - calibration_emg: (B, K_max, 16, L_max)
            - num_samples: (B,)
            - lengths: (B, K_max)
        """
        batch_size = len(session_names)
        
        # Sample for each item
        all_emg = []
        all_num_samples = []
        all_lengths = []
        
        k_max = 0
        l_max = 0
        
        for session in session_names:
            emg, num_samples, lengths = self.sampler.get_calibration_batch(
                session, device=None  # Don't move to device yet
            )
            all_emg.append(emg)
            all_num_samples.append(num_samples.item())
            all_lengths.append(lengths)
            
            k_max = max(k_max, emg.shape[0])
            l_max = max(l_max, emg.shape[2])
        
        # Pad to uniform shape
        padded_emg = torch.zeros(batch_size, k_max, 16, l_max)
        padded_lengths = torch.zeros(batch_size, k_max, dtype=torch.long)
        
        for i, (emg, lengths) in enumerate(zip(all_emg, all_lengths)):
            k, c, l = emg.shape
            padded_emg[i, :k, :, :l] = emg
            padded_lengths[i, :len(lengths)] = lengths
        
        num_samples = torch.tensor(all_num_samples, dtype=torch.long)
        
        if device:
            padded_emg = padded_emg.to(device)
            num_samples = num_samples.to(device)
            padded_lengths = padded_lengths.to(device)
        
        return {
            'calibration_emg': padded_emg,
            'num_calibration_samples': num_samples,
            'calibration_lengths': padded_lengths,
        }
