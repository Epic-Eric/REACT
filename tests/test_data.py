# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Unit tests for data loading and calibration sampling.
"""

import pytest
import torch

from src.data_loader.calibration import (
    KSamplingStrategy,
    CalibrationConfig,
    CalibrationSampler,
)
from src.utils.data import (
    DataConfig,
    CalibratedEmgDataset,
    create_dataloaders,
    create_dummy_batch,
    check_dataset_available,
    MINI_SPLIT,
    DEFAULT_DATA_DIR,
)


# Check if emg2pose dataset is available for real data tests
HAS_EMG2POSE_DATA = check_dataset_available()


class TestCalibrationConfig:
    """Tests for CalibrationConfig."""
    
    def test_defaults(self):
        """Test default configuration."""
        config = CalibrationConfig()
        assert config.k_min == 0
        assert config.k_max == 30
        assert config.strategy == KSamplingStrategy.UNIFORM
    
    def test_fixed_k(self):
        """Test fixed k configuration."""
        config = CalibrationConfig(
            k_min=10,
            k_max=10,
            strategy=KSamplingStrategy.FIXED,
        )
        assert config.k_min == config.k_max == 10
    
    def test_validation(self):
        """Test config validates k_max >= k_min."""
        # This should work
        config = CalibrationConfig(k_min=5, k_max=10)
        assert config.k_min <= config.k_max


class TestCalibrationSampler:
    """Tests for CalibrationSampler."""
    
    @pytest.fixture
    def mock_registry(self):
        """Create mock user session registry."""
        # Simple dict mimicking registry structure
        return {
            "user_001": ["session_001", "session_002", "session_003", "session_004"],
            "user_002": ["session_005", "session_006", "session_007"],
            "user_003": ["session_008", "session_009"],
        }
    
    def test_sample_k_uniform(self, mock_registry):
        """Test uniform k sampling."""
        config = CalibrationConfig(
            k_min=1,
            k_max=5,
            strategy=KSamplingStrategy.UNIFORM,
        )
        
        # Sample k values many times
        ks = []
        for _ in range(100):
            k = CalibrationSampler._sample_k_value(config)
            ks.append(k)
            assert 1 <= k <= 5
        
        # Should have some variation
        assert len(set(ks)) > 1
    
    def test_sample_k_fixed(self, mock_registry):
        """Test fixed k sampling."""
        config = CalibrationConfig(
            k_min=10,
            k_max=10,
            strategy=KSamplingStrategy.FIXED,
        )
        
        for _ in range(10):
            k = CalibrationSampler._sample_k_value(config)
            assert k == 10


class TestDataConfig:
    """Tests for DataConfig."""
    
    def test_defaults(self):
        """Test default config values."""
        config = DataConfig()
        assert config.window_length == 10_000
        assert config.stride == 2_000
        assert config.batch_size == 32


class TestCreateDummyBatch:
    """Tests for create_dummy_batch utility."""
    
    def test_batch_shapes(self):
        """Test batch has correct shapes."""
        batch = create_dummy_batch(
            batch_size=4,
            emg_channels=16,
            num_joints=20,
            seq_length=10_000,
            calibration_k=5,
        )
        
        assert batch["emg"].shape == (4, 16, 10_000)
        assert batch["joint_angles"].shape == (4, 20, 10_000)
        assert batch["calibration_emg"].shape == (4, 5, 16, 10_000)
    
    def test_device_placement(self):
        """Test batch on correct device."""
        batch = create_dummy_batch(device="cpu")
        
        assert batch["emg"].device.type == "cpu"
        assert batch["joint_angles"].device.type == "cpu"
    
    def test_custom_sizes(self):
        """Test batch with custom sizes."""
        batch = create_dummy_batch(
            batch_size=2,
            emg_channels=8,
            num_joints=10,
            seq_length=5_000,
            calibration_k=3,
        )
        
        assert batch["emg"].shape == (2, 8, 5_000)
        assert batch["joint_angles"].shape == (2, 10, 5_000)
        assert batch["calibration_emg"].shape == (2, 3, 8, 5_000)
    
    def test_no_ik_failure_mask(self):
        """Test no_ik_failure mask is included."""
        batch = create_dummy_batch(batch_size=4, seq_length=1000)
        
        assert "no_ik_failure" in batch
        assert batch["no_ik_failure"].shape == (4, 1000)
        assert batch["no_ik_failure"].dtype == torch.bool


class TestMiniSplit:
    """Tests for mini split configuration."""
    
    def test_split_structure(self):
        """Test mini split has correct structure."""
        assert "train" in MINI_SPLIT
        assert "val" in MINI_SPLIT
        assert "test" in MINI_SPLIT
    
    def test_split_sessions(self):
        """Test split contains sessions."""
        assert len(MINI_SPLIT["train"]) >= 1
        assert len(MINI_SPLIT["val"]) >= 1
        assert len(MINI_SPLIT["test"]) >= 1


@pytest.mark.skipif(not HAS_EMG2POSE_DATA, reason="emg2pose_dataset_mini not available")
class TestCalibratedEmgDataset:
    """Tests for CalibratedEmgDataset (requires real data)."""
    
    def test_dataset_creation(self):
        """Test dataset can be created."""
        dataset = CalibratedEmgDataset(
            data_dir=DEFAULT_DATA_DIR,
            session_names=MINI_SPLIT["train"][:1],  # Just one session
            window_length=10_000,
            stride=10_000,  # No overlap for faster test
            calibration_k=5,
        )
        
        assert len(dataset) > 0
    
    def test_dataset_item_structure(self):
        """Test dataset item has correct structure."""
        dataset = CalibratedEmgDataset(
            data_dir=DEFAULT_DATA_DIR,
            session_names=MINI_SPLIT["train"][:1],
            window_length=10_000,
            stride=10_000,
            calibration_k=5,
        )
        
        item = dataset[0]
        
        assert "emg" in item
        assert "joint_angles" in item
        assert "calibration_emg" in item
        assert "calibration_k" in item
        assert "user_id" in item
    
    def test_calibration_shape(self):
        """Test calibration samples have correct shape."""
        k = 5
        dataset = CalibratedEmgDataset(
            data_dir=DEFAULT_DATA_DIR,
            session_names=MINI_SPLIT["train"][:1],
            window_length=10_000,
            stride=10_000,
            calibration_k=k,
        )
        
        item = dataset[0]
        
        # Calibration should be (K, C, T)
        assert item["calibration_emg"].shape[0] == k


@pytest.mark.skipif(not HAS_EMG2POSE_DATA, reason="emg2pose_dataset_mini not available")
class TestCreateDataloaders:
    """Tests for dataloader creation (requires real data)."""
    
    def test_creates_all_loaders(self):
        """Test creates train, val, and test loaders."""
        train_loader, val_loader, test_loader = create_dataloaders(
            batch_size=4,
            num_workers=0,  # Avoid multiprocessing in tests
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
    
    def test_batch_structure(self):
        """Test batch has correct structure."""
        train_loader, _, _ = create_dataloaders(
            batch_size=2,
            num_workers=0,
        )
        
        batch = next(iter(train_loader))
        
        assert "emg" in batch
        assert "joint_angles" in batch
        assert "calibration_emg" in batch
        assert batch["emg"].dim() == 3  # (B, C, T)


class TestCheckDatasetAvailable:
    """Tests for dataset availability check."""
    
    def test_returns_bool(self):
        """Test function returns boolean."""
        result = check_dataset_available()
        assert isinstance(result, bool)
    
    def test_nonexistent_path(self):
        """Test returns False for nonexistent path."""
        from pathlib import Path
        result = check_dataset_available(Path("/nonexistent/path/to/data"))
        assert result is False

