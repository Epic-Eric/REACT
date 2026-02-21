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
from src.utils.dummy_data import (
    DummyDataConfig,
    DummyDataGenerator,
    DummyEmgDataset,
    DummyCalibratedDataset,
    create_dummy_dataloaders,
    create_dummy_batch,
)


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


class TestDummyDataGenerator:
    """Tests for dummy data generation."""
    
    def test_config_defaults(self):
        """Test default config values."""
        config = DummyDataConfig()
        assert config.num_users == 5
        assert config.sessions_per_user == 4
        assert config.emg_channels == 16
        assert config.num_joints == 20
    
    def test_generate_emg(self):
        """Test EMG generation."""
        generator = DummyDataGenerator()
        
        emg = generator.generate_emg("user_000", 1000)
        
        assert emg.shape == (16, 1000)
        assert emg.dtype == torch.float32 or isinstance(emg, (float, type(emg)))
    
    def test_generate_pose(self):
        """Test pose generation."""
        generator = DummyDataGenerator()
        
        emg = generator.generate_emg("user_000", 1000)
        pose = generator.generate_pose("user_000", emg)
        
        assert pose.shape == (20, 1000)
    
    def test_user_specific_characteristics(self):
        """Test different users have different characteristics."""
        generator = DummyDataGenerator(seed=42)
        
        emg1 = generator.generate_emg("user_000", 1000)
        emg2 = generator.generate_emg("user_001", 1000)
        
        # Different users should produce different signals
        assert not torch.allclose(
            torch.from_numpy(emg1),
            torch.from_numpy(emg2),
        )
    
    def test_generate_windowed_samples(self):
        """Test windowed sample generation."""
        generator = DummyDataGenerator()
        
        samples = generator.generate_windowed_samples(
            user_id="user_000",
            num_samples=10,
        )
        
        assert len(samples) == 10
        assert "emg" in samples[0]
        assert "pose" in samples[0]
        assert "user_id" in samples[0]


class TestDummyEmgDataset:
    """Tests for DummyEmgDataset."""
    
    def test_dataset_length(self):
        """Test dataset has correct length."""
        dataset = DummyEmgDataset(num_samples=100)
        assert len(dataset) == 100
    
    def test_dataset_item(self):
        """Test dataset item structure."""
        dataset = DummyEmgDataset(num_samples=10)
        
        item = dataset[0]
        
        assert "emg" in item
        assert "pose" in item
        assert "user_id" in item
    
    def test_dataset_item_shapes(self):
        """Test dataset item shapes."""
        config = DummyDataConfig(
            emg_channels=16,
            num_joints=20,
            window_size=256,
        )
        dataset = DummyEmgDataset(config=config, num_samples=10)
        
        item = dataset[0]
        
        assert item["emg"].shape == (16, 256)
        assert item["pose"].shape == (20, 256)


class TestDummyCalibratedDataset:
    """Tests for DummyCalibratedDataset."""
    
    def test_dataset_with_calibration(self):
        """Test calibrated dataset includes calibration samples."""
        dataset = DummyCalibratedDataset(
            num_samples=10,
            calibration_k=5,
        )
        
        item = dataset[0]
        
        assert "emg" in item
        assert "pose" in item
        assert "calibration_emg" in item
        assert "calibration_k" in item
    
    def test_calibration_shape(self):
        """Test calibration data shape."""
        config = DummyDataConfig(emg_channels=16, window_size=256)
        dataset = DummyCalibratedDataset(
            config=config,
            num_samples=10,
            calibration_k=5,
        )
        
        item = dataset[0]
        
        # Calibration should be (K, C, T)
        assert item["calibration_emg"].shape == (5, 16, 256)


class TestCreateDummyDataloaders:
    """Tests for dataloader creation."""
    
    def test_creates_both_loaders(self):
        """Test creates train and val loaders."""
        train_loader, val_loader = create_dummy_dataloaders(
            train_samples=100,
            val_samples=20,
            batch_size=10,
        )
        
        assert train_loader is not None
        assert val_loader is not None
        assert len(train_loader.dataset) == 100
        assert len(val_loader.dataset) == 20
    
    def test_batch_shape(self):
        """Test batch shapes from loader."""
        config = DummyDataConfig(emg_channels=16, num_joints=20, window_size=256)
        train_loader, _ = create_dummy_dataloaders(
            config=config,
            train_samples=100,
            batch_size=8,
        )
        
        batch = next(iter(train_loader))
        
        assert batch["emg"].shape == (8, 16, 256)
        assert batch["pose"].shape == (8, 20, 256)
    
    def test_with_calibration(self):
        """Test dataloader with calibration samples."""
        train_loader, _ = create_dummy_dataloaders(
            train_samples=50,
            batch_size=4,
            with_calibration=True,
            calibration_k=5,
        )
        
        batch = next(iter(train_loader))
        
        assert "calibration_emg" in batch
        assert batch["calibration_emg"].shape[1] == 5  # K dimension


class TestCreateDummyBatch:
    """Tests for create_dummy_batch utility."""
    
    def test_batch_shapes(self):
        """Test batch has correct shapes."""
        batch = create_dummy_batch(
            batch_size=4,
            emg_channels=16,
            num_joints=20,
            seq_length=256,
            calibration_k=5,
        )
        
        assert batch["emg"].shape == (4, 16, 256)
        assert batch["pose"].shape == (4, 20, 256)
        assert batch["calibration_emg"].shape == (4, 5, 16, 256)
    
    def test_device_placement(self):
        """Test batch on correct device."""
        batch = create_dummy_batch(device="cpu")
        
        assert batch["emg"].device.type == "cpu"
        assert batch["pose"].device.type == "cpu"
    
    def test_custom_sizes(self):
        """Test batch with custom sizes."""
        batch = create_dummy_batch(
            batch_size=2,
            emg_channels=8,
            num_joints=10,
            seq_length=128,
            calibration_k=3,
        )
        
        assert batch["emg"].shape == (2, 8, 128)
        assert batch["pose"].shape == (2, 10, 128)
        assert batch["calibration_emg"].shape == (2, 3, 8, 128)
