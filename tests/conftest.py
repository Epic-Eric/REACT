# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest
import torch


# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture(scope="session")
def device():
    """Get available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@pytest.fixture
def sample_emg():
    """Create sample EMG data."""
    return torch.randn(4, 16, 256)


@pytest.fixture
def sample_pose():
    """Create sample pose data."""
    return torch.randn(4, 20, 256)


@pytest.fixture
def sample_calibration():
    """Create sample calibration data."""
    return torch.randn(4, 5, 16, 256)


@pytest.fixture
def sample_features():
    """Create sample encoded features."""
    return torch.randn(4, 64, 256)


@pytest.fixture
def sample_user_embedding():
    """Create sample user embedding."""
    return torch.randn(4, 128)
