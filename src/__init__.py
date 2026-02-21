# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
REACT-EMG: Real-time EMG-based Adaptive Calibration Transformer

A FiLM-conditioned user-adaptive EMG-to-pose prediction system.
"""

__version__ = "0.1.0"
__author__ = "REACT-EMG Team"

from . import models
from . import data_loader
from . import engine
from . import utils

__all__ = ["models", "data_loader", "engine", "utils", "__version__"]
