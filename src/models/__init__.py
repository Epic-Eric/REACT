# Copyright (c) 2026 REACT-EMG Authors
"""Model components for REACT-EMG."""

from .blocks import (
    FiLMLayer,
    CharacteristicCNN,
    AttentionScorer,
    TemporalAttentionPooling,
    TransformerGroupEncoder,
)
from .user_encoder import UserEncoder, UserEncoderConfig
from .hybrid_model import FiLMConditionedModel, FiLMConditionedModelConfig

__all__ = [
    "FiLMLayer",
    "CharacteristicCNN",
    "AttentionScorer",
    "TemporalAttentionPooling",
    "TransformerGroupEncoder",
    "UserEncoder",
    "UserEncoderConfig",
    "FiLMConditionedModel",
    "FiLMConditionedModelConfig",
]
