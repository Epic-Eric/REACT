# Copyright (c) 2026 REACT-EMG Authors
"""Model components for REACT-EMG."""

from .blocks import (
    FiLMLayer,
    CharacteristicCNN,
    AttentionScorer,
    TemporalAttentionPooling,
    GRUTemporalPooling,
    TransformerGroupEncoder,
)
from .user_encoder import UserEncoder, UserEncoderConfig
from .hybrid_model import (
    FiLMConditionedModel,
    FiLMConditionedModelConfig,
    load_pretrained_encoder,
)

__all__ = [
    "FiLMLayer",
    "CharacteristicCNN",
    "AttentionScorer",
    "TemporalAttentionPooling",
    "GRUTemporalPooling",
    "TransformerGroupEncoder",
    "UserEncoder",
    "UserEncoderConfig",
    "FiLMConditionedModel",
    "FiLMConditionedModelConfig",
    "load_pretrained_encoder",
]
