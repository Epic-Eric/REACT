# Copyright (c) 2026 REACT-EMG Authors
"""Neural network building blocks for REACT-EMG."""

from .film_layer import FiLMLayer, FiLMConfig
from .characteristic_cnn import CharacteristicCNN, CharacteristicCNNConfig
from .attention_scorer import (
    AttentionScorer,
    AttentionScorerConfig,
    TemporalAttentionPooling,
    TemporalAttentionPoolingConfig,
)
from .transformer_block import TransformerGroupEncoder, TransformerGroupEncoderConfig

__all__ = [
    "FiLMLayer",
    "FiLMConfig",
    "CharacteristicCNN",
    "CharacteristicCNNConfig",
    "AttentionScorer",
    "AttentionScorerConfig",
    "TemporalAttentionPooling",
    "TemporalAttentionPoolingConfig",
    "TransformerGroupEncoder",
    "TransformerGroupEncoderConfig",
]
