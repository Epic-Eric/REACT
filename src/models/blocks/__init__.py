# Copyright (c) 2026 REACT-EMG Authors
"""Neural network building blocks for REACT-EMG."""

from .film_layer import FiLMLayer
from .characteristic_cnn import CharacteristicCNN
from .attention_scorer import AttentionScorer, TemporalAttentionPooling
from .transformer_block import TransformerGroupEncoder

__all__ = [
    "FiLMLayer",
    "CharacteristicCNN",
    "AttentionScorer",
    "TemporalAttentionPooling",
    "TransformerGroupEncoder",
]
