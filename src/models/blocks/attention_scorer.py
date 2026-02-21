# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Attention Scorer and Temporal Attention Pooling.

These modules compute attention weights over temporal features
and produce fixed-size representations via weighted pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionScorerConfig:
    """Configuration for Attention Scorer.
    
    Attributes:
        in_channels: Number of input feature channels.
        hidden_channels: Hidden layer channels for multi-layer scoring.
        num_layers: Number of 1x1 conv layers for scoring.
        temperature: Softmax temperature for attention sharpness.
        dropout: Dropout probability.
    """
    in_channels: int = 64
    hidden_channels: int = 32
    num_layers: int = 2
    temperature: float = 1.0
    dropout: float = 0.1


class AttentionScorer(nn.Module):
    """Computes attention scores over temporal dimension.
    
    Uses 1x1 convolutions to produce a scalar relevance score
    for each time step, which can be used for weighted pooling.
    
    Args:
        config: AttentionScorerConfig object.
    """
    
    def __init__(self, config: AttentionScorerConfig):
        super().__init__()
        self.config = config
        
        layers = []
        current_channels = config.in_channels
        
        for i in range(config.num_layers - 1):
            layers.extend([
                nn.Conv1d(current_channels, config.hidden_channels, kernel_size=1),
                nn.LayerNorm([config.hidden_channels]),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            current_channels = config.hidden_channels
        
        # Final layer outputs single score per timestep
        layers.append(nn.Conv1d(current_channels, 1, kernel_size=1))
        
        self.scorer = nn.Sequential(*layers)
        self.temperature = config.temperature
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute attention scores.
        
        Args:
            x: Input features of shape (B, C, L).
            mask: Optional boolean mask of shape (B, L), True for valid positions.
        
        Returns:
            Attention weights of shape (B, L), normalized via softmax.
        """
        # Compute raw scores: (B, 1, L) -> (B, L)
        scores = self.scorer(x).squeeze(1)
        
        # Apply temperature scaling
        scores = scores / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Normalize with softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        return attention_weights
    
    def forward_with_logits(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention scores and return both logits and weights.
        
        Useful for visualization and analysis.
        
        Returns:
            Tuple of (attention_weights, raw_logits).
        """
        scores = self.scorer(x).squeeze(1)
        scores = scores / self.temperature
        
        if mask is not None:
            masked_scores = scores.masked_fill(~mask, float('-inf'))
        else:
            masked_scores = scores
        
        attention_weights = F.softmax(masked_scores, dim=-1)
        
        return attention_weights, scores


@dataclass
class TemporalAttentionPoolingConfig:
    """Configuration for Temporal Attention Pooling.
    
    Attributes:
        in_channels: Number of input feature channels.
        scorer_hidden: Hidden channels for attention scorer.
        scorer_layers: Number of layers in attention scorer.
        temperature: Softmax temperature.
        dropout: Dropout probability.
    """
    in_channels: int = 64
    scorer_hidden: int = 32
    scorer_layers: int = 2
    temperature: float = 1.0
    dropout: float = 0.1


class TemporalAttentionPooling(nn.Module):
    """Pools temporal features using learned attention weights.
    
    Produces a fixed-size representation from variable-length sequences
    by computing attention-weighted sum over time.
    
    Args:
        config: TemporalAttentionPoolingConfig object.
    """
    
    def __init__(self, config: TemporalAttentionPoolingConfig):
        super().__init__()
        self.config = config
        
        scorer_config = AttentionScorerConfig(
            in_channels=config.in_channels,
            hidden_channels=config.scorer_hidden,
            num_layers=config.scorer_layers,
            temperature=config.temperature,
            dropout=config.dropout,
        )
        self.attention_scorer = AttentionScorer(scorer_config)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool features using attention.
        
        Args:
            x: Input features of shape (B, C, L).
            mask: Optional boolean mask of shape (B, L).
        
        Returns:
            Pooled features of shape (B, C).
        """
        # Get attention weights: (B, L)
        attention_weights = self.attention_scorer(x, mask)
        
        # Transpose for easier broadcasting: (B, C, L) -> (B, L, C)
        x_t = x.transpose(-1, -2)
        
        # Weighted sum: (B, L, C) * (B, L, 1) -> (B, L, C) -> sum -> (B, C)
        pooled = torch.einsum('blc,bl->bc', x_t, attention_weights)
        
        return pooled
    
    def forward_with_attention(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool features and return attention weights.
        
        Useful for visualization.
        
        Returns:
            Tuple of (pooled_features, attention_weights).
        """
        attention_weights = self.attention_scorer(x, mask)
        x_t = x.transpose(-1, -2)
        pooled = torch.einsum('blc,bl->bc', x_t, attention_weights)
        
        return pooled, attention_weights


class MultiHeadTemporalAttentionPooling(nn.Module):
    """Multi-head variant of temporal attention pooling.
    
    Uses multiple attention heads to capture different aspects
    of the temporal features.
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert in_channels % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.heads = nn.ModuleList([
            TemporalAttentionPooling(
                TemporalAttentionPoolingConfig(
                    in_channels=self.head_dim,
                    temperature=temperature,
                    dropout=dropout,
                )
            )
            for _ in range(num_heads)
        ])
        
        self.output_proj = nn.Linear(in_channels, in_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Multi-head attention pooling.
        
        Args:
            x: Input features of shape (B, C, L).
            mask: Optional boolean mask of shape (B, L).
        
        Returns:
            Pooled features of shape (B, C).
        """
        B, C, L = x.shape
        
        # Split into heads: (B, num_heads, head_dim, L)
        x_heads = x.view(B, self.num_heads, self.head_dim, L)
        
        # Pool each head
        head_outputs = []
        for i, head in enumerate(self.heads):
            head_out = head(x_heads[:, i], mask)  # (B, head_dim)
            head_outputs.append(head_out)
        
        # Concatenate heads: (B, C)
        concat = torch.cat(head_outputs, dim=-1)
        
        return self.output_proj(concat)
