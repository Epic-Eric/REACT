# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Transformer-based Group Encoder for user embedding generation.

This module implements a transformer encoder with a learnable query token
that attends to calibration samples to produce a single user embedding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


@dataclass
class TransformerGroupEncoderConfig:
    """Configuration for Transformer Group Encoder.
    
    Attributes:
        input_dim: Dimension of input calibration samples.
        hidden_dim: Transformer hidden dimension.
        output_dim: Output user embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        dropout: Dropout probability.
        max_samples: Maximum number of calibration samples.
        use_cls_token: Whether to use a learnable [CLS] token for aggregation.
        pre_norm: Whether to use pre-normalization (more stable training).
    """
    input_dim: int = 64
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    max_samples: int = 32
    use_cls_token: bool = True
    pre_norm: bool = True


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention module."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention forward pass.
        
        Args:
            x: Input tensor of shape (B, N, D).
            mask: Optional attention mask of shape (B, N) for key masking,
                  or (B, N, N) for full attention masking.
        
        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            # Handle both 2D key mask (B, N) and 3D attention mask (B, N, N)
            if mask.dim() == 2:
                # Key mask: (B, N) -> (B, 1, 1, N) for broadcasting
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                # Full attention mask: (B, N, N) -> (B, 1, N, N)
                mask = mask.unsqueeze(1)
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre/post normalization."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadSelfAttention(hidden_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pre_norm:
            x = x + self.attn(self.norm1(x), mask)
            x = x + self.mlp(self.norm2(x))
        else:
            x = self.norm1(x + self.attn(x, mask))
            x = self.norm2(x + self.mlp(x))
        return x


class TransformerGroupEncoder(nn.Module):
    """Transformer encoder for aggregating calibration samples.
    
    Uses a learnable query token (similar to [CLS] in BERT) that attends
    to the calibration samples to produce a single user embedding.
    
    Args:
        config: TransformerGroupEncoderConfig object.
    """
    
    def __init__(self, config: TransformerGroupEncoderConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim) * 0.02)
        
        # Optional positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, config.max_samples + 1, config.hidden_dim) * 0.02
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
                pre_norm=config.pre_norm,
            )
            for _ in range(config.num_layers)
        ])
        
        # Final normalization (for pre-norm architecture)
        self.final_norm = nn.LayerNorm(config.hidden_dim) if config.pre_norm else nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable but expressive training.

        Previous gain=0.02 caused near-zero outputs regardless of input,
        making user embeddings ≈ 0 and FiLM conditioning non-functional.
        Standard Xavier (gain=1.0) ensures meaningful forward signal.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Scale down output projection to avoid large initial embeddings
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.1)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        calibration_samples: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate user embedding from calibration samples.
        
        Args:
            calibration_samples: Tensor of shape (B, K, D) where K is the
                                number of calibration samples and D is input_dim.
            mask: Optional boolean mask of shape (B, K), True for valid samples.
        
        Returns:
            User embedding of shape (B, output_dim).
        """
        B, K, D = calibration_samples.shape
        
        # Project inputs
        x = self.input_proj(calibration_samples)  # (B, K, hidden_dim)
        
        # Prepend learnable query token
        query = self.query_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([query, x], dim=1)  # (B, K+1, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :K+1]
        
        # Create attention mask if needed (query can attend to all)
        if mask is not None:
            # Prepend True for query token
            query_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([query_mask, mask], dim=1)  # (B, K+1)
        else:
            full_mask = None
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, full_mask)
        
        # Apply final normalization
        x = self.final_norm(x)
        
        # Extract query token output
        user_embedding = x[:, 0]  # (B, hidden_dim)
        
        # Project to output dimension
        user_embedding = self.output_proj(user_embedding)  # (B, output_dim)
        
        return user_embedding
    
    def forward_with_attention(
        self,
        calibration_samples: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass that also returns attention weights for visualization."""
        B, K, D = calibration_samples.shape
        
        x = self.input_proj(calibration_samples)
        query = self.query_token.expand(B, -1, -1)
        x = torch.cat([query, x], dim=1)
        x = x + self.pos_encoding[:, :K+1]
        
        if mask is not None:
            query_mask = torch.ones(B, 1, device=mask.device, dtype=mask.dtype)
            full_mask = torch.cat([query_mask, mask], dim=1)
        else:
            full_mask = None
        
        attention_weights = []
        for layer in self.layers:
            # Store attention before applying layer
            x_normed = layer.norm1(x) if layer.pre_norm else x
            attn = layer.attn(x_normed, full_mask)
            attention_weights.append(attn.detach())
            x = layer(x, full_mask)
        
        x = self.final_norm(x)
        user_embedding = self.output_proj(x[:, 0])
        
        return user_embedding, attention_weights


class ZeroSampleHandler(nn.Module):
    """Handles the case when no calibration samples are available.
    
    Provides a learned default user embedding when k=0.
    """
    
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.default_embedding = nn.Parameter(torch.randn(embedding_dim) * 0.02)
    
    def forward(
        self,
        user_embedding: torch.Tensor,
        num_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Replace embeddings where num_samples == 0 with default.
        
        Args:
            user_embedding: User embeddings of shape (B, D).
            num_samples: Number of calibration samples per batch item (B,).
        
        Returns:
            User embeddings with defaults where no samples available.
        """
        mask = (num_samples == 0).unsqueeze(-1)  # (B, 1)
        return torch.where(mask, self.default_embedding, user_embedding)
