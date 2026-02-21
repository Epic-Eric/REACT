# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Characteristic CNN for extracting user-specific temporal patterns from EMG features.

This CNN processes encoded EMG features to extract characteristics
that are relevant for user identification and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class CharacteristicCNNConfig:
    """Configuration for Characteristic CNN.
    
    Attributes:
        in_channels: Number of input feature channels.
        hidden_channels: Hidden layer channels.
        out_channels: Output feature channels.
        kernel_size: Temporal convolution kernel size.
        num_layers: Number of convolutional layers.
        dropout: Dropout probability.
        use_residual: Whether to use residual connections.
        norm_type: Normalization type ('layer', 'batch', 'none').
    """
    in_channels: int = 64
    hidden_channels: int = 64
    out_channels: int = 64
    kernel_size: int = 3
    num_layers: int = 2
    dropout: float = 0.1
    use_residual: bool = True
    norm_type: str = "layer"


class CharacteristicCNNBlock(nn.Module):
    """Single convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.1,
        norm_type: str = "layer",
    ):
        super().__init__()
        
        # Padding to maintain sequence length
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=norm_type != "batch",
        )
        
        if norm_type == "layer":
            self.norm = nn.LayerNorm(out_channels)
        elif norm_type == "batch":
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = nn.Identity()
        
        self.norm_type = norm_type
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, L).
        
        Returns:
            Output tensor of shape (B, C_out, L).
        """
        x = self.conv(x)
        
        if self.norm_type == "layer":
            # LayerNorm expects (B, L, C)
            x = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        elif self.norm_type == "batch":
            x = self.norm(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CharacteristicCNN(nn.Module):
    """Multi-layer CNN for extracting user characteristics from EMG features.
    
    Processes temporal EMG features to extract user-specific patterns
    while maintaining temporal resolution.
    
    Args:
        config: CharacteristicCNNConfig object.
    """
    
    def __init__(self, config: CharacteristicCNNConfig):
        super().__init__()
        self.config = config
        
        layers = []
        
        # Input projection if needed
        current_channels = config.in_channels
        if config.in_channels != config.hidden_channels:
            layers.append(
                CharacteristicCNNBlock(
                    in_channels=config.in_channels,
                    out_channels=config.hidden_channels,
                    kernel_size=1,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                )
            )
            current_channels = config.hidden_channels
        
        # Main convolutional layers
        for i in range(config.num_layers):
            out_ch = config.out_channels if i == config.num_layers - 1 else config.hidden_channels
            layers.append(
                CharacteristicCNNBlock(
                    in_channels=current_channels,
                    out_channels=out_ch,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                    norm_type=config.norm_type,
                )
            )
            current_channels = out_ch
        
        self.layers = nn.ModuleList(layers)
        
        # Residual projection if dimensions differ
        self.use_residual = config.use_residual
        if config.use_residual and config.in_channels != config.out_channels:
            self.residual_proj = nn.Conv1d(
                config.in_channels, config.out_channels, kernel_size=1
            )
        else:
            self.residual_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract characteristic features from encoded EMG.
        
        Args:
            x: Input features of shape (B, C, L).
        
        Returns:
            Characteristic features of shape (B, C_out, L).
        """
        residual = x
        
        for layer in self.layers:
            x = layer(x)
        
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
        
        return x
