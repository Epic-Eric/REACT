# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Feature-wise Linear Modulation (FiLM) Layer.

FiLM applies affine transformations conditioned on external inputs:
    output = gamma * features + beta

Reference:
    Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class FiLMConfig:
    """Configuration for FiLM layer.
    
    Attributes:
        feature_dim: Dimension of features to modulate.
        conditioning_dim: Dimension of conditioning input (user embedding).
        hidden_dim: Hidden layer dimension in the conditioning MLP.
        use_bias: Whether to include bias (beta) in modulation.
        init_gamma_one: Initialize gamma weights to produce ones.
        init_beta_zero: Initialize beta weights to produce zeros.
    """
    feature_dim: int = 64
    conditioning_dim: int = 128
    hidden_dim: int = 128
    use_bias: bool = True
    init_gamma_one: bool = True
    init_beta_zero: bool = True


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation layer.
    
    Generates gamma (scale) and beta (shift) parameters from a conditioning
    input and applies them to feature maps.
    
    Args:
        config: FiLMConfig object containing layer parameters.
    """
    
    def __init__(self, config: FiLMConfig):
        super().__init__()
        self.config = config
        
        # MLP to generate FiLM parameters from conditioning input
        self.conditioning_mlp = nn.Sequential(
            nn.Linear(config.conditioning_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.feature_dim * 2),  # gamma + beta
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize weights for stable training start."""
        # Initialize final layer to produce gamma=1, beta=0
        final_layer = self.conditioning_mlp[-1]
        nn.init.zeros_(final_layer.bias)
        
        if self.config.init_gamma_one:
            # Initialize gamma weights to produce ones
            nn.init.zeros_(final_layer.weight[:self.config.feature_dim])
            final_layer.bias.data[:self.config.feature_dim] = 1.0
        
        if self.config.init_beta_zero:
            # Initialize beta weights to produce zeros
            nn.init.zeros_(final_layer.weight[self.config.feature_dim:])
            final_layer.bias.data[self.config.feature_dim:] = 0.0
    
    def forward(
        self,
        features: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FiLM conditioning to features.
        
        Args:
            features: Input features of shape (B, C, L) where B is batch,
                     C is channels (feature_dim), L is sequence length.
            conditioning: Conditioning vector of shape (B, conditioning_dim).
        
        Returns:
            Modulated features of shape (B, C, L).
        """
        # Generate gamma and beta from conditioning
        film_params = self.conditioning_mlp(conditioning)  # (B, 2*feature_dim)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # Each: (B, feature_dim)
        
        # Reshape for broadcasting over sequence dimension
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)    # (B, C, 1)
        
        # Apply FiLM transformation
        if self.config.use_bias:
            return gamma * features + beta
        else:
            return gamma * features
    
    def get_film_params(
        self, conditioning: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get raw gamma and beta parameters.
        
        Args:
            conditioning: Conditioning vector of shape (B, conditioning_dim).
        
        Returns:
            Tuple of (gamma, beta), each of shape (B, feature_dim).
        """
        film_params = self.conditioning_mlp(conditioning)
        return torch.chunk(film_params, 2, dim=-1)


class FiLMGenerator(nn.Module):
    """Generates FiLM parameters without applying them.
    
    Useful when FiLM parameters need to be applied at multiple locations
    or with different features.
    """
    
    def __init__(self, config: FiLMConfig):
        super().__init__()
        self.config = config
        self.film_layer = FiLMLayer(config)
    
    def forward(self, conditioning: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate FiLM parameters.
        
        Args:
            conditioning: Conditioning vector of shape (B, conditioning_dim).
        
        Returns:
            Tuple of (gamma, beta), each of shape (B, feature_dim).
        """
        return self.film_layer.get_film_params(conditioning)


def apply_film(
    features: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Apply pre-computed FiLM parameters to features.
    
    Args:
        features: Input features of shape (B, C, L) or (B, C).
        gamma: Scale parameters of shape (B, C).
        beta: Shift parameters of shape (B, C).
    
    Returns:
        Modulated features of same shape as input.
    """
    if features.dim() == 3:
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
    return gamma * features + beta
