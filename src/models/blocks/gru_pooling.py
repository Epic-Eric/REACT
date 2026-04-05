# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
GRU-based Temporal Pooling.

Replaces attention-based pooling with a bidirectional GRU that
processes the temporal sequence and uses the final hidden state
as the fixed-size representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class GRUTemporalPoolingConfig:
    """Configuration for GRU Temporal Pooling.

    Attributes:
        in_channels: Number of input feature channels.
        hidden_size: GRU hidden size (output will be projected to in_channels).
        num_layers: Number of stacked GRU layers.
        bidirectional: Whether to use bidirectional GRU.
        dropout: Dropout between GRU layers (only used when num_layers > 1).
    """
    in_channels: int = 64
    hidden_size: int = 64
    num_layers: int = 3
    bidirectional: bool = True
    dropout: float = 0.1


class GRUTemporalPooling(nn.Module):
    """Pools temporal features using a (bidirectional) GRU.

    Processes the ``(B, C, L)`` feature sequence through a multi-layer GRU
    and returns a fixed-size ``(B, C)`` vector derived from the final hidden
    states.  When bidirectional, the forward and backward final hidden states
    are concatenated and linearly projected back to ``in_channels``.

    Args:
        config: GRUTemporalPoolingConfig object.
    """

    def __init__(self, config: GRUTemporalPoolingConfig):
        super().__init__()
        self.config = config

        self.gru = nn.GRU(
            input_size=config.in_channels,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=config.bidirectional,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )

        # Project concatenated bidirectional output back to in_channels
        if config.bidirectional:
            self.output_proj = nn.Linear(config.hidden_size * 2, config.in_channels)
        else:
            self.output_proj = (
                nn.Linear(config.hidden_size, config.in_channels)
                if config.hidden_size != config.in_channels
                else nn.Identity()
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool features using GRU final hidden state.

        Args:
            x: Input features of shape ``(B, C, L)``.
            mask: Optional boolean mask of shape ``(B, L)``, True for valid positions.

        Returns:
            Pooled features of shape ``(B, C)``.
        """
        # (B, C, L) -> (B, L, C) for GRU batch_first=True
        x_t = x.transpose(-1, -2)
        B, L, C = x_t.shape

        if mask is not None:
            # Compute actual lengths from mask
            lengths = mask.sum(dim=-1).clamp(min=1).cpu()  # (B,)
            packed = pack_padded_sequence(
                x_t, lengths, batch_first=True, enforce_sorted=False,
            )
            _, h_n = self.gru(packed)  # h_n: (num_layers*D, B, hidden)
        else:
            _, h_n = self.gru(x_t)  # h_n: (num_layers*D, B, hidden)

        # Extract final layer hidden state(s)
        if self.config.bidirectional:
            # h_n shape: (num_layers*2, B, hidden)
            # Last layer forward: h_n[-2], last layer backward: h_n[-1]
            h_fwd = h_n[-2]  # (B, hidden)
            h_bwd = h_n[-1]  # (B, hidden)
            h_out = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, hidden*2)
        else:
            h_out = h_n[-1]  # (B, hidden)

        # Project to in_channels: (B, C)
        return self.output_proj(h_out)
