# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
User Encoder Pipeline.

Combines the characteristic CNN, attention pooling, and transformer group encoder
to generate user embeddings from calibration recordings.

Pipeline:
1. Encode each calibration recording with pretrained encoder
2. Process with Characteristic CNN to extract user-specific features
3. Pool each recording via temporal attention to get fixed-size vectors
4. Aggregate all recordings with Transformer Group Encoder using learnable query
5. Output final user embedding for FiLM conditioning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn

from .blocks import (
    CharacteristicCNN,
    CharacteristicCNNConfig,
    TemporalAttentionPooling,
    TemporalAttentionPoolingConfig,
    TransformerGroupEncoder,
    TransformerGroupEncoderConfig,
)


@dataclass
class UserEncoderConfig:
    """Configuration for User Encoder.
    
    Attributes:
        feature_dim: Feature dimension from pretrained encoder.
        characteristic_cnn: Configuration for characteristic CNN.
        attention_pooling: Configuration for temporal attention pooling.
        group_encoder: Configuration for transformer group encoder.
        max_calibration_samples: Maximum number of calibration recordings.
        zero_embedding_dim: Dimension for zero-sample fallback embedding.
    """
    feature_dim: int = 64
    
    # Sub-module configs with defaults
    characteristic_cnn: CharacteristicCNNConfig = field(
        default_factory=lambda: CharacteristicCNNConfig(
            in_channels=64,
            hidden_channels=64,
            out_channels=64,
            kernel_size=3,
            num_layers=2,
            dropout=0.1,
        )
    )
    attention_pooling: TemporalAttentionPoolingConfig = field(
        default_factory=lambda: TemporalAttentionPoolingConfig(
            in_channels=64,
            scorer_hidden=32,
            scorer_layers=2,
            temperature=1.0,
            dropout=0.1,
        )
    )
    group_encoder: TransformerGroupEncoderConfig = field(
        default_factory=lambda: TransformerGroupEncoderConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=128,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            max_samples=32,
        )
    )
    
    max_calibration_samples: int = 30
    zero_embedding_dim: int = 128
    

class UserEncoder(nn.Module):
    """Complete user encoding pipeline.
    
    Takes calibration recordings (after encoding) and produces a single
    user embedding suitable for FiLM conditioning.
    
    Args:
        config: UserEncoderConfig object.
    """
    
    def __init__(self, config: UserEncoderConfig):
        super().__init__()
        self.config = config
        
        # Characteristic CNN for extracting user-specific patterns
        self.characteristic_cnn = CharacteristicCNN(config.characteristic_cnn)
        
        # Temporal attention pooling for each recording
        self.attention_pooling = TemporalAttentionPooling(config.attention_pooling)
        
        # Transformer for aggregating across recordings
        self.group_encoder = TransformerGroupEncoder(config.group_encoder)
        
        # Fallback embedding when no calibration samples
        self.zero_embedding = nn.Parameter(
            torch.randn(config.zero_embedding_dim) * 0.02
        )
    
    def forward(
        self,
        encoded_recordings: List[torch.Tensor],
        recording_masks: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Generate user embedding from encoded calibration recordings.
        
        Args:
            encoded_recordings: List of K tensors, each of shape (C, L_i)
                               where L_i can vary per recording.
            recording_masks: Optional list of K boolean masks, each (L_i,).
        
        Returns:
            User embedding of shape (output_dim,).
        """
        num_recordings = len(encoded_recordings)
        
        # Handle zero recordings case
        if num_recordings == 0:
            return self.zero_embedding
        
        # Process each recording individually
        pooled_features = []
        for i, recording in enumerate(encoded_recordings):
            # Add batch dimension: (C, L) -> (1, C, L)
            recording = recording.unsqueeze(0)
            
            # Extract characteristics: (1, C, L)
            char_features = self.characteristic_cnn(recording)
            
            # Pool to fixed size: (1, C)
            mask = recording_masks[i].unsqueeze(0) if recording_masks else None
            pooled = self.attention_pooling(char_features, mask)
            
            pooled_features.append(pooled.squeeze(0))  # (C,)
        
        # Stack into calibration sample block: (K, C)
        calibration_block = torch.stack(pooled_features, dim=0)
        
        # Add batch dimension and pass through group encoder: (1, K, C) -> (1, D)
        calibration_block = calibration_block.unsqueeze(0)
        user_embedding = self.group_encoder(calibration_block)
        
        return user_embedding.squeeze(0)  # (D,)
    
    def forward_batch(
        self,
        encoded_recordings_batch: List[List[torch.Tensor]],
        recording_masks_batch: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Batch forward for multiple users.
        
        Args:
            encoded_recordings_batch: List of B lists, each containing K_i recordings.
            recording_masks_batch: Optional list of B lists of masks.
        
        Returns:
            User embeddings of shape (B, output_dim).
        """
        batch_embeddings = []
        for i, recordings in enumerate(encoded_recordings_batch):
            masks = recording_masks_batch[i] if recording_masks_batch else None
            embedding = self.forward(recordings, masks)
            batch_embeddings.append(embedding)
        
        return torch.stack(batch_embeddings, dim=0)
    
    def forward_padded(
        self,
        encoded_recordings: torch.Tensor,
        num_samples: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batch-efficient forward with pre-padded inputs.
        
        For training efficiency, calibration recordings can be pre-padded
        to uniform shape.
        
        Args:
            encoded_recordings: Tensor of shape (B, K_max, C, L_max) where
                               K_max is max calibration samples across batch,
                               L_max is max sequence length.
            num_samples: Number of actual calibration samples per batch item (B,).
            lengths: Actual sequence lengths (B, K_max).
        
        Returns:
            User embeddings of shape (B, output_dim).
        """
        B, K_max, C, L_max = encoded_recordings.shape
        device = encoded_recordings.device
        
        # Process through characteristic CNN
        # Reshape: (B, K_max, C, L_max) -> (B*K_max, C, L_max)
        x_flat = encoded_recordings.view(B * K_max, C, L_max)
        char_features_flat = self.characteristic_cnn(x_flat)
        
        # Create length mask for attention pooling
        if lengths is not None:
            # (B, K_max) -> (B*K_max, L_max)
            lengths_flat = lengths.view(B * K_max)
            seq_mask = torch.arange(L_max, device=device).expand(B * K_max, -1)
            seq_mask = seq_mask < lengths_flat.unsqueeze(-1)
        else:
            seq_mask = None
        
        # Pool each recording: (B*K_max, C)
        pooled_flat = self.attention_pooling(char_features_flat, seq_mask)
        
        # Reshape back: (B, K_max, C)
        calibration_block = pooled_flat.view(B, K_max, C)
        
        # Create sample mask for group encoder
        sample_mask = torch.arange(K_max, device=device).expand(B, -1)
        sample_mask = sample_mask < num_samples.unsqueeze(-1)  # (B, K_max)
        
        # Generate user embeddings: (B, output_dim)
        user_embeddings = self.group_encoder(calibration_block, sample_mask)
        
        # Replace with zero embedding where no samples
        zero_mask = (num_samples == 0).unsqueeze(-1)
        user_embeddings = torch.where(
            zero_mask,
            self.zero_embedding.unsqueeze(0).expand(B, -1),
            user_embeddings,
        )
        
        return user_embeddings


class CalibrationSampleEncoder(nn.Module):
    """Encodes raw EMG recordings into features for the User Encoder.
    
    Uses a frozen pretrained encoder (e.g., vemg2pose TDS encoder).
    """
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        freeze: bool = True,
    ):
        super().__init__()
        self.encoder = pretrained_encoder
        
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()
    
    def forward(self, emg: torch.Tensor) -> torch.Tensor:
        """Encode raw EMG.
        
        Args:
            emg: Raw EMG signal of shape (B, 16, L).
        
        Returns:
            Encoded features of shape (B, C, L').
        """
        with torch.no_grad():
            return self.encoder(emg)


class UserEncoderPipeline(nn.Module):
    """Complete pipeline from raw calibration EMG to user embedding.
    
    Combines:
    1. Pretrained EMG encoder (frozen)
    2. User encoder (trainable)
    """
    
    def __init__(
        self,
        pretrained_encoder: nn.Module,
        user_encoder_config: UserEncoderConfig,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        self.calibration_encoder = CalibrationSampleEncoder(
            pretrained_encoder, freeze=freeze_encoder
        )
        self.user_encoder = UserEncoder(user_encoder_config)
    
    def forward(
        self,
        calibration_emg_batch: torch.Tensor,
        num_samples: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate user embeddings from raw calibration EMG.
        
        Args:
            calibration_emg_batch: Raw EMG of shape (B, K_max, 16, L_max).
            num_samples: Number of actual samples per batch item (B,).
            lengths: Actual sequence lengths (B, K_max).
        
        Returns:
            User embeddings of shape (B, output_dim).
        """
        B, K_max, C_in, L_in = calibration_emg_batch.shape
        
        # Encode all calibration recordings
        # Reshape: (B, K_max, 16, L) -> (B*K_max, 16, L)
        emg_flat = calibration_emg_batch.view(B * K_max, C_in, L_in)
        
        # Encode with frozen encoder
        encoded_flat = self.calibration_encoder(emg_flat)  # (B*K_max, C_out, L_out)
        
        _, C_out, L_out = encoded_flat.shape
        
        # Reshape back: (B, K_max, C_out, L_out)
        encoded = encoded_flat.view(B, K_max, C_out, L_out)
        
        # Adjust lengths for downsampled sequence
        if lengths is not None:
            # Estimate downsampling factor
            downsample_factor = L_in / L_out
            adjusted_lengths = (lengths / downsample_factor).long().clamp(min=1)
        else:
            adjusted_lengths = None
        
        # Generate user embeddings
        return self.user_encoder.forward_padded(encoded, num_samples, adjusted_lengths)
