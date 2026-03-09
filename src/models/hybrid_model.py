# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
FiLM-Conditioned Hybrid Model.

Main model architecture that combines:
1. Frozen pretrained EMG encoder (from vemg2pose)
2. User encoder for generating user-specific embeddings
3. FiLM conditioning layer
4. Prediction head for pose estimation

This enables user-adaptive pose prediction through learned calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import FiLMLayer, FiLMConfig
from .user_encoder import UserEncoder, UserEncoderConfig


@dataclass
class PredictionHeadConfig:
    """Configuration for the pose prediction head.
    
    Attributes:
        in_channels: Input feature channels (after FiLM).
        hidden_dims: List of hidden layer dimensions.
        out_channels: Output pose dimensions (20 DOF for joint angles).
        dropout: Dropout probability.
        use_layer_norm: Whether to use layer normalization.
        output_scale: Scale factor for output (useful for velocity prediction).
    """
    in_channels: int = 64
    hidden_dims: list = field(default_factory=lambda: [256, 128])
    out_channels: int = 20
    dropout: float = 0.1
    use_layer_norm: bool = True
    output_scale: float = 0.01


@dataclass
class FiLMConditionedModelConfig:
    """Configuration for the complete FiLM-conditioned model.
    
    Attributes:
        feature_dim: Dimension of encoded EMG features.
        user_embedding_dim: Dimension of user embedding from encoder.
        film: FiLM layer configuration.
        user_encoder: User encoder configuration.
        prediction_head: Prediction head configuration.
        freeze_encoder: Whether to freeze the pretrained encoder.
        pretrained_checkpoint: Path to pretrained model checkpoint.
    """
    feature_dim: int = 64
    user_embedding_dim: int = 128
    
    film: FiLMConfig = field(default_factory=lambda: FiLMConfig(
        feature_dim=64,
        conditioning_dim=128,
        hidden_dim=128,
    ))
    
    user_encoder: UserEncoderConfig = field(default_factory=UserEncoderConfig)
    
    prediction_head: PredictionHeadConfig = field(
        default_factory=PredictionHeadConfig
    )
    
    freeze_encoder: bool = True
    pretrained_checkpoint: Optional[str] = None


class TemporalPredictionHead(nn.Module):
    """MLP head for temporal pose prediction.
    
    Operates on each time step independently.
    """
    
    def __init__(self, config: PredictionHeadConfig):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.in_channels
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.out_channels))
        
        self.mlp = nn.Sequential(*layers)
        self.scale = config.output_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict poses from features.
        
        Args:
            x: Features of shape (B, C, L).
        
        Returns:
            Pose predictions of shape (B, out_channels, L).
        """
        # Transpose for MLP: (B, C, L) -> (B, L, C)
        x = x.transpose(-1, -2)
        
        # Apply MLP: (B, L, C) -> (B, L, out)
        x = self.mlp(x)
        
        # Transpose back: (B, L, out) -> (B, out, L)
        x = x.transpose(-1, -2)
        
        return x * self.scale


class StatefulPredictionHead(nn.Module):
    """LSTM-based prediction head with state conditioning.
    
    Similar to vemg2pose, but operates on FiLM-conditioned features.
    """
    
    def __init__(
        self,
        in_channels: int = 64,
        state_channels: int = 20,
        hidden_size: int = 512,
        num_layers: int = 2,
        out_channels: int = 40,  # position + velocity
        output_scale: float = 0.01,
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=in_channels + state_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.output_proj = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_channels),
        )
        
        self.scale = output_scale
        self.hidden = None
    
    def reset_state(self):
        """Reset LSTM hidden state."""
        self.hidden = None
    
    def forward_step(
        self,
        features: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Single step forward.
        
        Args:
            features: Features at current timestep (B, C).
            state: Current state/pose (B, state_dim).
        
        Returns:
            Output prediction (B, out_channels).
        """
        x = torch.cat([features, state], dim=-1)
        x = x.unsqueeze(1)  # (B, 1, input_dim)
        
        out, self.hidden = self.lstm(x, self.hidden)
        out = out.squeeze(1)  # (B, hidden)
        
        return self.output_proj(out) * self.scale


class FiLMConditionedModel(nn.Module):
    """Main FiLM-conditioned EMG-to-pose model.
    
    Architecture:
    1. Encode main recording with frozen pretrained encoder
    2. Encode calibration recordings and generate user embedding
    3. Apply FiLM conditioning to main recording features
    4. Predict poses with conditioned features
    """
    
    def __init__(
        self,
        config: FiLMConditionedModelConfig,
        pretrained_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        
        # Pretrained encoder (will be set later if not provided)
        self.encoder = pretrained_encoder
        if pretrained_encoder is not None and config.freeze_encoder:
            self._freeze_encoder()
        
        # User encoder for generating user-specific embeddings
        self.user_encoder = UserEncoder(config.user_encoder)
        
        # FiLM conditioning layer
        self.film_layer = FiLMLayer(config.film)
        
        # Prediction head
        self.prediction_head = TemporalPredictionHead(config.prediction_head)
        
        # Track context requirements from encoder
        self.left_context = 0
        self.right_context = 0
        if pretrained_encoder is not None:
            if hasattr(pretrained_encoder, 'left_context'):
                self.left_context = pretrained_encoder.left_context
            if hasattr(pretrained_encoder, 'right_context'):
                self.right_context = pretrained_encoder.right_context
    
    def _freeze_encoder(self):
        """Freeze pretrained encoder parameters."""
        if self.encoder is None:
            return
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
    
    def set_encoder(self, encoder: nn.Module, freeze: bool = True):
        """Set pretrained encoder after initialization."""
        self.encoder = encoder
        if freeze:
            self._freeze_encoder()
        
        # Update context from encoder
        if hasattr(encoder, 'left_context'):
            self.left_context = encoder.left_context
        if hasattr(encoder, 'right_context'):
            self.right_context = encoder.right_context
    
    def encode(self, emg: torch.Tensor) -> torch.Tensor:
        """Encode raw EMG with pretrained encoder.
        
        Args:
            emg: Raw EMG of shape (B, 16, L).
        
        Returns:
            Encoded features of shape (B, C, L').
        """
        if self.encoder is None:
            raise RuntimeError("Encoder not set. Call set_encoder() first.")
        
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            return self.encoder(emg)
    
    def forward(
        self,
        emg: torch.Tensor,
        calibration_features: torch.Tensor,
        num_calibration_samples: torch.Tensor,
        calibration_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with pre-encoded calibration recordings.
        
        Args:
            emg: Raw EMG of shape (B, 16, L).
            calibration_features: Pre-encoded calibration recordings 
                                 (B, K_max, C, L_cal).
            num_calibration_samples: Number of actual samples per batch item (B,).
            calibration_lengths: Actual lengths of calibration recordings (B, K_max).
        
        Returns:
            Pose predictions of shape (B, out_channels, L').
        """
        # Encode main recording
        features = self.encode(emg)  # (B, C, L')
        
        # Generate user embeddings from calibration data
        user_embeddings = self.user_encoder.forward_padded(
            calibration_features,
            num_calibration_samples,
            calibration_lengths,
        )  # (B, user_embedding_dim)
        
        # Apply FiLM conditioning
        conditioned_features = self.film_layer(features, user_embeddings)
        
        # Predict poses
        predictions = self.prediction_head(conditioned_features)
        
        return predictions
    
    def forward_with_raw_calibration(
        self,
        emg: torch.Tensor,
        calibration_emg: torch.Tensor,
        num_calibration_samples: torch.Tensor,
        calibration_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass encoding calibration recordings on-the-fly.
        
        Args:
            emg: Raw EMG of shape (B, 16, L).
            calibration_emg: Raw calibration EMG (B, K_max, 16, L_cal).
            num_calibration_samples: Number of actual samples per batch item (B,).
            calibration_lengths: Actual lengths (B, K_max).
        
        Returns:
            Pose predictions of shape (B, out_channels, L').
        """
        B, K_max, C_in, L_cal = calibration_emg.shape
        
        # Encode main recording
        features = self.encode(emg)  # (B, C, L')
        
        # Encode calibration recordings
        cal_flat = calibration_emg.view(B * K_max, C_in, L_cal)
        cal_features_flat = self.encode(cal_flat)
        _, C, L_cal_enc = cal_features_flat.shape
        cal_features = cal_features_flat.view(B, K_max, C, L_cal_enc)
        
        # Adjust lengths for encoding downsampling
        if calibration_lengths is not None:
            ds_factor = L_cal / L_cal_enc
            cal_lengths_adj = (calibration_lengths / ds_factor).long().clamp(min=1)
        else:
            cal_lengths_adj = None
        
        # Generate user embeddings
        user_embeddings = self.user_encoder.forward_padded(
            cal_features, num_calibration_samples, cal_lengths_adj
        )
        
        # Apply FiLM conditioning
        conditioned_features = self.film_layer(features, user_embeddings)
        
        # Predict poses
        return self.prediction_head(conditioned_features)
    
    def forward_with_full_recordings(
        self,
        emg: torch.Tensor,
        calibration_recordings: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Forward pass with variable-length FULL calibration recordings.
        
        This is the proper way to use calibration data - each recording
        is the full EMG sequence (not windowed).
        
        Args:
            emg: Raw EMG of shape (B, 16, L) - the main recording windows.
            calibration_recordings: List of B items, each is a list of K_i tensors
                                   of shape (16, L_i) with variable L_i.
        
        Returns:
            Pose predictions of shape (B, out_channels, L').
        """
        B = emg.shape[0]
        device = emg.device
        
        # Encode main recording
        features = self.encode(emg)  # (B, C, L')
        
        # Process each batch item's calibration recordings separately
        user_embeddings_list = []
        
        for batch_idx in range(B):
            recordings = calibration_recordings[batch_idx]
            
            if len(recordings) == 0:
                # No calibration data - use zero embedding
                user_embed = torch.zeros(
                    self.config.user_embedding_dim, device=device
                )
            else:
                # Encode each full recording and pool
                recording_embeddings = []
                
                for rec in recordings:
                    try:
                        # rec shape: (16, L_i) - variable length
                        rec_batch = rec.unsqueeze(0).to(device)  # (1, 16, L_i)
                        
                        # Encode with pretrained encoder
                        rec_features = self.encode(rec_batch)  # (1, C, L_i')
                        
                        # Process through characteristic CNN
                        char_features = self.user_encoder.characteristic_cnn(
                            rec_features
                        )  # (1, C, L_i')
                        
                        # Pool via temporal attention to get fixed-size vector
                        pooled = self.user_encoder.attention_pooling(
                            char_features
                        )  # (1, C)
                        
                        recording_embeddings.append(pooled.squeeze(0))  # (C,)
                    except Exception:
                        # Skip recordings that fail to encode
                        continue
                
                # Handle case where all recordings failed to encode
                if len(recording_embeddings) == 0:
                    user_embed = torch.zeros(
                        self.config.user_embedding_dim, device=device
                    )
                else:
                    # Stack K recording embeddings
                    stacked = torch.stack(recording_embeddings, dim=0)  # (K, C)
                    stacked = stacked.unsqueeze(0)  # (1, K, C)
                    
                    # Aggregate with transformer group encoder (no mask needed)
                    user_embed = self.user_encoder.group_encoder(
                        stacked
                    ).squeeze(0)  # (user_embedding_dim,)
            
            user_embeddings_list.append(user_embed)
        
        # Stack all user embeddings
        user_embeddings = torch.stack(user_embeddings_list, dim=0)  # (B, embed_dim)
        
        # Apply FiLM conditioning
        conditioned_features = self.film_layer(features, user_embeddings)
        
        # Predict poses
        return self.prediction_head(conditioned_features)
    
    def get_user_embedding(
        self,
        calibration_features: torch.Tensor,
        num_calibration_samples: torch.Tensor,
        calibration_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get user embedding without pose prediction.
        
        Useful for analysis and visualization.
        """
        return self.user_encoder.forward_padded(
            calibration_features,
            num_calibration_samples,
            calibration_lengths,
        )
    
    def get_film_params(
        self,
        user_embedding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get FiLM parameters (gamma, beta) from user embedding.
        
        Useful for analysis and visualization.
        """
        return self.film_layer.get_film_params(user_embedding)


class FiLMConditionedModelForTraining(nn.Module):
    """Wrapper with training-specific functionality.
    
    Includes loss computation and metric tracking.
    """
    
    def __init__(
        self,
        config: FiLMConditionedModelConfig,
        pretrained_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = FiLMConditionedModel(config, pretrained_encoder)
        self.config = config
    
    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.
        
        Args:
            batch: Dictionary containing:
                - emg: (B, 16, L)
                - joint_angles: (B, 20, L)
                - calibration_features: (B, K_max, C, L_cal)
                - num_calibration_samples: (B,)
                - calibration_lengths: (B, K_max) [optional]
                - no_ik_failure: (B, L) [optional]
        
        Returns:
            Tuple of (predictions, targets).
        """
        predictions = self.model(
            emg=batch['emg'],
            calibration_features=batch['calibration_features'],
            num_calibration_samples=batch['num_calibration_samples'],
            calibration_lengths=batch.get('calibration_lengths'),
        )
        
        # Get targets aligned with predictions
        targets = batch['joint_angles']
        
        # Handle context trimming
        left_ctx = self.model.left_context
        right_ctx = self.model.right_context
        if left_ctx > 0 or right_ctx > 0:
            start = left_ctx
            end = -right_ctx if right_ctx > 0 else None
            targets = targets[..., start:end]
        
        return predictions, targets
    
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training losses.
        
        Returns:
            Dictionary of loss components.
        """
        # Ensure predictions and targets have same length
        min_len = min(predictions.shape[-1], targets.shape[-1])
        predictions = predictions[..., :min_len]
        targets = targets[..., :min_len]
        
        # Mean absolute error
        if mask is not None:
            mask = mask[..., :min_len]
            diff = (predictions - targets).abs()
            diff = diff * mask.unsqueeze(1)
            mae = diff.sum() / (mask.sum() * predictions.shape[1])
        else:
            mae = F.l1_loss(predictions, targets)
        
        # Mean squared error
        if mask is not None:
            diff_sq = (predictions - targets) ** 2
            diff_sq = diff_sq * mask.unsqueeze(1)
            mse = diff_sq.sum() / (mask.sum() * predictions.shape[1])
        else:
            mse = F.mse_loss(predictions, targets)
        
        return {
            'mae': mae,
            'mse': mse,
            'total_loss': mae,  # Use MAE as primary loss
        }


def load_pretrained_encoder(
    checkpoint_path: str,
    encoder_key: str = 'model.network',
    device: str = 'cpu',
) -> nn.Module:
    """Load pretrained encoder from emg2pose checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file.
        encoder_key: Key prefix for encoder weights in state dict.
        device: Device to load to.
    
    Returns:
        Loaded encoder module.
    """
    # Import emg2pose networks
    # Add emg2pose submodule to path if not already installed
    import sys
    from pathlib import Path
    emg2pose_path = Path(__file__).resolve().parent.parent.parent / "emg2pose"
    if emg2pose_path.exists() and str(emg2pose_path) not in sys.path:
        sys.path.insert(0, str(emg2pose_path))
    
    try:
        from emg2pose.networks import TdsNetwork, TdsStage, Conv1dBlock
    except ImportError:
        raise ImportError(
            "emg2pose not found. Install with: pip install -e emg2pose/"
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Extract encoder weights
    encoder_state = {}
    prefix = encoder_key + '.'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            encoder_state[new_key] = value
    
    # Recreate encoder architecture (matching vemg2pose defaults)
    conv_blocks = [
        Conv1dBlock(16, 256, kernel_size=11, stride=5),
        Conv1dBlock(256, 256, kernel_size=5, stride=2),
    ]
    
    tds_stages = [
        TdsStage(
            in_channels=256,
            in_conv_kernel_width=17,
            in_conv_stride=4,
            num_blocks=2,
            channels=16,
            feature_width=16,
            kernel_width=9,
        ),
        TdsStage(
            in_channels=256,
            in_conv_kernel_width=9,
            in_conv_stride=2,
            num_blocks=2,
            channels=16,
            feature_width=16,
            kernel_width=5,
            out_channels=64,
        ),
    ]
    
    encoder = TdsNetwork(conv_blocks=conv_blocks, tds_stages=tds_stages)
    encoder.load_state_dict(encoder_state)
    
    return encoder
