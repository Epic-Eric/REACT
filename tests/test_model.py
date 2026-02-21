# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Unit tests for hybrid model and user encoder.
"""

import pytest
import torch

from src.models.hybrid_model import (
    FiLMConditionedModelConfig,
    FiLMConditionedModel,
    PredictionHeadConfig,
    TemporalPredictionHead,
)
from src.models.user_encoder import (
    UserEncoderConfig,
    UserEncoder,
    CalibrationSampleEncoder,
)


class TestPredictionHead:
    """Tests for prediction head."""
    
    def test_config_defaults(self):
        """Test PredictionHeadConfig defaults."""
        config = PredictionHeadConfig(input_dim=64, output_dim=20)
        assert config.input_dim == 64
        assert config.output_dim == 20
        assert config.hidden_dim == 128
        assert config.num_layers == 3
    
    def test_temporal_head_forward(self):
        """Test TemporalPredictionHead forward."""
        config = PredictionHeadConfig(input_dim=64, output_dim=20, temporal=True)
        head = TemporalPredictionHead(config)
        
        features = torch.randn(4, 64, 100)  # (B, C, T)
        output = head(features)
        
        assert output.shape == (4, 20, 100)
    
    def test_mlp_head_forward(self):
        """Test MLP prediction head."""
        config = PredictionHeadConfig(input_dim=64, output_dim=20, temporal=False)
        head = TemporalPredictionHead(config)
        
        # MLP processes each timestep independently
        features = torch.randn(4, 64, 100)
        output = head(features)
        
        assert output.shape == (4, 20, 100)


class TestUserEncoder:
    """Tests for user encoder pipeline."""
    
    def test_config_defaults(self):
        """Test UserEncoderConfig defaults."""
        config = UserEncoderConfig(input_dim=64)
        assert config.input_dim == 64
        assert config.user_embedding_dim == 128
        assert config.max_calibration_samples == 30
    
    def test_user_encoder_forward(self):
        """Test UserEncoder forward pass."""
        config = UserEncoderConfig(input_dim=64, user_embedding_dim=128)
        encoder = UserEncoder(config)
        
        # k=5 calibration samples
        calibration_features = torch.randn(4, 5, 64, 100)  # (B, K, C, T)
        user_embedding = encoder(calibration_features)
        
        assert user_embedding.shape == (4, 128)
    
    def test_user_encoder_variable_k(self):
        """Test UserEncoder with different k values."""
        config = UserEncoderConfig(input_dim=64, user_embedding_dim=128)
        encoder = UserEncoder(config)
        
        for k in [1, 5, 10, 20, 30]:
            calibration_features = torch.randn(2, k, 64, 100)
            user_embedding = encoder(calibration_features)
            assert user_embedding.shape == (2, 128)
    
    def test_user_encoder_zero_samples(self):
        """Test UserEncoder with k=0."""
        config = UserEncoderConfig(input_dim=64, user_embedding_dim=128)
        encoder = UserEncoder(config)
        
        # Empty calibration
        calibration_features = torch.randn(2, 0, 64, 100)
        user_embedding = encoder(calibration_features)
        
        assert user_embedding.shape == (2, 128)
    
    def test_calibration_sample_encoder(self):
        """Test CalibrationSampleEncoder."""
        encoder = CalibrationSampleEncoder(
            input_dim=64,
            hidden_dim=64,
            output_dim=64,
            num_cnn_layers=2,
        )
        
        # Single sample
        features = torch.randn(4, 64, 100)
        embedding = encoder(features)
        
        assert embedding.shape == (4, 64)


class TestFiLMConditionedModel:
    """Tests for full FiLM-conditioned model."""
    
    @pytest.fixture
    def model_config(self):
        """Create default model config."""
        return FiLMConditionedModelConfig(
            feature_dim=64,
            user_embedding_dim=128,
            num_joints=20,
        )
    
    def test_config_defaults(self, model_config):
        """Test FiLMConditionedModelConfig defaults."""
        assert model_config.feature_dim == 64
        assert model_config.user_embedding_dim == 128
        assert model_config.num_joints == 20
    
    def test_model_forward(self, model_config):
        """Test FiLMConditionedModel forward pass."""
        model = FiLMConditionedModel(model_config)
        
        encoded_features = torch.randn(4, 64, 100)
        calibration_features = torch.randn(4, 5, 64, 100)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        assert "predictions" in output
        assert output["predictions"].shape == (4, 20, 100)
    
    def test_model_forward_with_user_embedding(self, model_config):
        """Test forward with precomputed user embedding."""
        model = FiLMConditionedModel(model_config)
        
        encoded_features = torch.randn(4, 64, 100)
        user_embedding = torch.randn(4, 128)
        
        output = model(
            encoded_features=encoded_features,
            user_embedding=user_embedding,
        )
        
        assert output["predictions"].shape == (4, 20, 100)
    
    def test_model_output_dict(self, model_config):
        """Test model returns all expected outputs."""
        model = FiLMConditionedModel(model_config)
        
        encoded_features = torch.randn(4, 64, 100)
        calibration_features = torch.randn(4, 5, 64, 100)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        expected_keys = ["predictions", "user_embedding", "film_gamma", "film_beta"]
        for key in expected_keys:
            assert key in output
    
    def test_model_gradient_flow(self, model_config):
        """Test gradients flow through model."""
        model = FiLMConditionedModel(model_config)
        
        encoded_features = torch.randn(4, 64, 100, requires_grad=True)
        calibration_features = torch.randn(4, 5, 64, 100, requires_grad=True)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        loss = output["predictions"].mean()
        loss.backward()
        
        # Check gradients exist
        assert encoded_features.grad is not None
        assert calibration_features.grad is not None
    
    def test_model_deterministic(self, model_config):
        """Test model gives same output for same input in eval mode."""
        model = FiLMConditionedModel(model_config)
        model.eval()
        
        encoded_features = torch.randn(2, 64, 100)
        calibration_features = torch.randn(2, 5, 64, 100)
        
        with torch.no_grad():
            out1 = model(
                encoded_features=encoded_features,
                calibration_features=calibration_features,
            )
            out2 = model(
                encoded_features=encoded_features,
                calibration_features=calibration_features,
            )
        
        torch.testing.assert_close(out1["predictions"], out2["predictions"])
    
    def test_model_batch_size_one(self, model_config):
        """Test model works with batch size 1."""
        model = FiLMConditionedModel(model_config)
        
        encoded_features = torch.randn(1, 64, 100)
        calibration_features = torch.randn(1, 3, 64, 100)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        assert output["predictions"].shape == (1, 20, 100)


class TestModelEdgeCases:
    """Edge case tests for model components."""
    
    def test_very_short_sequence(self):
        """Test model with very short sequences."""
        config = FiLMConditionedModelConfig(
            feature_dim=64,
            user_embedding_dim=128,
            num_joints=20,
        )
        model = FiLMConditionedModel(config)
        
        # Very short sequence
        encoded_features = torch.randn(2, 64, 10)
        calibration_features = torch.randn(2, 3, 64, 10)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        assert output["predictions"].shape == (2, 20, 10)
    
    def test_single_calibration_sample(self):
        """Test model with single calibration sample."""
        config = FiLMConditionedModelConfig(
            feature_dim=64,
            user_embedding_dim=128,
            num_joints=20,
        )
        model = FiLMConditionedModel(config)
        
        encoded_features = torch.randn(2, 64, 100)
        calibration_features = torch.randn(2, 1, 64, 100)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        assert output["predictions"].shape == (2, 20, 100)
    
    def test_max_calibration_samples(self):
        """Test model with maximum calibration samples."""
        config = FiLMConditionedModelConfig(
            feature_dim=64,
            user_embedding_dim=128,
            num_joints=20,
        )
        model = FiLMConditionedModel(config)
        
        encoded_features = torch.randn(2, 64, 100)
        calibration_features = torch.randn(2, 30, 64, 100)
        
        output = model(
            encoded_features=encoded_features,
            calibration_features=calibration_features,
        )
        
        assert output["predictions"].shape == (2, 20, 100)
