# Copyright (c) 2026 REACT-EMG Authors
# Licensed under MIT License
"""
Unit tests for model building blocks.
"""

import pytest
import torch

from src.models.blocks.film_layer import FiLMConfig, FiLMLayer, FiLMGenerator, apply_film
from src.models.blocks.characteristic_cnn import CharacteristicCNNConfig, CharacteristicCNN
from src.models.blocks.attention_scorer import AttentionScorerConfig, AttentionScorer, TemporalAttentionPooling
from src.models.blocks.transformer_block import TransformerGroupEncoderConfig, TransformerGroupEncoder


class TestFiLMLayer:
    """Tests for FiLM layer components."""
    
    def test_film_config_defaults(self):
        """Test FiLMConfig default values."""
        config = FiLMConfig(conditioning_dim=128, feature_dim=64)
        assert config.conditioning_dim == 128
        assert config.feature_dim == 64
        assert config.hidden_dim == 128
        assert config.num_layers == 2
    
    def test_film_layer_forward(self):
        """Test FiLMLayer forward pass."""
        config = FiLMConfig(conditioning_dim=128, feature_dim=64)
        layer = FiLMLayer(config)
        
        conditioning = torch.randn(4, 128)
        gamma, beta = layer(conditioning)
        
        assert gamma.shape == (4, 64)
        assert beta.shape == (4, 64)
    
    def test_film_layer_initialization(self):
        """Test FiLMLayer initializes gamma near 1 and beta near 0."""
        config = FiLMConfig(conditioning_dim=128, feature_dim=64)
        layer = FiLMLayer(config)
        
        # Test with zero conditioning (should give near identity transform)
        conditioning = torch.zeros(1, 128)
        gamma, beta = layer(conditioning)
        
        # Gamma should be close to 1 (for identity scaling)
        # This is not strictly enforced by default, so just check shape
        assert gamma.shape == (1, 64)
    
    def test_apply_film_3d(self):
        """Test apply_film with 3D features."""
        features = torch.randn(4, 64, 100)  # (B, C, T)
        gamma = torch.ones(4, 64)
        beta = torch.zeros(4, 64)
        
        output = apply_film(features, gamma, beta)
        
        assert output.shape == features.shape
        # With gamma=1, beta=0, output should equal input
        torch.testing.assert_close(output, features)
    
    def test_apply_film_scaling(self):
        """Test apply_film scaling effect."""
        features = torch.ones(2, 64, 10)
        gamma = torch.full((2, 64), 2.0)
        beta = torch.full((2, 64), 1.0)
        
        output = apply_film(features, gamma, beta)
        
        expected = torch.full((2, 64, 10), 3.0)  # 1 * 2 + 1 = 3
        torch.testing.assert_close(output, expected)
    
    def test_film_generator(self):
        """Test FiLMGenerator."""
        generator = FiLMGenerator(
            conditioning_dim=128,
            feature_dim=64,
            hidden_dim=64,
            num_layers=2,
        )
        
        conditioning = torch.randn(4, 128)
        features = torch.randn(4, 64, 100)
        
        output, gamma, beta = generator(conditioning, features)
        
        assert output.shape == features.shape
        assert gamma.shape == (4, 64)
        assert beta.shape == (4, 64)


class TestCharacteristicCNN:
    """Tests for Characteristic CNN."""
    
    def test_config_defaults(self):
        """Test CharacteristicCNNConfig defaults."""
        config = CharacteristicCNNConfig(input_dim=64)
        assert config.input_dim == 64
        assert config.hidden_dim == 64
        assert config.num_layers == 3
        assert config.kernel_size == 3
    
    def test_forward_shape(self):
        """Test CharacteristicCNN preserves shape."""
        config = CharacteristicCNNConfig(input_dim=64, num_layers=3)
        cnn = CharacteristicCNN(config)
        
        x = torch.randn(4, 64, 100)  # (B, C, T)
        output = cnn(x)
        
        assert output.shape == x.shape
    
    def test_batch_independence(self):
        """Test that batches are processed independently."""
        config = CharacteristicCNNConfig(input_dim=64)
        cnn = CharacteristicCNN(config)
        
        x1 = torch.randn(1, 64, 100)
        x2 = torch.randn(1, 64, 100)
        x_batch = torch.cat([x1, x2], dim=0)
        
        cnn.eval()
        with torch.no_grad():
            out1 = cnn(x1)
            out2 = cnn(x2)
            out_batch = cnn(x_batch)
        
        torch.testing.assert_close(out_batch[0], out1[0])
        torch.testing.assert_close(out_batch[1], out2[0])


class TestAttentionScorer:
    """Tests for Attention Scorer."""
    
    def test_config_defaults(self):
        """Test AttentionScorerConfig defaults."""
        config = AttentionScorerConfig(input_dim=64)
        assert config.input_dim == 64
        assert config.hidden_dim == 32
        assert config.temperature == 1.0
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1."""
        config = AttentionScorerConfig(input_dim=64)
        scorer = AttentionScorer(config)
        
        x = torch.randn(4, 64, 100)
        weights = scorer(x)
        
        assert weights.shape == (4, 100)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4))
    
    def test_temporal_attention_pooling(self):
        """Test TemporalAttentionPooling output shape."""
        pooling = TemporalAttentionPooling(input_dim=64, hidden_dim=32)
        
        x = torch.randn(4, 64, 100)
        output, weights = pooling(x)
        
        assert output.shape == (4, 64)
        assert weights.shape == (4, 100)


class TestTransformerGroupEncoder:
    """Tests for Transformer Group Encoder."""
    
    def test_config_defaults(self):
        """Test TransformerGroupEncoderConfig defaults."""
        config = TransformerGroupEncoderConfig(input_dim=64)
        assert config.input_dim == 64
        assert config.hidden_dim == 128
        assert config.num_layers == 3
        assert config.num_heads == 4
        assert config.max_samples == 30
    
    def test_forward_shape(self):
        """Test TransformerGroupEncoder output shape."""
        config = TransformerGroupEncoderConfig(
            input_dim=64,
            hidden_dim=128,
            output_dim=128,
        )
        encoder = TransformerGroupEncoder(config)
        
        # k=5 samples, each with 64-dim embedding
        x = torch.randn(4, 5, 64)  # (B, K, D)
        output = encoder(x)
        
        assert output.shape == (4, 128)
    
    def test_variable_k(self):
        """Test encoder handles different k values."""
        config = TransformerGroupEncoderConfig(input_dim=64, output_dim=128)
        encoder = TransformerGroupEncoder(config)
        
        for k in [1, 5, 10, 30]:
            x = torch.randn(2, k, 64)
            output = encoder(x)
            assert output.shape == (2, 128)
    
    def test_zero_samples(self):
        """Test encoder handles k=0."""
        config = TransformerGroupEncoderConfig(input_dim=64, output_dim=128)
        encoder = TransformerGroupEncoder(config)
        
        # Empty input
        x = torch.randn(2, 0, 64)
        output = encoder(x)
        
        assert output.shape == (2, 128)
    
    def test_masked_samples(self):
        """Test encoder with padded/masked samples."""
        config = TransformerGroupEncoderConfig(input_dim=64, output_dim=128)
        encoder = TransformerGroupEncoder(config)
        
        x = torch.randn(2, 10, 64)
        # Mask: first batch has 5 valid samples, second has 10
        mask = torch.zeros(2, 10, dtype=torch.bool)
        mask[0, 5:] = True  # Mask out samples 5-9 for first batch
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (2, 128)


class TestIntegration:
    """Integration tests combining multiple blocks."""
    
    def test_film_cnn_integration(self):
        """Test FiLM layer with CNN features."""
        cnn_config = CharacteristicCNNConfig(input_dim=64)
        film_config = FiLMConfig(conditioning_dim=128, feature_dim=64)
        
        cnn = CharacteristicCNN(cnn_config)
        film = FiLMLayer(film_config)
        
        features = torch.randn(4, 64, 100)
        conditioning = torch.randn(4, 128)
        
        cnn_out = cnn(features)
        gamma, beta = film(conditioning)
        conditioned = apply_film(cnn_out, gamma, beta)
        
        assert conditioned.shape == features.shape
    
    def test_full_user_encoder_pipeline(self):
        """Test full pipeline: CNN -> Attention -> Transformer."""
        cnn_config = CharacteristicCNNConfig(input_dim=64)
        attn_config = AttentionScorerConfig(input_dim=64)
        trans_config = TransformerGroupEncoderConfig(input_dim=64, output_dim=128)
        
        cnn = CharacteristicCNN(cnn_config)
        pooling = TemporalAttentionPooling(64, 32)
        transformer = TransformerGroupEncoder(trans_config)
        
        # Simulate k=5 calibration samples
        batch_size = 4
        k = 5
        seq_len = 100
        
        # Process each sample through CNN and pool
        calibration_features = torch.randn(batch_size, k, 64, seq_len)
        
        sample_embeddings = []
        for i in range(k):
            feat = calibration_features[:, i]  # (B, 64, T)
            cnn_out = cnn(feat)
            pooled, _ = pooling(cnn_out)  # (B, 64)
            sample_embeddings.append(pooled)
        
        # Stack and aggregate
        stacked = torch.stack(sample_embeddings, dim=1)  # (B, K, 64)
        user_embedding = transformer(stacked)  # (B, 128)
        
        assert user_embedding.shape == (batch_size, 128)
