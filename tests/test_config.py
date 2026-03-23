"""
Tests for AMRPAConfig

This module tests the configuration class including:
- Default values
- Validation logic
- Type checking
- Edge cases
"""

import pytest
from amrpa.core.config import AMRPAConfig


class TestAMRPAConfigDefaults:
    """Test default configuration values."""
    
    def test_default_initialization(self):
        """Test that config initializes with correct defaults."""
        config = AMRPAConfig()
        
        # Memory parameters
        assert config.memory_decay == 0.9
        assert config.memory_noise == 0.001
        assert config.adaptive_window == True
        assert config.max_window_size == 4
        
        # Gating parameters
        assert config.gate_sensitivity == 2.0
        assert config.gate_bias == -0.25
        assert config.gate_activation == 'sigmoid'
        
        # Selection parameters
        assert config.alpha_temperature == 0.25
        assert config.alpha_mlp_hidden == 384
        assert config.alpha_mlp_layers == 2
        
        # Injection parameters
        assert config.target_layers == 'last_4'
        assert config.freeze_base_model == True
        assert config.freeze_embeddings == True
        
        # Training parameters
        assert config.dropout == 0.2
        assert config.diversity_weight == 0.005
        assert config.gate_reg_weight == 0.05
        
        # Tracking
        assert config.track_metrics == True


class TestAMRPAConfigValidation:
    """Test configuration validation."""
    
    def test_valid_memory_decay(self):
        """Test valid memory decay values."""
        config = AMRPAConfig(memory_decay=0.95)
        assert config.memory_decay == 0.95
    
    def test_invalid_memory_decay_too_low(self):
        """Test that memory decay < 0 raises error."""
        with pytest.raises(ValueError, match="memory_decay must be between 0 and 1"):
            AMRPAConfig(memory_decay=-0.1)
    
    def test_invalid_memory_decay_too_high(self):
        """Test that memory decay > 1 raises error."""
        with pytest.raises(ValueError, match="memory_decay must be between 0 and 1"):
            AMRPAConfig(memory_decay=1.5)
    
    def test_valid_memory_noise(self):
        """Test valid memory noise values."""
        config = AMRPAConfig(memory_noise=0.01)
        assert config.memory_noise == 0.01
    
    def test_invalid_memory_noise(self):
        """Test that negative memory noise raises error."""
        with pytest.raises(ValueError, match="memory_noise must be non-negative"):
            AMRPAConfig(memory_noise=-0.001)
    
    def test_valid_max_window_size(self):
        """Test valid window size."""
        config = AMRPAConfig(max_window_size=8)
        assert config.max_window_size == 8
    
    def test_invalid_max_window_size(self):
        """Test that window size < 1 raises error."""
        with pytest.raises(ValueError, match="max_window_size must be at least 1"):
            AMRPAConfig(max_window_size=0)
    
    def test_valid_gate_activation(self):
        """Test valid gate activations."""
        for activation in ['sigmoid', 'tanh', 'relu']:
            config = AMRPAConfig(gate_activation=activation)
            assert config.gate_activation == activation
    
    def test_invalid_gate_activation(self):
        """Test that invalid activation raises error."""
        with pytest.raises(ValueError, match="gate_activation must be one of"):
            AMRPAConfig(gate_activation='softmax')
    
    def test_valid_alpha_mlp_layers(self):
        """Test valid MLP layer count."""
        config = AMRPAConfig(alpha_mlp_layers=3)
        assert config.alpha_mlp_layers == 3
    
    def test_invalid_alpha_mlp_layers(self):
        """Test that MLP layers < 1 raises error."""
        with pytest.raises(ValueError, match="alpha_mlp_layers must be at least 1"):
            AMRPAConfig(alpha_mlp_layers=0)
    
    def test_valid_dropout(self):
        """Test valid dropout values."""
        config = AMRPAConfig(dropout=0.3)
        assert config.dropout == 0.3
    
    def test_invalid_dropout_negative(self):
        """Test that negative dropout raises error."""
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            AMRPAConfig(dropout=-0.1)
    
    def test_invalid_dropout_too_high(self):
        """Test that dropout > 1 raises error."""
        with pytest.raises(ValueError, match="dropout must be between 0 and 1"):
            AMRPAConfig(dropout=1.5)


class TestAMRPAConfigTargetLayers:
    """Test target layers specification."""
    
    def test_string_target_layers(self):
        """Test string target layer specifications."""
        for target in ['all', 'last_4', 'first_3', 'middle_6']:
            config = AMRPAConfig(target_layers=target)
            assert config.target_layers == target
    
    def test_list_target_layers(self):
        """Test list target layer specifications."""
        layers = [0, 1, 2, 3]
        config = AMRPAConfig(target_layers=layers)
        assert config.target_layers == layers
    
    def test_invalid_target_layers_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError, match="target_layers must be a string or list of integers"):
            AMRPAConfig(target_layers=123)
    
    def test_invalid_target_layers_list_content(self):
        """Test that list with non-integers raises error."""
        with pytest.raises(ValueError, match="target_layers list must contain only integers"):
            AMRPAConfig(target_layers=[1, 2, "three"])


class TestAMRPAConfigMethods:
    """Test configuration methods."""
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = AMRPAConfig(memory_decay=0.95, target_layers='last_6')
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['memory_decay'] == 0.95
        assert config_dict['target_layers'] == 'last_6'
        assert 'gate_sensitivity' in config_dict
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        config = AMRPAConfig(
            memory_decay=0.85,
            gate_sensitivity=3.0,
            alpha_temperature=0.5,
            target_layers=[8, 9, 10, 11]
        )
        
        assert config.memory_decay == 0.85
        assert config.gate_sensitivity == 3.0
        assert config.alpha_temperature == 0.5
        assert config.target_layers == [8, 9, 10, 11]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
