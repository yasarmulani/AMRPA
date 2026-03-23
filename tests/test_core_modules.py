"""
Tests for Core AMRPA Modules

This module tests the core components:
- MemoryModule
- GatingModule
- SelectionModule
"""

import pytest
import torch
import torch.nn as nn
from amrpa.core.config import AMRPAConfig
from amrpa.core.memory import MemoryModule
from amrpa.core.gating import GatingModule
from amrpa.core.selection import SelectionModule


class TestMemoryModule:
    """Test MemoryModule functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AMRPAConfig(
            memory_decay=0.9,
            memory_noise=0.001,
            adaptive_window=True,
            max_window_size=4
        )
    
    @pytest.fixture
    def memory_module(self, config):
        """Create memory module."""
        return MemoryModule(config)
    
    def test_initialization(self, memory_module, config):
        """Test module initialization."""
        assert memory_module.config == config
    
    def test_adaptive_window_size_layer_1(self, memory_module):
        """Test adaptive window for layer 1."""
        window = memory_module.adaptive_window_size(1)
        assert window == 1
    
    def test_adaptive_window_size_layer_3(self, memory_module):
        """Test adaptive window for layer 3."""
        window = memory_module.adaptive_window_size(3)
        assert window == 2  # floor(log2(3)) + 1
    
    def test_adaptive_window_size_layer_10(self, memory_module):
        """Test adaptive window for layer 10."""
        window = memory_module.adaptive_window_size(10)
        assert window == 4  # max_window_size
    
    def test_apply_decay(self, memory_module):
        """Test decay application."""
        attention = torch.ones(2, 10, 10)
        decayed = memory_module.apply_decay(attention, k=1)
        
        assert decayed.shape == attention.shape
        # Should be less than original due to decay
        assert decayed.mean() < attention.mean()
    
    def test_apply_decay_increases_with_k(self, memory_module):
        """Test that decay increases with k."""
        attention = torch.ones(2, 10, 10)
        decayed_1 = memory_module.apply_decay(attention, k=1)
        decayed_2 = memory_module.apply_decay(attention, k=2)
        
        # Older patterns should be more decayed
        assert decayed_2.mean() < decayed_1.mean()
    
    def test_construct_memory_no_history(self, memory_module):
        """Test memory construction with no history."""
        memory = memory_module.construct_memory([], layer_idx=1)
        assert memory is None
    
    def test_construct_memory_layer_1(self, memory_module):
        """Test memory construction for layer 1."""
        history = [torch.randn(2, 10, 10)]
        memory = memory_module.construct_memory(history, layer_idx=1)
        assert memory is None  # Layer 1 has no memory
    
    def test_construct_memory_with_history(self, memory_module):
        """Test memory construction with history."""
        history = [
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10)
        ]
        memory = memory_module.construct_memory(history, layer_idx=3)
        
        assert memory is not None
        assert memory.shape == (2, 10, 10)


class TestGatingModule:
    """Test GatingModule functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AMRPAConfig(
            gate_sensitivity=2.0,
            gate_bias=-0.25,
            gate_activation='sigmoid'
        )
    
    @pytest.fixture
    def gating_module(self, config):
        """Create gating module."""
        return GatingModule(config, hidden_size=768)
    
    def test_initialization(self, gating_module, config):
        """Test module initialization."""
        assert gating_module.config == config
        assert gating_module.hidden_size == 768
        assert isinstance(gating_module.gamma_g, nn.Parameter)
        assert isinstance(gating_module.bias_g, nn.Parameter)
    
    def test_activation_sigmoid(self):
        """Test sigmoid activation."""
        config = AMRPAConfig(gate_activation='sigmoid')
        module = GatingModule(config, hidden_size=768)
        assert isinstance(module.activation, nn.Sigmoid)
    
    def test_activation_tanh(self):
        """Test tanh activation."""
        config = AMRPAConfig(gate_activation='tanh')
        module = GatingModule(config, hidden_size=768)
        assert isinstance(module.activation, nn.Tanh)
    
    def test_activation_relu(self):
        """Test relu activation."""
        config = AMRPAConfig(gate_activation='relu')
        module = GatingModule(config, hidden_size=768)
        assert isinstance(module.activation, nn.ReLU)
    
    def test_compute_similarity(self, gating_module):
        """Test similarity computation."""
        query = torch.randn(2, 10, 768)
        memory = torch.randn(2, 10, 10)
        values = torch.randn(2, 10, 768)
        
        similarity = gating_module.compute_similarity(query, memory, values)
        
        assert similarity.shape == (2, 10)
    
    def test_forward(self, gating_module):
        """Test forward pass."""
        query = torch.randn(2, 10, 768)
        memory = torch.randn(2, 10, 10)
        values = torch.randn(2, 10, 768)
        
        gates = gating_module(query, memory, values)
        
        assert gates.shape == (2, 10)
        # Gates should be in [0, 1] for sigmoid
        assert gates.min() >= 0
        assert gates.max() <= 1


class TestSelectionModule:
    """Test SelectionModule functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AMRPAConfig(
            alpha_temperature=0.25,
            alpha_mlp_hidden=384,
            alpha_mlp_layers=2,
            dropout=0.1
        )
    
    @pytest.fixture
    def selection_module(self, config):
        """Create selection module."""
        return SelectionModule(config, hidden_size=768)
    
    def test_initialization(self, selection_module, config):
        """Test module initialization."""
        assert selection_module.config == config
        assert selection_module.hidden_size == 768
        assert isinstance(selection_module.mlp_alpha, nn.Sequential)
    
    def test_compute_alpha_scores(self, selection_module):
        """Test alpha score computation."""
        query = torch.randn(2, 10, 768)
        memory_patterns = [
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10)
        ]
        values = torch.randn(2, 10, 768)
        
        scores = selection_module.compute_alpha_scores(query, memory_patterns, values)
        
        assert scores.shape == (2, 10, 3)  # 3 patterns
    
    def test_forward(self, selection_module):
        """Test forward pass."""
        query = torch.randn(2, 10, 768)
        memory_patterns = [
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10),
            torch.randn(2, 10, 10)
        ]
        values = torch.randn(2, 10, 768)
        
        alpha_weights = selection_module(query, memory_patterns, values)
        
        assert alpha_weights.shape == (2, 10, 3)
        # Weights should sum to 1 (softmax)
        assert torch.allclose(alpha_weights.sum(dim=-1), torch.ones(2, 10), atol=1e-5)
    
    def test_forward_empty_patterns(self, selection_module):
        """Test forward with empty patterns."""
        query = torch.randn(2, 10, 768)
        values = torch.randn(2, 10, 768)
        
        alpha_weights = selection_module(query, [], values)
        assert alpha_weights is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
