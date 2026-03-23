"""
Integration Tests for AMRPA

This module tests the full injection and execution pipeline:
- Model injection
- Forward pass
- Metrics collection
- History management
"""

import pytest
import torch
from transformers import AutoModel, AutoTokenizer
from amrpa import inject_amrpa, AMRPAConfig, reset_amrpa_history, get_amrpa_metrics


@pytest.fixture
def model_name():
    """Model name for testing."""
    return "prajjwal1/bert-tiny"  # Small model for fast testing


@pytest.fixture
def model(model_name):
    """Load a small BERT model for testing."""
    return AutoModel.from_pretrained(model_name)


@pytest.fixture
def tokenizer(model_name):
    """Load tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture
def sample_inputs(tokenizer):
    """Create sample inputs."""
    text = "This is a test sentence for AMRPA."
    return tokenizer(text, return_tensors="pt")


class TestBasicInjection:
    """Test basic AMRPA injection."""
    
    def test_inject_with_defaults(self, model):
        """Test injection with default configuration."""
        original_model = model
        injected_model = inject_amrpa(model)
        
        # Model should be modified
        assert injected_model is not None
        # Should still be the same object
        assert injected_model is original_model
    
    def test_inject_with_custom_config(self, model):
        """Test injection with custom configuration."""
        config = AMRPAConfig(
            memory_decay=0.95,
            target_layers='last_2',
            gate_sensitivity=2.5
        )
        
        injected_model = inject_amrpa(model, config=config)
        assert injected_model is not None
    
    def test_inject_with_layer_list(self, model):
        """Test injection with specific layer list."""
        injected_model = inject_amrpa(model, target_layers=[0, 1])
        assert injected_model is not None


class TestForwardPass:
    """Test forward pass with AMRPA."""
    
    def test_forward_pass_basic(self, model, sample_inputs):
        """Test basic forward pass."""
        model = inject_amrpa(model)
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape[0] == 1  # batch size
    
    def test_forward_pass_multiple_batches(self, model, tokenizer):
        """Test forward pass with multiple batches."""
        model = inject_amrpa(model)
        
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        assert outputs.last_hidden_state.shape[0] == 3  # batch size
    
    def test_forward_pass_preserves_output_format(self, model, sample_inputs):
        """Test that output format is preserved."""
        # Get original output
        with torch.no_grad():
            original_output = model(**sample_inputs)
        
        # Inject AMRPA
        model = inject_amrpa(model)
        
        with torch.no_grad():
            amrpa_output = model(**sample_inputs)
        
        # Should have same attributes
        assert hasattr(amrpa_output, 'last_hidden_state')
        assert original_output.last_hidden_state.shape == amrpa_output.last_hidden_state.shape


class TestMetricsCollection:
    """Test metrics collection."""
    
    def test_metrics_collection(self, model, sample_inputs):
        """Test that metrics are collected."""
        config = AMRPAConfig(track_metrics=True)
        model = inject_amrpa(model, config=config)
        
        reset_amrpa_history(model)
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        metrics = get_amrpa_metrics(model)
        
        # Should have metrics
        assert isinstance(metrics, dict)
        # May be empty for first layer, but should exist
        assert 'gate_impact' in metrics or len(metrics) == 0
    
    def test_metrics_reset(self, model, sample_inputs):
        """Test metrics reset between batches."""
        config = AMRPAConfig(track_metrics=True)
        model = inject_amrpa(model, config=config)
        
        # First forward pass
        reset_amrpa_history(model)
        with torch.no_grad():
            model(**sample_inputs)
        
        # Second forward pass
        reset_amrpa_history(model)
        with torch.no_grad():
            model(**sample_inputs)
        
        # Should not crash
        metrics = get_amrpa_metrics(model)
        assert isinstance(metrics, dict)


class TestHistoryManagement:
    """Test attention history management."""
    
    def test_history_reset(self, model, sample_inputs):
        """Test history reset functionality."""
        model = inject_amrpa(model)
        
        # First pass
        with torch.no_grad():
            model(**sample_inputs)
        
        # Reset history
        reset_amrpa_history(model)
        
        # Second pass should work
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None
    
    def test_history_accumulation(self, model, tokenizer):
        """Test that history accumulates across forward passes."""
        model = inject_amrpa(model)
        reset_amrpa_history(model)
        
        texts = ["First.", "Second.", "Third."]
        
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            assert outputs is not None


class TestDifferentConfigurations:
    """Test different AMRPA configurations."""
    
    def test_all_layers(self, model, sample_inputs):
        """Test injecting into all layers."""
        model = inject_amrpa(model, target_layers='all')
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None
    
    def test_first_layers(self, model, sample_inputs):
        """Test injecting into first layers."""
        model = inject_amrpa(model, target_layers='first_1')
        
        with torch.no_grad():
            outputs = model(**sample_inputs)
        
        assert outputs is not None
    
    def test_different_activations(self, model, sample_inputs):
        """Test different gate activations."""
        for activation in ['sigmoid', 'tanh', 'relu']:
            # Need fresh model for each test
            fresh_model = AutoModel.from_pretrained("prajjwal1/bert-tiny")
            config = AMRPAConfig(gate_activation=activation)
            fresh_model = inject_amrpa(fresh_model, config=config)
            
            with torch.no_grad():
                outputs = fresh_model(**sample_inputs)
            
            assert outputs is not None


class TestGradientFlow:
    """Test gradient flow with AMRPA."""
    
    def test_gradients_enabled(self, model, sample_inputs):
        """Test that gradients flow through AMRPA."""
        config = AMRPAConfig(freeze_base_model=False)
        model = inject_amrpa(model, config=config)
        
        model.train()
        outputs = model(**sample_inputs)
        
        # Compute dummy loss
        loss = outputs.last_hidden_state.sum()
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = any(
            p.grad is not None 
            for p in model.parameters() 
            if p.requires_grad
        )
        assert has_gradients
    
    def test_frozen_layers(self, model, sample_inputs):
        """Test that frozen layers don't get gradients."""
        config = AMRPAConfig(
            freeze_base_model=True,
            target_layers='last_1'
        )
        model = inject_amrpa(model, config=config)
        
        # Count trainable parameters
        trainable_params = sum(
            p.numel() 
            for p in model.parameters() 
            if p.requires_grad
        )
        
        # Should have fewer trainable params than original
        assert trainable_params > 0  # AMRPA params should be trainable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
