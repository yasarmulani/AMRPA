"""
Base Architecture Adapter

This module defines the abstract base class for architecture-specific adapters.
Each adapter handles the specifics of different transformer architectures.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch.nn as nn


class BaseArchitectureAdapter(ABC):
    """Abstract base class for architecture-specific adapters.
    
    Adapters handle the architecture-specific details of injecting AMRPA
    into different transformer models (BERT, GPT-2, T5, etc.).
    """
    
    @abstractmethod
    def get_num_layers(self, model: nn.Module) -> int:
        """Get the total number of transformer layers in the model.
        
        Args:
            model: The transformer model
            
        Returns:
            Number of transformer layers
        """
        pass
    
    @abstractmethod
    def get_attention_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the attention module at a specific layer index.
        
        Args:
            model: The transformer model
            layer_idx: Index of the layer (0-indexed)
            
        Returns:
            The attention module at the specified layer
        """
        pass
    
    @abstractmethod
    def get_qkv_projections(self, attention_module: nn.Module) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Extract Q, K, V projection layers from an attention module.
        
        Args:
            attention_module: The attention module
            
        Returns:
            Tuple of (query_proj, key_proj, value_proj)
        """
        pass
    
    @abstractmethod
    def get_hidden_size(self, model: nn.Module) -> int:
        """Get the hidden dimension of the model.
        
        Args:
            model: The transformer model
            
        Returns:
            Hidden dimension size
        """
        pass
    
    @abstractmethod
    def get_num_attention_heads(self, model: nn.Module) -> int:
        """Get the number of attention heads.
        
        Args:
            model: The transformer model
            
        Returns:
            Number of attention heads
        """
        pass
    
    @abstractmethod
    def replace_attention(self, model: nn.Module, layer_idx: int, new_attention: nn.Module):
        """Replace the attention module at a specific layer with AMRPA wrapper.
        
        Args:
            model: The transformer model
            layer_idx: Index of the layer to replace
            new_attention: The new AMRPA attention wrapper
        """
        pass
    
    @abstractmethod
    def extract_hidden_states(self, *args, **kwargs):
        """Extract hidden states from forward pass arguments.
        
        Different architectures pass hidden states differently in their
        attention forward methods. This method standardizes the extraction.
        
        Args:
            *args, **kwargs: Forward pass arguments
            
        Returns:
            Hidden states tensor
        """
        pass
    
    @abstractmethod
    def format_output(self, attention_output, *args, **kwargs):
        """Format AMRPA output to match the expected output format.
        
        Different architectures expect different output formats from
        their attention modules. This method handles the formatting.
        
        Args:
            attention_output: Output from AMRPA attention
            *args, **kwargs: Original forward pass arguments
            
        Returns:
            Formatted output matching architecture expectations
        """
        pass
    
    def get_attention_mask(self, *args, **kwargs) -> Optional[nn.Module]:
        """Extract attention mask from forward pass arguments.
        
        Args:
            *args, **kwargs: Forward pass arguments
            
        Returns:
            Attention mask tensor or None
        """
        # Default implementation - can be overridden
        return kwargs.get('attention_mask', None)
    
    def freeze_layer(self, model: nn.Module, layer_idx: int):
        """Freeze all parameters in a specific layer.
        
        Args:
            model: The transformer model
            layer_idx: Index of the layer to freeze
        """
        layer = self._get_layer(model, layer_idx)
        for param in layer.parameters():
            param.requires_grad = False
    
    def freeze_embeddings(self, model: nn.Module):
        """Freeze embedding layers of the model.
        
        Args:
            model: The transformer model
        """
        # Default implementation - should be overridden for specific architectures
        if hasattr(model, 'embeddings'):
            for param in model.embeddings.parameters():
                param.requires_grad = False
    
    @abstractmethod
    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get the entire transformer layer at a specific index.
        
        Args:
            model: The transformer model
            layer_idx: Index of the layer
            
        Returns:
            The transformer layer module
        """
        pass
    
    def validate_model(self, model: nn.Module) -> bool:
        """Validate that the model is compatible with this adapter.
        
        Args:
            model: The transformer model
            
        Returns:
            True if compatible, raises ValueError otherwise
        """
        try:
            num_layers = self.get_num_layers(model)
            hidden_size = self.get_hidden_size(model)
            num_heads = self.get_num_attention_heads(model)
            
            if num_layers <= 0:
                raise ValueError(f"Invalid number of layers: {num_layers}")
            if hidden_size <= 0:
                raise ValueError(f"Invalid hidden size: {hidden_size}")
            if num_heads <= 0:
                raise ValueError(f"Invalid number of attention heads: {num_heads}")
            
            return True
        except Exception as e:
            raise ValueError(f"Model validation failed: {str(e)}")
