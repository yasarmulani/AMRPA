"""
Encoder-Only Architecture Adapter

This module provides adapters for encoder-only transformer models
like BERT, RoBERTa, DistilBERT, and ALBERT.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import BaseArchitectureAdapter


class EncoderOnlyAdapter(BaseArchitectureAdapter):
    """Adapter for encoder-only transformer models (BERT, RoBERTa, etc.)."""
    
    def __init__(self, model_type: str = "auto"):
        """Initialize encoder-only adapter.
        
        Args:
            model_type: Type of model ('bert', 'roberta', 'distilbert', 'albert', 'auto')
        """
        self.model_type = model_type
    
    def get_num_layers(self, model: nn.Module) -> int:
        """Get number of encoder layers."""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return len(model.encoder.layer)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            # DistilBERT
            return len(model.transformer.layer)
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'albert_layer_groups'):
            # ALBERT
            return model.config.num_hidden_layers
        else:
            raise ValueError("Could not find encoder layers in model")
    
    def get_attention_module(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get attention module at specific layer."""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT, RoBERTa
            return model.encoder.layer[layer_idx].attention.self
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            # DistilBERT
            return model.transformer.layer[layer_idx].attention
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'albert_layer_groups'):
            # ALBERT (more complex due to layer sharing)
            group_idx = layer_idx // model.config.inner_group_num
            inner_idx = layer_idx % model.config.inner_group_num
            return model.encoder.albert_layer_groups[group_idx].albert_layers[inner_idx].attention
        else:
            raise ValueError(f"Could not find attention module at layer {layer_idx}")
    
    def get_qkv_projections(self, attention_module: nn.Module) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """Extract Q, K, V projections."""
        # Standard BERT/RoBERTa structure
        if hasattr(attention_module, 'query') and hasattr(attention_module, 'key') and hasattr(attention_module, 'value'):
            return attention_module.query, attention_module.key, attention_module.value
        # DistilBERT structure
        elif hasattr(attention_module, 'q_lin') and hasattr(attention_module, 'k_lin') and hasattr(attention_module, 'v_lin'):
            return attention_module.q_lin, attention_module.k_lin, attention_module.v_lin
        else:
            raise ValueError("Could not find Q/K/V projections in attention module")
    
    def get_hidden_size(self, model: nn.Module) -> int:
        """Get hidden dimension."""
        if hasattr(model, 'config'):
            return model.config.hidden_size
        else:
            raise ValueError("Model does not have config attribute")
    
    def get_num_attention_heads(self, model: nn.Module) -> int:
        """Get number of attention heads."""
        if hasattr(model, 'config'):
            return model.config.num_attention_heads
        else:
            raise ValueError("Model does not have config attribute")
    
    def replace_attention(self, model: nn.Module, layer_idx: int, new_attention: nn.Module):
        """Replace attention module with AMRPA wrapper."""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT, RoBERTa
            model.encoder.layer[layer_idx].attention.self = new_attention
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            # DistilBERT
            model.transformer.layer[layer_idx].attention = new_attention
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'albert_layer_groups'):
            # ALBERT
            group_idx = layer_idx // model.config.inner_group_num
            inner_idx = layer_idx % model.config.inner_group_num
            model.encoder.albert_layer_groups[group_idx].albert_layers[inner_idx].attention = new_attention
        else:
            raise ValueError(f"Could not replace attention at layer {layer_idx}")
    
    def extract_hidden_states(self, *args, **kwargs):
        """Extract hidden states from forward arguments."""
        # First positional argument is usually hidden_states
        if len(args) > 0:
            return args[0]
        # Check kwargs
        elif 'hidden_states' in kwargs:
            return kwargs['hidden_states']
        else:
            raise ValueError("Could not extract hidden_states from arguments")
    
    def format_output(self, attention_output, *args, **kwargs):
        """Format AMRPA output to match expected format."""
        # Most encoder-only models (BERT, RoBERTa) expect a tuple from their self-attention
        # modules, which is then unpacked as (attention_output, attention_weights).
        
        # We check if attention weights were requested
        output_attentions = kwargs.get('output_attentions', False)
        
        # For BERT/RoBERTa compatibility, we should almost always return a tuple
        # if the caller expects to unpack it.
        if isinstance(attention_output, tuple):
            return attention_output
        
        # If we just have the tensor, wrap it in a tuple
        # We use None for weights if they weren't requested, but still return the tuple
        # to satisfy the unpacking: attention_output, attn_weights = self.self(...)
        return (attention_output, None)
    
    def _get_layer(self, model: nn.Module, layer_idx: int) -> nn.Module:
        """Get entire transformer layer."""
        if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            return model.encoder.layer[layer_idx]
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'layer'):
            return model.transformer.layer[layer_idx]
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'albert_layer_groups'):
            group_idx = layer_idx // model.config.inner_group_num
            inner_idx = layer_idx % model.config.inner_group_num
            return model.encoder.albert_layer_groups[group_idx].albert_layers[inner_idx]
        else:
            raise ValueError(f"Could not find layer {layer_idx}")
    
    def freeze_embeddings(self, model: nn.Module):
        """Freeze embedding layers."""
        if hasattr(model, 'embeddings'):
            for param in model.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(model, 'bert') and hasattr(model.bert, 'embeddings'):
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(model, 'roberta') and hasattr(model.roberta, 'embeddings'):
            for param in model.roberta.embeddings.parameters():
                param.requires_grad = False
        elif hasattr(model, 'distilbert') and hasattr(model.distilbert, 'embeddings'):
            for param in model.distilbert.embeddings.parameters():
                param.requires_grad = False


class BERTAdapter(EncoderOnlyAdapter):
    """Specific adapter for BERT models."""
    
    def __init__(self):
        super().__init__(model_type="bert")


class RoBERTaAdapter(EncoderOnlyAdapter):
    """Specific adapter for RoBERTa models."""
    
    def __init__(self):
        super().__init__(model_type="roberta")


class DistilBERTAdapter(EncoderOnlyAdapter):
    """Specific adapter for DistilBERT models."""
    
    def __init__(self):
        super().__init__(model_type="distilbert")


class ALBERTAdapter(EncoderOnlyAdapter):
    """Specific adapter for ALBERT models."""
    
    def __init__(self):
        super().__init__(model_type="albert")
