"""
Architecture Detector

This module detects the architecture type of a transformer model.
"""

import torch.nn as nn
from typing import Optional


class ArchitectureDetector:
    """Detects transformer architecture type."""
    
    @staticmethod
    def detect(model: nn.Module) -> str:
        """Detect architecture type from model.
        
        Args:
            model: Transformer model
            
        Returns:
            Architecture type string: 'encoder_only', 'decoder_only', or 'encoder_decoder'
            
        Raises:
            ValueError: If architecture cannot be detected
        """
        # Check config if available
        if hasattr(model, 'config'):
            config = model.config
            
            # Check model_type attribute
            if hasattr(config, 'model_type'):
                model_type = config.model_type.lower()
                
                # Encoder-only models
                if model_type in ['bert', 'roberta', 'distilbert', 'albert', 'electra']:
                    return 'encoder_only'
                
                # Decoder-only models
                if model_type in ['gpt2', 'gpt_neo', 'gptj', 'gpt_neox', 'llama', 'mistral', 'opt']:
                    return 'decoder_only'
                
                # Encoder-decoder models
                if model_type in ['t5', 'bart', 'mbart', 'pegasus', 'marian']:
                    return 'encoder_decoder'
        
        # Fallback: structural detection
        return ArchitectureDetector._detect_by_structure(model)
    
    @staticmethod
    def _detect_by_structure(model: nn.Module) -> str:
        """Detect architecture by model structure.
        
        Args:
            model: Transformer model
            
        Returns:
            Architecture type string
            
        Raises:
            ValueError: If architecture cannot be detected
        """
        # Check for encoder and decoder attributes
        has_encoder = hasattr(model, 'encoder')
        has_decoder = hasattr(model, 'decoder')
        
        # Encoder-decoder
        if has_encoder and has_decoder:
            return 'encoder_decoder'
        
        # Encoder-only
        if has_encoder and not has_decoder:
            # Check if it's actually encoder-only (not encoder-decoder with different naming)
            if hasattr(model.encoder, 'layer'):
                return 'encoder_only'
        
        # Decoder-only (check for transformer attribute common in GPT models)
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'h'):  # GPT-2 style
                return 'decoder_only'
            if hasattr(model.transformer, 'layers'):  # Some other decoder-only models
                return 'decoder_only'
        
        # Check for 'h' attribute (GPT-2 style)
        if hasattr(model, 'h'):
            return 'decoder_only'
        
        # Check for 'layers' attribute
        if hasattr(model, 'layers'):
            # Could be decoder-only
            return 'decoder_only'
        
        raise ValueError(
            "Could not detect model architecture. "
            "Please specify architecture type manually or ensure model has standard structure."
        )
    
    @staticmethod
    def get_model_info(model: nn.Module) -> dict:
        """Get detailed model information.
        
        Args:
            model: Transformer model
            
        Returns:
            Dictionary with model information
        """
        info = {
            'architecture': None,
            'model_type': None,
            'num_layers': None,
            'hidden_size': None,
            'num_attention_heads': None
        }
        
        try:
            info['architecture'] = ArchitectureDetector.detect(model)
        except:
            pass
        
        if hasattr(model, 'config'):
            config = model.config
            info['model_type'] = getattr(config, 'model_type', None)
            info['num_layers'] = getattr(config, 'num_hidden_layers', None)
            info['hidden_size'] = getattr(config, 'hidden_size', None)
            info['num_attention_heads'] = getattr(config, 'num_attention_heads', None)
        
        return info
