"""
AMRPA Injector

This module provides the main injection logic for AMRPA.
"""

import torch.nn as nn
from typing import Union, List, Optional
from ..core.config import AMRPAConfig
from ..core.attention import AMRPAAttentionWrapper
from ..architectures.encoder_only import EncoderOnlyAdapter
from ..utils.layer_utils import LayerTargeter
from .detector import ArchitectureDetector


def inject_amrpa(
    model: nn.Module,
    config: Optional[AMRPAConfig] = None,
    mode: str = "auto",
    target_layers: Optional[Union[str, List[int]]] = None,
    adapter = None
) -> nn.Module:
    """Inject AMRPA into a transformer model.
    
    This is the main entry point for adding AMRPA to any HuggingFace model.
    
    Args:
        model: HuggingFace transformer model
        config: AMRPA configuration (uses default if None)
        mode: Architecture mode ('auto', 'encoder_only', 'decoder_only', 'encoder_decoder')
        target_layers: Which layers to inject AMRPA into (uses config if None)
        adapter: Custom architecture adapter (auto-detected if None)
        
    Returns:
        Modified model with AMRPA injected
        
    Examples:
        >>> # Simple usage with defaults
        >>> model = inject_amrpa(model)
        
        >>> # Custom configuration
        >>> config = AMRPAConfig(memory_decay=0.95, target_layers='last_6')
        >>> model = inject_amrpa(model, config=config)
        
        >>> # Specific layers
        >>> model = inject_amrpa(model, target_layers=[8, 9, 10, 11])
    """
    # Use default config if not provided
    if config is None:
        config = AMRPAConfig()
    
    # Override target_layers if provided
    if target_layers is not None:
        config.target_layers = target_layers
    
    # Detect architecture if mode is auto
    if mode == "auto":
        mode = ArchitectureDetector.detect(model)
        print(f"Detected architecture: {mode}")
    
    # Get appropriate adapter
    if adapter is None:
        adapter = _get_adapter(mode, model)
    
    # Validate model
    adapter.validate_model(model)
    
    # Get total number of layers
    total_layers = adapter.get_num_layers(model)
    
    # Resolve target layers
    layer_indices = LayerTargeter.resolve(config.target_layers, total_layers)
    
    print(f"\nInjecting AMRPA into {len(layer_indices)} layers:")
    print(f"  {LayerTargeter.get_description(layer_indices, total_layers)}")
    
    # Inject AMRPA into each target layer
    for layer_idx in layer_indices:
        # Get original attention module
        original_attention = adapter.get_attention_module(model, layer_idx)
        
        # Create AMRPA wrapper
        amrpa_wrapper = AMRPAAttentionWrapper(
            original_attention=original_attention,
            config=config,
            layer_idx=layer_idx,
            adapter=adapter
        )
        
        # Replace attention module
        adapter.replace_attention(model, layer_idx, amrpa_wrapper)
        
        print(f"  ✓ Layer {layer_idx}: AMRPA injected")
    
    # Freeze layers if requested
    if config.freeze_base_model:
        _freeze_non_amrpa_layers(model, adapter, layer_indices, total_layers)
    
    if config.freeze_embeddings:
        adapter.freeze_embeddings(model)
        print("  ✓ Embeddings frozen")
    
    print(f"\n✓ AMRPA injection complete!")
    
    return model


def _get_adapter(mode: str, model: nn.Module):
    """Get appropriate adapter for architecture mode.
    
    Args:
        mode: Architecture mode
        model: Transformer model
        
    Returns:
        Architecture adapter instance
    """
    if mode == "encoder_only":
        return EncoderOnlyAdapter()
    elif mode == "decoder_only":
        raise NotImplementedError("Decoder-only models not yet implemented")
    elif mode == "encoder_decoder":
        raise NotImplementedError("Encoder-decoder models not yet implemented")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _freeze_non_amrpa_layers(
    model: nn.Module,
    adapter,
    amrpa_layers: List[int],
    total_layers: int
):
    """Freeze layers that don't have AMRPA.
    
    Args:
        model: Transformer model
        adapter: Architecture adapter
        amrpa_layers: List of AMRPA layer indices
        total_layers: Total number of layers
    """
    amrpa_set = set(amrpa_layers)
    frozen_count = 0
    
    for layer_idx in range(total_layers):
        if layer_idx not in amrpa_set:
            adapter.freeze_layer(model, layer_idx)
            frozen_count += 1
    
    if frozen_count > 0:
        print(f"  ✓ {frozen_count} non-AMRPA layers frozen")


def get_amrpa_wrappers(model: nn.Module) -> List[AMRPAAttentionWrapper]:
    """Get all AMRPA wrappers from a model.
    
    Args:
        model: Model with AMRPA injected
        
    Returns:
        List of AMRPA wrapper instances
    """
    wrappers = []
    
    for module in model.modules():
        if isinstance(module, AMRPAAttentionWrapper):
            wrappers.append(module)
    
    return wrappers


def reset_amrpa_history(model: nn.Module):
    """Reset attention history for all AMRPA wrappers.
    
    Args:
        model: Model with AMRPA injected
    """
    for wrapper in get_amrpa_wrappers(model):
        wrapper.reset_history()


def get_amrpa_metrics(model: nn.Module) -> dict:
    """Collect metrics from all AMRPA wrappers.
    
    Args:
        model: Model with AMRPA injected
        
    Returns:
        Dictionary of aggregated metrics
    """
    wrappers = get_amrpa_wrappers(model)
    
    if not wrappers:
        return {}
    
    # Aggregate metrics
    all_metrics = {}
    for wrapper in wrappers:
        if hasattr(wrapper, 'last_metrics') and wrapper.last_metrics:
            for key, value in wrapper.last_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
    
    # Average metrics
    averaged_metrics = {}
    for key, values in all_metrics.items():
        if values:
            import torch
            stacked = torch.stack(values)
            averaged_metrics[key] = stacked.mean(dim=0)
    
    return averaged_metrics
