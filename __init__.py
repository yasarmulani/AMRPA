"""
AMRPA - Adaptive Multi-Layer Recursive Preconditioned Attention

A modular, architecture-agnostic library for augmenting transformer models
with cross-layer memory integration.

Example usage:
    >>> from amrpa import inject_amrpa, AMRPAConfig
    >>> from transformers import AutoModel
    >>> 
    >>> # Load model
    >>> model = AutoModel.from_pretrained('roberta-base')
    >>> 
    >>> # Simple injection with defaults
    >>> model = inject_amrpa(model)
    >>> 
    >>> # Custom configuration
    >>> config = AMRPAConfig(
    ...     memory_decay=0.95,
    ...     target_layers='last_6',
    ...     gate_sensitivity=2.5
    ... )
    >>> model = inject_amrpa(model, config=config)
"""

from .core.config import AMRPAConfig
from .injection.injector import (
    inject_amrpa,
    get_amrpa_wrappers,
    reset_amrpa_history,
    get_amrpa_metrics
)

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "AMRPAConfig",
    
    # Main API
    "inject_amrpa",
    "get_amrpa_wrappers",
    "reset_amrpa_history",
    "get_amrpa_metrics",
]
