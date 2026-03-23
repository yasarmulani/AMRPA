"""
AMRPA Configuration Module

This module defines the configuration class for AMRPA mechanism.
"""

from dataclasses import dataclass, field
from typing import Union, List, Optional


@dataclass
class AMRPAConfig:
    """Configuration for AMRPA (Adaptive Multi-Layer Recursive Preconditioned Attention).
    
    This class contains all hyperparameters for the AMRPA mechanism including
    memory construction, gating, selection, and training parameters.
    
    Attributes:
        # Memory Parameters
        memory_decay (float): Exponential decay factor for older attention patterns (γ).
            Range: [0.5, 0.95]. Default: 0.9
        memory_noise (float): Noise level added to prevent complete information loss (ε).
            Range: [0.0, 0.01]. Default: 0.001
        adaptive_window (bool): Whether to use adaptive window sizing based on layer depth.
            Default: True
        max_window_size (int): Maximum number of past layers to remember.
            Range: [1, 10]. Default: 4
        
        # Gating Parameters
        gate_sensitivity (float): Learnable scaling factor for gate computation (γ_g).
            Range: [0.5, 5.0]. Default: 2.0
        gate_bias (float): Learnable bias term for gate computation (b_g).
            Range: [-2.0, 2.0]. Default: -0.25
        gate_activation (str): Activation function for gating.
            Options: ['sigmoid', 'tanh', 'relu']. Default: 'sigmoid'
        
        # Selection Parameters
        alpha_temperature (float): Temperature for softmax in alpha weight computation.
            Range: [0.1, 2.0]. Default: 0.25
        alpha_mlp_hidden (int): Hidden dimension for alpha MLP.
            Range: [64, 1024]. Default: 384
        alpha_mlp_layers (int): Number of layers in alpha MLP.
            Range: [1, 4]. Default: 2
        
        # Injection Parameters
        target_layers (Union[str, List[int]]): Which layers to inject AMRPA into.
            Options: 'all', 'last_N', 'first_N', 'middle_N', [list of indices], 'start:end'
            Default: 'last_4'
        freeze_base_model (bool): Whether to freeze non-AMRPA layers.
            Default: True
        freeze_embeddings (bool): Whether to freeze embedding layers.
            Default: True
        
        # Training Parameters
        dropout (float): Dropout rate for AMRPA components.
            Range: [0.0, 0.7]. Default: 0.2
        diversity_weight (float): Weight for alpha diversity regularization loss.
            Range: [0.0, 0.1]. Default: 0.005
        gate_reg_weight (float): Weight for gate regularization loss.
            Range: [0.0, 0.2]. Default: 0.05
        
        # Tracking Parameters
        track_metrics (bool): Whether to track AMRPA mechanism metrics.
            Default: True
        metric_device (str): Device to store metrics on ('cpu' or 'cuda').
            Default: 'cpu'
    """
    
    # Memory parameters
    memory_decay: float = 0.9
    memory_noise: float = 0.001
    adaptive_window: bool = True
    max_window_size: int = 4
    
    # Gating parameters
    gate_sensitivity: float = 2.0
    gate_bias: float = -0.25
    gate_activation: str = "sigmoid"
    
    # Selection parameters
    alpha_temperature: float = 0.25
    alpha_mlp_hidden: int = 384
    alpha_mlp_layers: int = 2
    
    # Injection parameters
    target_layers: Union[str, List[int]] = "last_4"
    freeze_base_model: bool = True
    freeze_embeddings: bool = True
    
    # Training parameters
    dropout: float = 0.2
    diversity_weight: float = 0.005
    gate_reg_weight: float = 0.05
    
    # Tracking parameters
    track_metrics: bool = True
    metric_device: str = "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate all configuration parameters."""
        # Memory parameters
        if not 0.0 <= self.memory_decay <= 1.0:
            raise ValueError(f"memory_decay must be between 0 and 1, got {self.memory_decay}")
        
        if self.memory_noise < 0:
            raise ValueError(f"memory_noise must be non-negative, got {self.memory_noise}")
        
        if self.max_window_size < 1:
            raise ValueError(f"max_window_size must be at least 1, got {self.max_window_size}")
        
        # Gating parameters
        if self.gate_activation not in ['sigmoid', 'tanh', 'relu']:
            raise ValueError(f"gate_activation must be one of ['sigmoid', 'tanh', 'relu'], got {self.gate_activation}")
        
        # Selection parameters
        if self.alpha_mlp_layers < 1:
            raise ValueError(f"alpha_mlp_layers must be at least 1, got {self.alpha_mlp_layers}")
        
        # Training parameters
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        
        # Injection parameters
        if not isinstance(self.target_layers, (str, list)):
            raise ValueError(f"target_layers must be a string or list of integers, got {type(self.target_layers)}")
        
        if isinstance(self.target_layers, list):
            if not all(isinstance(idx, int) for idx in self.target_layers):
                raise ValueError("target_layers list must contain only integers")
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'memory_decay': self.memory_decay,
            'memory_noise': self.memory_noise,
            'adaptive_window': self.adaptive_window,
            'max_window_size': self.max_window_size,
            'gate_sensitivity': self.gate_sensitivity,
            'gate_bias': self.gate_bias,
            'gate_activation': self.gate_activation,
            'alpha_temperature': self.alpha_temperature,
            'alpha_mlp_hidden': self.alpha_mlp_hidden,
            'alpha_mlp_layers': self.alpha_mlp_layers,
            'target_layers': self.target_layers,
            'freeze_base_model': self.freeze_base_model,
            'freeze_embeddings': self.freeze_embeddings,
            'dropout': self.dropout,
            'diversity_weight': self.diversity_weight,
            'gate_reg_weight': self.gate_reg_weight,
            'track_metrics': self.track_metrics,
            'metric_device': self.metric_device,
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def __repr__(self):
        """String representation of config."""
        return f"AMRPAConfig({self.to_dict()})"
