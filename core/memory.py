"""
Memory Module

This module implements the memory construction logic for AMRPA.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional


class MemoryModule(nn.Module):
    """Constructs memory from past attention patterns.
    
    This module implements:
    - Claim 3: Fading Ink (exponential decay + noise)
    - Claim 4: Adaptive Memory Depth (layer-wise window sizing)
    """
    
    def __init__(self, config):
        """Initialize memory module.
        
        Args:
            config: AMRPAConfig instance
        """
        super().__init__()
        self.config = config
    
    def adaptive_window_size(self, layer_idx: int) -> int:
        """Compute adaptive window size based on layer depth.
        
        Claim 4: Adaptive Memory Depth
        
        Args:
            layer_idx: Current layer index (1-indexed)
            
        Returns:
            Window size for this layer
        """
        if not self.config.adaptive_window:
            return self.config.max_window_size
        
        # Layer-adaptive window sizing
        if layer_idx <= 2:
            return 1
        elif 2 < layer_idx <= 8:
            return math.floor(math.log2(layer_idx)) + 1
        else:
            return self.config.max_window_size
    
    def apply_decay(
        self,
        attention_pattern: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Apply exponential decay and noise to attention pattern.
        
        Claim 3: Fading Ink
        
        Args:
            attention_pattern: Past attention pattern (batch, seq, seq)
            k: Steps back in history (1-indexed)
            
        Returns:
            Decayed attention pattern
        """
        # Exponential decay
        decay_factor = self.config.memory_decay ** k
        
        # Add noise to prevent complete information loss
        noise = torch.rand_like(attention_pattern) * self.config.memory_noise
        
        decayed_pattern = decay_factor * attention_pattern + noise
        
        return decayed_pattern
    
    def construct_memory(
        self,
        attention_history: List[torch.Tensor],
        layer_idx: int,
        alpha_weights: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Construct memory from attention history.
        
        Args:
            attention_history: List of past attention patterns
            layer_idx: Current layer index (1-indexed)
            alpha_weights: Optional selection weights (batch, seq, window_size)
                         If None, uses uniform weighting
            
        Returns:
            Aggregated memory tensor or None if no history available
        """
        if not attention_history or layer_idx <= 1:
            return None
        
        # Get adaptive window size
        window_size = self.adaptive_window_size(layer_idx)
        
        # Get relevant history (most recent window_size patterns)
        memory_window = attention_history[-min(layer_idx - 1, window_size):]
        
        if not memory_window:
            return None
        
        # Apply decay to each pattern
        decayed_patterns = []
        for k, past_attention in enumerate(reversed(memory_window), 1):
            decayed = self.apply_decay(past_attention, k)
            decayed_patterns.append(decayed)
        
        # Stack patterns
        memory_stack = torch.stack(decayed_patterns, dim=-1)  # (batch, seq, seq, window)
        
        # Apply selection weights if provided
        if alpha_weights is not None:
            # alpha_weights: (batch, seq, window)
            # Expand for matrix multiplication
            weighted_memory = torch.sum(
                alpha_weights.unsqueeze(2) * memory_stack,
                dim=-1
            )  # (batch, seq, seq)
        else:
            # Uniform weighting
            weighted_memory = memory_stack.mean(dim=-1)
        
        return weighted_memory
    
    def forward(
        self,
        attention_history: List[torch.Tensor],
        layer_idx: int,
        alpha_weights: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Forward pass.
        
        Args:
            attention_history: List of past attention patterns
            layer_idx: Current layer index (1-indexed)
            alpha_weights: Optional selection weights
            
        Returns:
            Memory tensor or None
        """
        return self.construct_memory(attention_history, layer_idx, alpha_weights)
