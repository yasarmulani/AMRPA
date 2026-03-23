"""
Selection Module

This module implements the dynamic memory selection mechanism for AMRPA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SelectionModule(nn.Module):
    """Dynamic memory selection using MLP.
    
    This module implements Claim 2: Dynamic Memory Selection
    - Learns to weight different memory patterns
    - Uses MLP to compute selection scores
    - Applies temperature-scaled softmax
    """
    
    def __init__(self, config, hidden_size: int):
        """Initialize selection module.
        
        Args:
            config: AMRPAConfig instance
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Projection for attention patterns
        self.proj_attention = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # MLP for alpha computation
        layers = []
        input_dim = hidden_size * 2  # Concatenate query and projected attention
        
        for i in range(config.alpha_mlp_layers):
            if i == config.alpha_mlp_layers - 1:
                # Last layer outputs single score
                layers.append(nn.Linear(input_dim if i == 0 else config.alpha_mlp_hidden, 1))
            else:
                # Hidden layers
                layers.append(nn.Linear(
                    input_dim if i == 0 else config.alpha_mlp_hidden,
                    config.alpha_mlp_hidden
                ))
                layers.append(nn.ReLU())
                if config.dropout > 0:
                    layers.append(nn.Dropout(config.dropout))
        
        self.mlp_alpha = nn.Sequential(*layers)
    
    def compute_alpha_scores(
        self,
        query: torch.Tensor,
        memory_patterns: List[torch.Tensor],
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute selection scores for each memory pattern.
        
        Args:
            query: Query tensor (batch, seq, hidden)
            memory_patterns: List of decayed attention patterns
            values: Value tensor (batch, seq, hidden)
            
        Returns:
            Alpha scores (batch, seq, window_size)
        """
        alpha_scores = []
        
        for past_attention in memory_patterns:
            # Project past attention through values
            projected_values = torch.matmul(past_attention, values)  # (batch, seq, hidden)
            proj_attention = self.proj_attention(projected_values)  # (batch, seq, hidden)
            
            # Concatenate with query
            alpha_input = torch.cat([query, proj_attention], dim=-1)  # (batch, seq, 2*hidden)
            
            # Compute score
            alpha_score = self.mlp_alpha(alpha_input)  # (batch, seq, 1)
            alpha_scores.append(alpha_score)
        
        # Stack scores
        alpha_tensor = torch.cat(alpha_scores, dim=-1)  # (batch, seq, window_size)
        
        return alpha_tensor
    
    def forward(
        self,
        query: torch.Tensor,
        memory_patterns: List[torch.Tensor],
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute alpha weights for memory selection.
        
        Args:
            query: Query tensor (batch, seq, hidden)
            memory_patterns: List of decayed attention patterns
            values: Value tensor (batch, seq, hidden)
            
        Returns:
            Alpha weights (batch, seq, window_size)
        """
        if not memory_patterns:
            return None
        
        # Compute scores
        alpha_scores = self.compute_alpha_scores(query, memory_patterns, values)
        
        # Apply temperature-scaled softmax
        alpha_weights = F.softmax(alpha_scores / self.config.alpha_temperature, dim=-1)
        
        return alpha_weights
