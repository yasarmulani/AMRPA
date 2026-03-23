"""
Gating Module

This module implements the similarity-based gating mechanism for AMRPA.
"""

import torch
import torch.nn as nn
import math


class GatingModule(nn.Module):
    """Similarity-based gating mechanism.
    
    This module implements Claim 1: Smart Gatekeeper
    - Computes similarity between current query and memory
    - Applies learnable scaling and bias
    - Uses activation function to produce gate values
    """
    
    def __init__(self, config, hidden_size: int):
        """Initialize gating module.
        
        Args:
            config: AMRPAConfig instance
            hidden_size: Hidden dimension size
        """
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        
        # Learnable parameters for gating
        self.gamma_g = nn.Parameter(torch.tensor(config.gate_sensitivity))
        self.bias_g = nn.Parameter(torch.tensor(config.gate_bias))
        
        # Projection for memory
        self.proj_memory = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Activation function
        if config.gate_activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif config.gate_activation == 'tanh':
            self.activation = nn.Tanh()
        elif config.gate_activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.gate_activation}")
    
    def compute_similarity(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute similarity between query and memory.
        
        Args:
            query: Query tensor (batch, seq, hidden)
            memory: Memory attention pattern (batch, seq, seq)
            values: Value tensor (batch, seq, hidden)
            
        Returns:
            Similarity scores (batch, seq)
        """
        # Project memory through values
        memory_values = torch.matmul(memory, values)  # (batch, seq, hidden)
        projected_memory = self.proj_memory(memory_values)  # (batch, seq, hidden)
        
        # Compute similarity (dot product)
        similarity = (query * projected_memory).sum(dim=-1)  # (batch, seq)
        
        # Scale by sqrt(d_k) for stability
        similarity = similarity / math.sqrt(self.hidden_size)
        
        return similarity
    
    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """Compute gate values.
        
        Args:
            query: Query tensor (batch, seq, hidden)
            memory: Memory attention pattern (batch, seq, seq)
            values: Value tensor (batch, seq, hidden)
            
        Returns:
            Gate values (batch, seq)
        """
        # Compute similarity
        similarity = self.compute_similarity(query, memory, values)
        
        # Apply learnable scaling and bias
        gate_logits = self.gamma_g * similarity + self.bias_g
        
        # Apply activation
        gates = self.activation(gate_logits)
        
        return gates
