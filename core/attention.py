"""
AMRPA Attention Wrapper

This module wraps the original attention mechanism with AMRPA logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .memory import MemoryModule
from .gating import GatingModule
from .selection import SelectionModule


class AMRPAAttentionWrapper(nn.Module):
    """Wraps attention module with AMRPA mechanism.
    
    This wrapper:
    - Maintains attention history
    - Constructs memory from past patterns
    - Computes selection weights (alpha)
    - Applies gating mechanism
    - Augments attention scores with memory
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        config,
        layer_idx: int,
        adapter
    ):
        """Initialize AMRPA wrapper.
        
        Args:
            original_attention: Original attention module to wrap
            config: AMRPAConfig instance
            layer_idx: Layer index (0-indexed)
            adapter: Architecture adapter instance
        """
        super().__init__()
        self.original_attention = original_attention
        self.config = config
        self.layer_idx = layer_idx + 1  # Convert to 1-indexed
        self.adapter = adapter
        
        # Get hidden size from adapter
        # We'll set this during first forward pass if needed
        self.hidden_size = None
        
        # AMRPA components (initialized lazily)
        self.memory_module = None
        self.gating_module = None
        self.selection_module = None
        self.memory_transform = None
        
        # Attention history (shared across batch)
        self.attention_history = []
        
        # Metrics tracking
        self.last_metrics = {}
    
    def _initialize_components(self, hidden_size: int):
        """Lazy initialization of AMRPA components.
        
        Args:
            hidden_size: Hidden dimension size
        """
        if self.hidden_size is not None:
            return
        
        self.hidden_size = hidden_size
        
        # Initialize modules
        self.memory_module = MemoryModule(self.config)
        self.gating_module = GatingModule(self.config, hidden_size)
        self.selection_module = SelectionModule(self.config, hidden_size)
        self.memory_transform = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def compute_memory_bias(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute memory bias to add to attention scores.
        
        Args:
            query: Query tensor (batch, seq, hidden)
            key: Key tensor (batch, seq, hidden)
            value: Value tensor (batch, seq, hidden)
            
        Returns:
            Tuple of (memory_bias, metrics)
        """
        batch_size, seq_len, hidden_size = query.shape
        device = query.device
        
        # Initialize components if needed
        self._initialize_components(hidden_size)
        
        # Initialize metrics
        metrics = {
            'gate_impact': torch.zeros(batch_size, device='cpu'),
            'alpha_diversity': torch.zeros(batch_size, device='cpu'),
            'memory_contribution': torch.zeros(batch_size, device='cpu'),
            'gate_variance': torch.zeros(batch_size, device='cpu'),
            'using_memory': torch.zeros(batch_size, device='cpu')
        }
        
        # Check if we have history
        if self.layer_idx <= 1 or not self.attention_history:
            return torch.zeros_like(torch.matmul(query, key.transpose(-2, -1))), metrics
        
        # Get window size
        window_size = self.memory_module.adaptive_window_size(self.layer_idx)
        memory_window = self.attention_history[-min(self.layer_idx - 1, window_size):]
        
        if not memory_window:
            return torch.zeros_like(torch.matmul(query, key.transpose(-2, -1))), metrics
        
        # Apply decay to patterns
        decayed_patterns = []
        for k, past_attention in enumerate(reversed(memory_window), 1):
            decayed = self.memory_module.apply_decay(past_attention, k)
            decayed_patterns.append(decayed)
        
        # Compute alpha weights
        alpha_weights = self.selection_module(query, decayed_patterns, value)
        
        # Compute alpha diversity (entropy)
        if alpha_weights is not None:
            token_entropy = -(alpha_weights * torch.log(alpha_weights + 1e-9)).sum(dim=-1)
            metrics['alpha_diversity'] = token_entropy.mean(dim=1).detach().cpu()
        
        # Construct memory
        memory = self.memory_module(self.attention_history, self.layer_idx, alpha_weights)
        
        if memory is None:
            return torch.zeros_like(torch.matmul(query, key.transpose(-2, -1))), metrics
        
        # Compute gates
        gates = self.gating_module(query, memory, value)  # (batch, seq)
        
        # Track gate metrics
        metrics['gate_impact'] = gates.mean(dim=1).detach().cpu()
        metrics['gate_variance'] = gates.var(dim=1).detach().cpu()
        metrics['using_memory'] = torch.ones(batch_size, device='cpu')
        
        # Transform memory and apply gating
        memory_values = torch.matmul(memory, value)  # (batch, seq, hidden)
        transformed_memory = self.memory_transform(memory_values)  # (batch, seq, hidden)
        gated_memory = gates.unsqueeze(-1) * transformed_memory  # (batch, seq, hidden)
        
        # Track memory contribution
        token_norms = gated_memory.norm(dim=-1)
        metrics['memory_contribution'] = token_norms.mean(dim=1).detach().cpu()
        
        # Compute memory bias
        memory_bias = torch.matmul(gated_memory, key.transpose(-2, -1))  # (batch, seq, seq)
        memory_bias = memory_bias / (hidden_size ** 0.5)  # Scale
        
        return memory_bias, metrics
    
    def forward(self, *args, **kwargs):
        """Forward pass with AMRPA augmentation.
        
        This method extracts Q, K, V from the original attention,
        computes AMRPA memory bias, and augments the attention scores.
        """
        # Extract hidden states using adapter
        hidden_states = self.adapter.extract_hidden_states(*args, **kwargs)
        
        # Get Q, K, V projections from original attention
        query_proj, key_proj, value_proj = self.adapter.get_qkv_projections(self.original_attention)
        
        # Compute Q, K, V
        query = query_proj(hidden_states)
        key = key_proj(hidden_states)
        value = value_proj(hidden_states)
        
        # Compute base attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = attention_scores / (query.size(-1) ** 0.5)
        
        # Apply attention mask if provided
        attention_mask = self.adapter.get_attention_mask(*args, **kwargs)
        if attention_mask is not None:
            # Handle different mask dimensions
            if attention_mask.dim() == 4:
                attention_mask = attention_mask.squeeze(1)
            if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.expand(-1, attention_scores.size(1), -1)
            attention_scores = attention_scores + attention_mask
        
        # Compute AMRPA memory bias
        memory_bias, metrics = self.compute_memory_bias(query, key, value)
        self.last_metrics = metrics
        
        # Add memory bias to scores
        final_scores = attention_scores + memory_bias
        
        # Compute attention probabilities
        attention_probs = F.softmax(final_scores, dim=-1)
        
        # Apply dropout if in training mode
        if self.training and hasattr(self.original_attention, 'dropout'):
            attention_probs = self.original_attention.dropout(attention_probs)
        
        # Compute context
        context = torch.matmul(attention_probs, value)
        
        # Store attention pattern in history
        self.attention_history.append(attention_probs.detach())
        
        # Format output using adapter
        output = self.adapter.format_output(context, *args, **kwargs)
        
        return output
    
    def reset_history(self):
        """Reset attention history."""
        self.attention_history = []
    
    def reset_metrics(self):
        """Reset metrics."""
        self.last_metrics = {}
