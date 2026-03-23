"""
Layer Targeting Utilities

This module provides utilities for flexible layer targeting in AMRPA injection.
"""

from typing import Union, List


class LayerTargeter:
    """Utility class for resolving layer targeting specifications."""
    
    @staticmethod
    def resolve(target: Union[str, List[int]], total_layers: int) -> List[int]:
        """Resolve layer targeting specification into list of layer indices.
        
        Args:
            target: Layer targeting specification. Can be:
                - List of integers: [8, 9, 10, 11]
                - 'all': All layers
                - 'last_N': Last N layers
                - 'first_N': First N layers
                - 'middle_N': Middle N layers
                - 'start:end': Slice notation (Python-style)
                - 'start:end:step': Slice with step
            total_layers: Total number of layers in the model
            
        Returns:
            List of layer indices (0-indexed)
            
        Examples:
            >>> LayerTargeter.resolve('last_4', 12)
            [8, 9, 10, 11]
            >>> LayerTargeter.resolve('first_3', 12)
            [0, 1, 2]
            >>> LayerTargeter.resolve('6:10', 12)
            [6, 7, 8, 9]
            >>> LayerTargeter.resolve('::2', 12)
            [0, 2, 4, 6, 8, 10]
        """
        if total_layers <= 0:
            raise ValueError("total_layers must be positive")

        if isinstance(target, list):
            if not target:
                raise ValueError("target_layers list cannot be empty")
            # Validate list
            for idx in target:
                if not isinstance(idx, int):
                    raise ValueError(f"Layer index must be integer, got {type(idx)}")
                if idx < 0:
                    raise ValueError(f"Layer indices must be non-negative, got {idx}")
                if idx >= total_layers:
                    raise ValueError(f"Layer index {idx} exceeds total layers ({total_layers})")
            return sorted(list(set(target)))
        
        if not isinstance(target, str):
            raise ValueError(f"target_layers must be a string or list of integers, got {type(target)}")
        
        # Handle string specifications
        target = target.strip()
        
        if target == 'all':
            return list(range(total_layers))
        
        if target.startswith('last_'):
            try:
                n = int(target.split('_')[1])
                if n <= 0:
                    raise ValueError(f"last_{n} must be positive")
                n = min(n, total_layers)
                return list(range(total_layers - n, total_layers))
            except (IndexError, ValueError) as e:
                if isinstance(e, ValueError) and "must be positive" in str(e):
                    raise e
                raise ValueError(f"Invalid last_N specification: {target}") from e
        
        if target.startswith('first_'):
            try:
                n = int(target.split('_')[1])
                if n <= 0:
                    raise ValueError(f"first_{n} must be positive")
                n = min(n, total_layers)
                return list(range(n))
            except (IndexError, ValueError) as e:
                if isinstance(e, ValueError) and "must be positive" in str(e):
                    raise e
                raise ValueError(f"Invalid first_N specification: {target}") from e
        
        if target.startswith('middle_'):
            try:
                n = int(target.split('_')[1])
                if n <= 0:
                    raise ValueError(f"middle_{n} must be positive")
                n = min(n, total_layers)
                start = (total_layers - n) // 2
                return list(range(start, start + n))
            except (IndexError, ValueError) as e:
                if isinstance(e, ValueError) and "must be positive" in str(e):
                    raise e
                raise ValueError(f"Invalid middle_N specification: {target}") from e
        
        if ':' in target:
            # Slice notation
            try:
                parts = target.split(':')
                start_raw = parts[0].strip()
                end_raw = parts[1].strip()
                step_raw = parts[2].strip() if len(parts) > 2 else ""
                
                start = int(start_raw) if start_raw else 0
                end = int(end_raw) if end_raw else total_layers
                step = int(step_raw) if step_raw else 1
                
                # Use Python's built-in range for slice handling
                # but we need to cap it to total_layers for the test cases
                layer_range = range(total_layers)
                indices = list(range(total_layers)[start:end:step])
                return indices
            except (IndexError, ValueError) as e:
                raise ValueError(f"Invalid slice specification: {target}") from e
        
        raise ValueError(f"Invalid target specification: {target}")
    
    @staticmethod
    def validate_layers(layer_indices: List[int], total_layers: int):
        """Validate that layer indices are valid.
        
        Args:
            layer_indices: List of layer indices
            total_layers: Total number of layers
            
        Raises:
            ValueError: If any index is invalid
        """
        for idx in layer_indices:
            if idx < 0 or idx >= total_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {total_layers-1}]")
    
    @staticmethod
    def get_description(layer_indices: List[int], total_layers: int) -> str:
        """Get human-readable description of layer targeting.
        
        Args:
            layer_indices: List of layer indices
            total_layers: Total number of layers
            
        Returns:
            Description string
        """
        if not layer_indices:
            return "No layers"
        
        if len(layer_indices) == total_layers:
            return f"All {total_layers} layers"
        
        if layer_indices == list(range(total_layers - len(layer_indices), total_layers)):
            return f"Last {len(layer_indices)} layers ({layer_indices[0]}-{layer_indices[-1]})"
        
        if layer_indices == list(range(len(layer_indices))):
            return f"First {len(layer_indices)} layers ({layer_indices[0]}-{layer_indices[-1]})"
        
        if len(layer_indices) <= 5:
            return f"layers {layer_indices}"
        else:
            return f"{len(layer_indices)} layers: [{layer_indices[0]}, {layer_indices[1]}, ..., {layer_indices[-1]}]"

