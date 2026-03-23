"""
Tests for LayerTargeter

This module tests the layer targeting utility including:
- String patterns (all, last_N, first_N, middle_N)
- Slice notation
- List specifications
- Edge cases and validation
"""

import pytest
from amrpa.utils.layer_utils import LayerTargeter


class TestLayerTargeterBasicPatterns:
    """Test basic layer targeting patterns."""
    
    def test_all_layers(self):
        """Test 'all' pattern."""
        result = LayerTargeter.resolve('all', total_layers=12)
        assert result == list(range(12))
        assert len(result) == 12
    
    def test_last_n_layers(self):
        """Test 'last_N' pattern."""
        result = LayerTargeter.resolve('last_4', total_layers=12)
        assert result == [8, 9, 10, 11]
        
        result = LayerTargeter.resolve('last_6', total_layers=12)
        assert result == [6, 7, 8, 9, 10, 11]
    
    def test_first_n_layers(self):
        """Test 'first_N' pattern."""
        result = LayerTargeter.resolve('first_3', total_layers=12)
        assert result == [0, 1, 2]
        
        result = LayerTargeter.resolve('first_5', total_layers=12)
        assert result == [0, 1, 2, 3, 4]
    
    def test_middle_n_layers(self):
        """Test 'middle_N' pattern."""
        result = LayerTargeter.resolve('middle_4', total_layers=12)
        assert result == [4, 5, 6, 7]
        
        result = LayerTargeter.resolve('middle_6', total_layers=12)
        assert result == [3, 4, 5, 6, 7, 8]


class TestLayerTargeterSliceNotation:
    """Test slice notation patterns."""
    
    def test_simple_slice(self):
        """Test simple slice notation."""
        result = LayerTargeter.resolve('6:10', total_layers=12)
        assert result == [6, 7, 8, 9]
    
    def test_slice_from_start(self):
        """Test slice from start."""
        result = LayerTargeter.resolve(':5', total_layers=12)
        assert result == [0, 1, 2, 3, 4]
    
    def test_slice_to_end(self):
        """Test slice to end."""
        result = LayerTargeter.resolve('8:', total_layers=12)
        assert result == [8, 9, 10, 11]
    
    def test_full_slice(self):
        """Test full slice."""
        result = LayerTargeter.resolve(':', total_layers=12)
        assert result == list(range(12))


class TestLayerTargeterListSpecification:
    """Test list-based layer specifications."""
    
    def test_simple_list(self):
        """Test simple list of layers."""
        result = LayerTargeter.resolve([0, 1, 2], total_layers=12)
        assert result == [0, 1, 2]
    
    def test_unsorted_list(self):
        """Test that unsorted list gets sorted."""
        result = LayerTargeter.resolve([5, 2, 8, 1], total_layers=12)
        assert result == [1, 2, 5, 8]
    
    def test_list_with_duplicates(self):
        """Test list with duplicate values."""
        result = LayerTargeter.resolve([1, 2, 2, 3], total_layers=12)
        assert result == [1, 2, 3]


class TestLayerTargeterEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_last_n_exceeds_total(self):
        """Test last_N when N > total_layers."""
        result = LayerTargeter.resolve('last_20', total_layers=12)
        assert result == list(range(12))
    
    def test_first_n_exceeds_total(self):
        """Test first_N when N > total_layers."""
        result = LayerTargeter.resolve('first_20', total_layers=12)
        assert result == list(range(12))
    
    def test_middle_n_exceeds_total(self):
        """Test middle_N when N > total_layers."""
        result = LayerTargeter.resolve('middle_20', total_layers=12)
        assert result == list(range(12))
    
    def test_single_layer_model(self):
        """Test with single layer model."""
        result = LayerTargeter.resolve('all', total_layers=1)
        assert result == [0]
    
    def test_last_1(self):
        """Test last_1 pattern."""
        result = LayerTargeter.resolve('last_1', total_layers=12)
        assert result == [11]
    
    def test_first_1(self):
        """Test first_1 pattern."""
        result = LayerTargeter.resolve('first_1', total_layers=12)
        assert result == [0]


class TestLayerTargeterValidation:
    """Test validation and error handling."""
    
    def test_invalid_pattern(self):
        """Test invalid pattern string."""
        with pytest.raises(ValueError, match="Invalid target specification"):
            LayerTargeter.resolve('invalid_pattern', total_layers=12)
    
    def test_negative_layer_index(self):
        """Test negative layer index in list."""
        with pytest.raises(ValueError, match="Layer indices must be non-negative"):
            LayerTargeter.resolve([-1, 0, 1], total_layers=12)
    
    def test_layer_index_out_of_range(self):
        """Test layer index exceeding total layers."""
        with pytest.raises(ValueError, match="Layer index .* exceeds total layers"):
            LayerTargeter.resolve([0, 1, 15], total_layers=12)
    
    def test_invalid_total_layers(self):
        """Test invalid total_layers value."""
        with pytest.raises(ValueError, match="total_layers must be positive"):
            LayerTargeter.resolve('all', total_layers=0)
    
    def test_empty_list(self):
        """Test empty list."""
        with pytest.raises(ValueError, match="target_layers list cannot be empty"):
            LayerTargeter.resolve([], total_layers=12)


class TestLayerTargeterDescription:
    """Test description generation."""
    
    def test_description_all_layers(self):
        """Test description for all layers."""
        layers = list(range(12))
        desc = LayerTargeter.get_description(layers, total_layers=12)
        assert "all 12 layers" in desc.lower()
    
    def test_description_last_layers(self):
        """Test description for last N layers."""
        layers = [8, 9, 10, 11]
        desc = LayerTargeter.get_description(layers, total_layers=12)
        assert "last 4 layers" in desc.lower()
    
    def test_description_first_layers(self):
        """Test description for first N layers."""
        layers = [0, 1, 2]
        desc = LayerTargeter.get_description(layers, total_layers=12)
        assert "first 3 layers" in desc.lower()
    
    def test_description_custom_layers(self):
        """Test description for custom layer selection."""
        layers = [1, 3, 5, 7]
        desc = LayerTargeter.get_description(layers, total_layers=12)
        assert "layers [1, 3, 5, 7]" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
