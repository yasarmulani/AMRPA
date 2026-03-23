# AMRPA Tests

This directory contains comprehensive tests for the AMRPA library.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── test_config.py           # Tests for AMRPAConfig
├── test_layer_utils.py      # Tests for LayerTargeter
├── test_core_modules.py     # Tests for Memory, Gating, Selection modules
└── test_integration.py      # End-to-end integration tests
```

## Running Tests

### Run all tests
```bash
pytest amrpa/tests/
```

### Run with verbose output
```bash
pytest amrpa/tests/ -v
```

### Run specific test file
```bash
pytest amrpa/tests/test_config.py -v
```

### Run specific test class
```bash
pytest amrpa/tests/test_config.py::TestAMRPAConfigDefaults -v
```

### Run specific test
```bash
pytest amrpa/tests/test_config.py::TestAMRPAConfigDefaults::test_default_initialization -v
```

### Run with coverage
```bash
pytest amrpa/tests/ --cov=amrpa --cov-report=html
```

### Skip slow tests
```bash
pytest amrpa/tests/ -m "not slow"
```

## Test Categories

### Unit Tests
- `test_config.py`: Configuration validation and defaults
- `test_layer_utils.py`: Layer targeting logic
- `test_core_modules.py`: Individual AMRPA components

### Integration Tests
- `test_integration.py`: Full pipeline tests with real models

## Requirements

Install test dependencies:
```bash
pip install -e ".[dev]"
```

Or manually:
```bash
pip install pytest pytest-cov transformers
```

## Writing New Tests

When adding new tests, follow these guidelines:

1. **Use descriptive names**: `test_memory_decay_with_zero_history`
2. **Use fixtures**: Reuse common setup code
3. **Test edge cases**: Empty inputs, boundary values, invalid inputs
4. **Add docstrings**: Explain what the test validates
5. **Use markers**: Mark slow or integration tests appropriately

Example:
```python
@pytest.mark.slow
def test_large_model_injection(self):
    """Test AMRPA injection on a large model."""
    # Test code here
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines. The integration tests use small models (`bert-tiny`) to ensure fast execution.
