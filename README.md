# AMRPA - Adaptive Multi-Layer Recursive Preconditioned Attention

A modular, architecture-agnostic library for augmenting transformer models with cross-layer memory integration.

## Features

- **Architecture-Agnostic**: Works with any HuggingFace transformer (BERT, RoBERTa, GPT-2, T5, etc.)
- **Flexible Layer Targeting**: Inject AMRPA into specific layers or layer ranges
- **Four Key Innovations**:
  1. **Smart Gatekeeper**: Similarity-based gating for controlled memory usage
  2. **Dynamic Memory Selection**: MLP-based learning of layer relevance
  3. **Fading Ink**: Exponential decay + noise for gradient stability
  4. **Adaptive Memory Depth**: Layer-specific window sizes

## Installation

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from amrpa import inject_amrpa
from transformers import AutoModel

# Load your model
model = AutoModel.from_pretrained('roberta-base')

# Inject AMRPA with defaults
model = inject_amrpa(model)

# Use the model normally
outputs = model(**inputs)
```

### Custom Configuration

```python
from amrpa import inject_amrpa, AMRPAConfig

# Create custom configuration
config = AMRPAConfig(
    memory_decay=0.95,          # Decay factor for older patterns
    memory_noise=0.001,          # Noise level
    target_layers='last_6',      # Inject into last 6 layers
    gate_sensitivity=2.5,        # Gate scaling factor
    alpha_temperature=0.3        # Selection temperature
)

# Inject with custom config
model = inject_amrpa(model, config=config)
```

### Specific Layer Targeting

```python
# Target specific layers
model = inject_amrpa(model, target_layers=[8, 9, 10, 11])

# Or use string patterns
model = inject_amrpa(model, target_layers='last_4')
model = inject_amrpa(model, target_layers='first_3')
model = inject_amrpa(model, target_layers='middle_6')
model = inject_amrpa(model, target_layers='6:10')  # Slice notation
```

## Configuration Options

### Memory Parameters
- `memory_decay` (float, default=0.9): Exponential decay factor γ
- `memory_noise` (float, default=0.001): Noise level ε
- `adaptive_window` (bool, default=True): Use adaptive window sizing
- `max_window_size` (int, default=4): Maximum memory window

### Gating Parameters
- `gate_sensitivity` (float, default=2.0): Scaling factor γ_g
- `gate_bias` (float, default=-0.25): Bias term b_g
- `gate_activation` (str, default='sigmoid'): Activation function

### Selection Parameters
- `alpha_temperature` (float, default=0.25): Softmax temperature
- `alpha_mlp_hidden` (int, default=384): MLP hidden dimension
- `alpha_mlp_layers` (int, default=2): Number of MLP layers

### Injection Parameters
- `target_layers` (str|List[int], default='last_4'): Layer targeting
- `freeze_base_model` (bool, default=True): Freeze non-AMRPA layers
- `freeze_embeddings` (bool, default=True): Freeze embeddings

### Training Parameters
- `dropout` (float, default=0.2): Dropout rate
- `diversity_weight` (float, default=0.005): Alpha diversity loss weight
- `gate_reg_weight` (float, default=0.05): Gate regularization weight

## Advanced Usage

### Collecting Metrics

```python
from amrpa import get_amrpa_metrics, reset_amrpa_history

# Reset history before each batch
reset_amrpa_history(model)

# Forward pass
outputs = model(**inputs)

# Collect metrics
metrics = get_amrpa_metrics(model)
print(f"Gate impact: {metrics['gate_impact'].mean():.4f}")
print(f"Alpha diversity: {metrics['alpha_diversity'].mean():.4f}")
```

### Training with AMRPA

```python
from transformers import Trainer

# Inject AMRPA
model = inject_amrpa(model, config=config)

# Train normally
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

## Supported Architectures

### Currently Implemented
- ✅ Encoder-only (BERT, RoBERTa, DistilBERT, ALBERT)

### Coming Soon
- 🚧 Decoder-only (GPT-2, LLaMA, Mistral)
- 🚧 Encoder-decoder (T5, BART)

## Citation

If you use AMRPA in your research, please cite:

```bibtex
@article{amrpa2024,
  title={Adaptive Multi-Layer Recursive Preconditioned Attention},
  author={Your Name},
  year={2024}
}
```

## License

MIT License
