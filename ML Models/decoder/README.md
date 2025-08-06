# AutoEncoder Modular Implementation

This repository contains a refactored, **SOLID-compliant** implementation of a convolutional auto-encoder for denoising MNIST digits.

## Quick start

```
# Install dependencies
pip install -r requirements.txt

# Train the model
python -m skyDiving.ae.cli train --epochs 5
```

## Project structure

```
skyDiving/ae/
├── __init__.py
├── config.py          # Hyper-parameter dataclass
├── data.py            # Data loading / preprocessing
├── noise.py           # Image-level utilities
├── models/            # Neural-network components
├── trainer.py         # Training helper
└── cli.py             # Typer CLI entry-point
```

Each module has a single responsibility and exposes a minimal public interface, which makes unit-testing straightforward.
