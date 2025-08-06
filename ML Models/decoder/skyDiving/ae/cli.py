from __future__ import annotations

import typer
import torch

from .config import TrainingConfig
from .data import MNISTAutoencoderData
from .models.autoencoder import AutoEncoder
from .trainer import Trainer

# Root Typer app executes the `train` logic directly so users can simply call
# ``python -m skyDiving.ae.cli --epochs 5``.

def train(
    batch_size: int = 128,
    epochs: int = 10,
    learning_rate: float = 0.01,
    num_filters: int = 16,
    num_noise_steps: int = 15,
    data_dir: str = ".",
):
    """Train the auto-encoder on the noisy MNIST subset."""

    cfg = TrainingConfig(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        num_filters=num_filters,
        num_extra_noise_steps=num_noise_steps,
    )

    # Prepare data
    dm = MNISTAutoencoderData(cfg, data_dir)

    # Build model & optimiser
    model = AutoEncoder(num_filters=cfg.num_filters)
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trainer = Trainer(model, optimiser, cfg.device)

    trainer.fit(dm.train_dataloader(), dm.test_dataloader(), cfg.epochs)


# ---------------------------------------------------------------------------
# Typer entry-point setup
# ---------------------------------------------------------------------------

def _cli_wrapper():
    """Invoke Typer using *train* as the root command."""

    typer.run(train)


if __name__ == "__main__":
    _cli_wrapper() 