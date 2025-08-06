from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

__all__ = ["Trainer"]

class Trainer:
    """Minimalist training helper for PyTorch models."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        loss_fn: Callable = F.binary_cross_entropy,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------
    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        epoch_loss = 0.0
        num_samples = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()

            batch_size = data.size(0)
            epoch_loss += loss.item() * batch_size
            num_samples += batch_size

        return epoch_loss / num_samples

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Evaluate model and return the average loss."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.loss_fn(output, target)

            batch_size = data.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------
    def fit(self, train_dl, val_dl, epochs: int) -> None:
        for epoch in range(epochs):
            start = time.time()
            train_loss = self.train_epoch(train_dl)
            val_loss = self.evaluate(val_dl)
            duration = time.time() - start
            print(
                f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | duration={duration:.2f}s"
            ) 