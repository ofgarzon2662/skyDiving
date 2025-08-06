from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import TrainingConfig
from .noise import add_noise

__all__ = ["MNISTAutoencoderData"]

class MNISTAutoencoderData:
    """Prepare noisy-MNIST datasets and related *DataLoader* objects.*"""

    def __init__(self, config: TrainingConfig, data_dir: Path | str = ".") -> None:
        self.cfg = config
        self.data_dir = Path(data_dir)
        self._is_ready = False

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def train_dataloader(self) -> DataLoader:
        if not self._is_ready:
            self._setup()
        return self._train_dl

    def test_dataloader(self) -> DataLoader:
        if not self._is_ready:
            self._setup()
        return self._test_dl

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------
    def _setup(self) -> None:
        """Load data from disk and construct Torch tensors and loaders."""
        x_train = np.load(self.data_dir / "X_train1k.npy")[: self.cfg.max_num_train]
        x_test = np.load(self.data_dir / "X_test.npy")[: self.cfg.max_num_train]

        # Rescale pixels to [0, 1] and add channel dimension.
        x_train = (x_train / 255.0)[:, np.newaxis, :, :]
        x_test = (x_test / 255.0)[:, np.newaxis, :, :]

        # Add synthetic noise.
        x_train_noisy = add_noise(x_train, self.cfg.num_extra_noise_steps)
        x_test_noisy = add_noise(x_test, self.cfg.num_extra_noise_steps)

        # Convert to tensors.
        train_dataset = TensorDataset(
            torch.from_numpy(x_train_noisy).float(),
            torch.from_numpy(x_train).float(),
        )
        test_dataset = TensorDataset(
            torch.from_numpy(x_test_noisy).float(),
            torch.from_numpy(x_test).float(),
        )

        # Build loaders.
        self._train_dl = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.cfg.device.type == "cuda",
            drop_last=True,
        )
        self._test_dl = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.cfg.device.type == "cuda",
            drop_last=True,
        )
        self._is_ready = True 