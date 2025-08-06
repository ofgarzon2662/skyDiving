from __future__ import annotations

from dataclasses import dataclass, field
import torch

__all__ = ["TrainingConfig"]

@dataclass
class TrainingConfig:
    """Configuration values for training the auto-encoder."""

    batch_size: int = 128
    max_num_train: int = 1024
    max_num_test: int = 128
    epochs: int = 10
    learning_rate: float = 0.01
    num_filters: int = 16
    num_extra_noise_steps: int = 15
    seed: int = 776
    device: torch.device = field(
        default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    def __post_init__(self) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(self.seed) 