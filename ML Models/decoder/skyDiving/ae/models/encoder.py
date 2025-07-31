from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Encoder"]

class Encoder(nn.Module):
    """Convolutional encoder mapping 28×28 inputs to latent feature maps."""

    def __init__(self, num_filters: int = 16):
        super().__init__()
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):  # type: ignore[override]
        x1 = F.relu(self.conv1(x))
        x1 = self.pool1(x1)
        x2 = F.relu(self.conv2(x1))
        x2 = self.pool2(x2)
        # Return both tensors to enable skip connections.
        return x1, x2 