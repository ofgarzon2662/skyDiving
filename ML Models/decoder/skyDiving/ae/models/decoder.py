from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Decoder"]

class Decoder(nn.Module):
    """Convolutional decoder reconstructing the input image."""

    def __init__(self, num_filters: int = 16, use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1)
        in_channels_conv2 = num_filters * 2 if use_skip else num_filters
        self.conv2 = nn.Conv2d(in_channels_conv2, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_filters, 1, kernel_size=3, padding=1)

    def forward(self, enc_x1, x):  # type: ignore[override]
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, size=(14, 14), mode="nearest")

        if self.use_skip:
            x = torch.cat((x, enc_x1), dim=1)

        x = F.relu(self.conv2(x))
        x = F.interpolate(x, size=(28, 28), mode="nearest")
        x = torch.sigmoid(self.conv3(x))
        return x 