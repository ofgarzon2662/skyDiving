from __future__ import annotations

import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder

__all__ = ["AutoEncoder"]

class AutoEncoder(nn.Module):
    """Full auto-encoder model consisting of encoder and decoder."""

    def __init__(self, num_filters: int = 16, use_skip: bool = False):
        super().__init__()
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters, use_skip)

    def forward(self, x):  # type: ignore[override]
        enc_x1, enc_x2 = self.encoder(x)
        return self.decoder(enc_x1, enc_x2) 