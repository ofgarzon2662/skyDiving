from __future__ import annotations

import numpy as np

__all__ = ["add_noise"]

def add_noise(
    images: np.ndarray,
    steps: int = 1,
    low: float = -0.2,
    high: float = 0.2,
) -> np.ndarray:
    """Return a copy of *images* with uniform random noise added.

    The images are clipped to stay within \[0, 1].
    """
    noisy = images.copy()
    for _ in range(steps):
        noisy += np.random.uniform(low, high, size=noisy.shape)
        np.clip(noisy, 0.0, 1.0, out=noisy)
    return noisy 