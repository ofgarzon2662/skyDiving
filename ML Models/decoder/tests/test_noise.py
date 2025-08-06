import numpy as np
from skyDiving.ae.noise import add_noise

def test_add_noise_clipping():
    x = np.zeros((1, 1, 2, 2))
    noisy = add_noise(x, steps=1, low=-0.5, high=0.5)
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0 