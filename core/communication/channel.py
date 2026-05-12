import numpy as np


def rayleigh_channel(num_bits):
    return (
        np.random.randn(num_bits) + 1j * np.random.randn(num_bits)
    ) / np.sqrt(2)


def apply_channel(h, signal):
    return h * signal