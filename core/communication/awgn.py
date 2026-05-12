import numpy as np
from config.settings import NOISE_STD


def generate_noise(num_bits):
    noise_var = NOISE_STD ** 2

    return np.sqrt(noise_var / 2) * (
        np.random.randn(num_bits) +
        1j * np.random.randn(num_bits)
    )