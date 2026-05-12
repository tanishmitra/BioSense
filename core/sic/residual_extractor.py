import numpy as np


def residual_energy(residual):
    return np.sum(np.abs(residual) ** 2)


def residual_envelope(residual):
    return np.abs(residual)