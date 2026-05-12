import numpy as np


def allocate_power(g_s, g_w):
    eps = 1e-12

    inv = np.array([
        1.0 / (g_s + eps),
        1.0 / (g_w + eps)
    ])

    inv = inv / np.sum(inv)

    return {
        "strong": float(inv[0]),
        "weak": float(inv[1])
    }