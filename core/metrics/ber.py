import numpy as np


def calculate_ber(original, decoded):
    return np.mean(original != decoded)