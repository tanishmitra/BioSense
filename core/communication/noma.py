import numpy as np


def superpose_signals(s_strong, s_weak, power):
    a_s = np.sqrt(power["strong"])
    a_w = np.sqrt(power["weak"])

    tx_strong = a_s * s_strong
    tx_weak = a_w * s_weak

    tx_superposed = tx_strong + tx_weak

    return tx_superposed, tx_strong, tx_weak