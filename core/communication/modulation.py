import numpy as np


def bpsk_modulate(bits):
    return (2 * bits - 1).astype(np.float32)


def bpsk_demodulate(symbols):
    return (np.real(symbols) >= 0).astype(int)