import numpy as np


class WeakFirstSIC:

    def decode(self, rx, h_weak, h_strong,
               a_w, a_s,
               bits_w, bits_s):
        eps = 1e-12

        z_w = rx / ((h_weak * a_w) + eps)

        dec_bits_w = (np.real(z_w) >= 0).astype(int)

        dec_sym_w = (
            2 * dec_bits_w - 1
        ).astype(np.complex128)

        recon_weak = h_weak * (a_w * dec_sym_w)

        rx_after_weak = rx - recon_weak

        z_s_after = rx_after_weak / ((h_strong * a_s) + eps)

        dec_bits_s = (np.real(z_s_after) >= 0).astype(int)

        dec_sym_s = (
            2 * dec_bits_s - 1
        ).astype(np.complex128)

        recon_strong = h_strong * (a_s * dec_sym_s)

        rx_after_strong = rx_after_weak - recon_strong

        ber_weak = float(np.mean(dec_bits_w != bits_w))
        ber_strong = float(np.mean(dec_bits_s != bits_s))

        return {
            "decoded_weak": dec_bits_w,
            "decoded_strong": dec_bits_s,
            "residual": rx_after_strong,
            "weak_cancelled_signal": rx_after_weak,
            "ber_weak": ber_weak,
            "ber_strong": ber_strong,
        }
