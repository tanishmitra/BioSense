import numpy as np


class WeakFirstSIC:

    def decode(self, rx, h_weak, h_strong,
               a_w, a_s,
               bits_w, bits_s):

        z_w = np.conj(h_weak) * rx

        dec_bits_w = (np.real(z_w) >= 0).astype(int)

        dec_sym_w = (
            2 * dec_bits_w - 1
        ).astype(np.complex128)

        recon_weak = h_weak * (a_w * dec_sym_w)

        rx_after_weak = rx - recon_weak

        z_s_after = np.conj(h_strong) * rx_after_weak

        dec_bits_s = (np.real(z_s_after) >= 0).astype(int)

        ber_weak = float(np.mean(dec_bits_w != bits_w))
        ber_strong = float(np.mean(dec_bits_s != bits_s))

        return {
            "decoded_weak": dec_bits_w,
            "decoded_strong": dec_bits_s,
            "residual": rx_after_weak,
            "ber_weak": ber_weak,
            "ber_strong": ber_strong,
        }