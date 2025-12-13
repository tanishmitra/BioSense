# models/edge_node.py  (SIC-enabled, spike = actual SIC residual)
import os
import time
import logging
import json
import numpy as np
import matplotlib

# Defer backend selection until config; matplotlib imported for plt usage
import matplotlib.pyplot as plt
from datetime import datetime
from models.patient import make_patient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
    logger.addHandler(ch)

# Physical constants
C = 3e8  # speed of light (m/s)


class EdgeNode:
    def __init__(self, config: dict):
        self.config = config
        self.mqtt_cfg = config.get("mqtt", {})
        self.sim_cfg = config.get("simulation_params", {})
        self.patients = {}
        self.results_dir = self.sim_cfg.get("save_plots_dir", "results")
        os.makedirs(self.results_dir, exist_ok=True)

        # dedupe guard
        self._last_alert_ts = {}
        self._dedupe_window = 1.0

        # setup backend depending on plot_show/headless
        self._setup_backend()

    def _setup_backend(self):
        plot_show = bool(self.sim_cfg.get("plot_show", False))
        headless = bool(self.sim_cfg.get("headless", True))
        if plot_show and not headless:
            logger.info("Plot display enabled (interactive). Ensure you run in an environment with a display.")
        else:
            matplotlib.use("Agg")
            logger.info("Headless mode or plot_show disabled — using Agg backend (no interactive display).")

    def register_patients(self, num_patients: int = None):
        if num_patients is None:
            num_patients = int(self.sim_cfg.get("num_patients", 4))
        self.patients = {}
        for i in range(num_patients):
            p = make_patient(i)
            self.patients[p.patient_id] = p
        logger.info(f"Registered {len(self.patients)} patients: {list(self.patients.keys())}")

    def handle_fall_alert(self, alerting_patient_id: str):
        # dedupe
        now_ts = time.time()
        last = self._last_alert_ts.get(alerting_patient_id, 0.0)
        if now_ts - last < self._dedupe_window:
            logger.info(f"Ignoring duplicate alert for {alerting_patient_id} (within {self._dedupe_window}s)")
            return
        self._last_alert_ts[alerting_patient_id] = now_ts

        logger.info(f"Received fall alert for: {alerting_patient_id}")

        if alerting_patient_id not in self.patients:
            logger.warning(f"Unknown patient id {alerting_patient_id} — skipping")
            return

        partner_id = self._find_best_partner(alerting_patient_id)
        if partner_id is None:
            logger.warning("No partner found for simulation.")
            return

        # determine strong/weak (by channel_gain)
        strong_id, weak_id = self._determine_strong_weak(alerting_patient_id, partner_id)
        power_levels = self._allocate_power(strong_id, weak_id)
        logger.info(f"Power Allocation -> {strong_id} (Strong): {power_levels['strong']:.4f}, {weak_id} (Weak): {power_levels['weak']:.4f}")

        sim_params = self.sim_cfg
        self._run_simulation_for_pair(strong_id, weak_id, power_levels, sim_params)

    def _find_best_partner(self, alerting_patient_id: str):
        alert_gain = self.patients[alerting_patient_id].channel_gain
        best = None
        best_dist = float('inf')
        for pid, p in self.patients.items():
            if pid == alerting_patient_id:
                continue
            dist = abs(p.channel_gain - alert_gain)
            if dist < best_dist:
                best_dist = dist
                best = pid
        return best

    def _determine_strong_weak(self, a_id: str, b_id: str):
        if self.patients[a_id].channel_gain >= self.patients[b_id].channel_gain:
            return a_id, b_id
        return b_id, a_id

    def _allocate_power(self, strong_id: str, weak_id: str):
        g_s = self.patients[strong_id].channel_gain
        g_w = self.patients[weak_id].channel_gain
        eps = 1e-12
        inv = np.array([1.0 / (g_s + eps), 1.0 / (g_w + eps)])
        inv = inv / np.sum(inv)
        power = {"strong": float(inv[0]), "weak": float(inv[1])}
        return power

    def _run_simulation_for_pair(self, strong_id: str, weak_id: str, power_levels: dict, sim_params: dict):
        """
        Main comm simulation with proper SIC (weak-first).
        Also produces interactive spike plot that shows the residual after SIC
        (i.e., received signal after reconstructing & removing weak user).
        """
        # simulation params
        num_bits = int(sim_params.get("num_bits", 2048))
        sampling_rate = float(sim_params.get("sampling_rate", 20000.0))
        plot_show = bool(self.sim_cfg.get("plot_show", False))

        # -------- TRANSMIT (symbol-level) -----------
        bits_s, s_strong = self.patients[strong_id].generate_bits(num_bits)
        bits_w, s_weak = self.patients[weak_id].generate_bits(num_bits)

        a_s = np.sqrt(power_levels["strong"])
        a_w = np.sqrt(power_levels["weak"])

        tx_strong = a_s * s_strong
        tx_weak = a_w * s_weak
        tx_superposed = tx_strong + tx_weak

        # -------- CHANNEL (per-symbol Rayleigh) -----------
        # assume known channels (ideal CSI for demo) per-symbol
        h_strong = (np.random.randn(num_bits) + 1j * np.random.randn(num_bits)) / np.sqrt(2)
        h_weak = (np.random.randn(num_bits) + 1j * np.random.randn(num_bits)) / np.sqrt(2)

        noise_var = 0.01**2
        noise = np.sqrt(noise_var / 2) * (np.random.randn(num_bits) + 1j * np.random.randn(num_bits))

        # composite received (realistic uplink: each user contribution goes through its own channel)
        rx = h_strong * (a_s * s_strong) + h_weak * (a_w * s_weak) + noise

        # compute simple instantaneous SNR estimates (informational)
        noise_power = noise_var * 2  # complex
        sig_power_strong = np.mean(np.abs(h_strong * (a_s * s_strong))**2)
        sig_power_weak = np.mean(np.abs(h_weak * (a_w * s_weak))**2)
        snr_strong_db = 10 * np.log10(sig_power_strong / (noise_power + 1e-12))
        snr_weak_db = 10 * np.log10(sig_power_weak / (noise_power + 1e-12))
        logger.info(f"Estimated COMM SNR -> {strong_id}: {snr_strong_db:.2f} dB, {weak_id}: {snr_weak_db:.2f} dB")

        # -------------------------
        # ----- PROPER SIC -------
        # -------------------------
        # Uplink decode order: weak first (your chosen option)
        # Step 1: Equalize for weak user and make hard decisions (coherent detection)
        z_w = np.conj(h_weak) * rx  # matched filter for weak
        dec_bits_w = (np.real(z_w) >= 0).astype(int)
        dec_sym_w = (2 * dec_bits_w - 1).astype(np.complex128)  # reconstructed symbols (+1/-1)

        # Reconstruct weak user's contribution in the complex baseband (with amplitude and channel)
        recon_weak = h_weak * (a_w * dec_sym_w)

        # Residual after removing reconstructed weak user (this is what we will plot for the spike)
        rx_after_weak = rx - recon_weak

        # Now decode strong user from residual
        z_s_after = np.conj(h_strong) * rx_after_weak
        dec_bits_s = (np.real(z_s_after) >= 0).astype(int)
        dec_sym_s = (2 * dec_bits_s - 1).astype(np.complex128)

        # Compute BERs (compare to original bits)
        ber_weak = float(np.mean(dec_bits_w != bits_w))
        ber_strong = float(np.mean(dec_bits_s != bits_s))
        logger.info(f"BER -> {weak_id} (weak): {ber_weak:.6f}, {strong_id} (strong): {ber_strong:.6f}")

        # -------------------------
        # ----- SPIKE PLOT -------
        # -------------------------
        # Build a short time-series to visualize the event realistically:
        # We'll create a time-series of length T where the weak user transmits a short high-power burst
        # at center t0. We'll run that time-series through channels and perform SIC time-domain,
        # then plot the envelope of the residual after weak-subtraction.
        if plot_show:
            self._show_spike_plot_sic(a_s, a_w, power_levels, head_channels=(h_strong, h_weak), burst_ampl=3.0)

    def _show_spike_plot_sic(self, a_s, a_w, power_levels, head_channels=None, burst_ampl=3.0):
        """
        Construct a time-series, inject a burst from the weak user, run through channels,
        perform per-sample SIC (weak-first), and display the residual envelope.
        - a_s, a_w: amplitude scalars sqrt(power)
        - head_channels: not used for time-series (we re-draw per-run channels for clarity)
        - burst_ampl: additional amplitude multiply for the weak-user burst
        """
        rng = np.random.RandomState(int(time.time()) % 2**32)
        T = int(2000)
        # generate BPSK streams
        bits_s = rng.randint(0, 2, size=T)
        bits_w = rng.randint(0, 2, size=T)
        s_s = (2 * bits_s - 1).astype(np.complex128)
        s_w = (2 * bits_w - 1).astype(np.complex128)

        # simulate slow-fading channels (constant for the short series; can be randomized)
        h_s = (rng.randn() + 1j * rng.randn()) / np.sqrt(2)
        h_w = (rng.randn() + 1j * rng.randn()) / np.sqrt(2)

        # injected burst parameters
        t0 = T // 2
        burst_len = 12

        # create transmitted series with a burst for weak user
        tx_s = a_s * s_s
        tx_w = a_w * s_w

        # apply burst amplitude to weak user locally (emergency transmit)
        tx_w_burst = tx_w.copy()
        tx_w_burst[t0:t0 + burst_len] = tx_w_burst[t0:t0 + burst_len] * burst_ampl

        # received series
        noise_ts = 0.02 * (rng.randn(T) + 1j * rng.randn(T))
        rx_ts = h_s * tx_s + h_w * tx_w_burst + noise_ts

        # perform sample-wise SIC (weak-first): decode weak per-sample, reconstruct, subtract, then decode strong
        # For demonstration we use per-sample matched filtering and hard decisions (coherent detection)
        dec_bits_w_ts = np.zeros(T, dtype=int)
        dec_bits_s_ts = np.zeros(T, dtype=int)
        rx_residual_ts = np.zeros_like(rx_ts, dtype=np.complex128)

        for n in range(T):
            # weak detection on composite sample
            z_w_n = np.conj(h_w) * rx_ts[n]
            bit_w_hat = 1 if np.real(z_w_n) >= 0 else 0
            dec_bits_w_ts[n] = bit_w_hat
            sym_w_hat = 2 * bit_w_hat - 1
            recon_weak_n = h_w * (a_w * sym_w_hat)

            # subtract
            rx_after_n = rx_ts[n] - recon_weak_n
            rx_residual_ts[n] = rx_after_n

            # decode strong on residual
            z_s_n = np.conj(h_s) * rx_after_n
            bit_s_hat = 1 if np.real(z_s_n) >= 0 else 0
            dec_bits_s_ts[n] = bit_s_hat

        # envelope of residual (magnitude)
        env_resid = np.abs(rx_residual_ts)

        # plot window around burst
        left = max(0, t0 - 200)
        right = min(T, t0 + 400)
        plt.figure(figsize=(10, 3))
        plt.plot(np.arange(left, right), env_resid[left:right], linewidth=0.7)
        plt.axvline(t0, color='k', linestyle='--', linewidth=0.8)
        plt.text(t0 + 5, env_resid[left:right].max() * 0.9, 'Burst & SIC residual', va='center')
        plt.xlabel('Time (samples)')
        plt.ylabel('Residual envelope (|r_res|)')
        plt.title('SIC Residual Envelope — Weak-first SIC (burst from weak user)')
        plt.xlim(left, right)
        plt.tight_layout()
        logger.info("Displaying SIC-residual spike plot (interactive). Close the figure to continue.")
        plt.show()
        plt.close()

    # End of EdgeNode class
