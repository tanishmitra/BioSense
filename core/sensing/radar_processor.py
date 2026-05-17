import numpy as np

from config.settings import (
    RADAR_EVENT_FRAME_THRESHOLD,
    RADAR_MAX_RANGE_M,
    RADAR_MIN_RANGE_MIGRATION_M,
    RADAR_SAMPLE_DELAY_PER_METER,
    RADAR_SNR_THRESHOLD_DB,
    RADAR_TRACK_LOSS_DROP_DB,
    RADAR_TRACK_LOSS_FRAMES,
    RADAR_TRACK_STABLE_FRAMES,
    RADAR_WINDOW_SIZE,
    RADAR_WINDOW_STEP,
)


class RadarProcessor:

    def analyze(
        self,
        received_signal,
        reference_waveform,
        max_range_m=RADAR_MAX_RANGE_M,
        sample_delay_per_meter=RADAR_SAMPLE_DELAY_PER_METER,
        window_size=RADAR_WINDOW_SIZE,
        step=RADAR_WINDOW_STEP,
        snr_threshold_db=RADAR_SNR_THRESHOLD_DB,
        expected_range_m=None,
    ):
        max_delay = max(1, int(round(max_range_m * sample_delay_per_meter)))
        profiles, starts = self._range_profiles(
            received_signal,
            reference_waveform,
            max_delay,
            window_size,
            step,
        )

        if profiles.size == 0:
            return self._empty_result()

        power = np.abs(profiles) ** 2
        noise_floor = np.median(power, axis=1) + 1e-12
        peak_bins = np.argmax(power, axis=1)
        peak_power = power[np.arange(len(power)), peak_bins]
        snr_db = 10.0 * np.log10(peak_power / noise_floor)
        event_frames = np.where(snr_db >= snr_threshold_db)[0]

        ranges_m = peak_bins / max(sample_delay_per_meter, 1e-12)
        event_ranges = ranges_m[event_frames]
        range_migration_m = (
            float(event_ranges.max() - event_ranges.min())
            if len(event_ranges) > 1 else 0.0
        )
        duration_samples = (
            int(starts[event_frames[-1]] - starts[event_frames[0]] + window_size)
            if len(event_frames) else 0
        )
        velocity_mps = self._estimate_velocity(
            event_ranges,
            duration_samples,
        )
        phase_stability = self._phase_variation(profiles, peak_bins, event_frames)
        track = self._track_loss_features(
            power,
            sample_delay_per_meter,
            snr_threshold_db,
            expected_range_m,
        )
        event_detected = (
            len(event_frames) >= RADAR_EVENT_FRAME_THRESHOLD
            and float(snr_db.max()) >= snr_threshold_db
        )

        return {
            "event_detected": bool(event_detected),
            "range_profiles": power,
            "window_starts": starts.astype(int),
            "peak_bins": peak_bins.astype(int),
            "peak_ranges_m": ranges_m,
            "snr_db": snr_db,
            "event_frames": event_frames.astype(int),
            "event_frame_count": int(len(event_frames)),
            "max_snr_db": float(snr_db.max()),
            "range_migration_m": range_migration_m,
            "estimated_velocity_mps": float(velocity_mps),
            "phase_variation_rad": float(phase_stability),
            "motion_detected": bool(range_migration_m >= RADAR_MIN_RANGE_MIGRATION_M),
            "track_bin": int(track["track_bin"]),
            "track_range_m": float(track["track_range_m"]),
            "track_snr_db": track["track_snr_db"],
            "track_baseline_snr_db": float(track["track_baseline_snr_db"]),
            "track_loss_threshold_db": float(track["track_loss_threshold_db"]),
            "track_loss_frames": track["track_loss_frames"],
            "track_loss_frame_count": int(len(track["track_loss_frames"])),
            "track_stable": bool(track["track_stable"]),
            "track_lost": bool(track["track_lost"]),
            "duration_samples": duration_samples,
            "threshold_snr_db": float(snr_threshold_db),
        }

    def _range_profiles(self, received_signal, reference_waveform, max_delay, window_size, step):
        received_signal = np.asarray(received_signal, dtype=np.complex128)
        reference_waveform = np.asarray(reference_waveform, dtype=np.complex128)
        usable = min(len(received_signal), len(reference_waveform))

        if usable < window_size + max_delay:
            window_size = max(8, usable - max_delay)
        if window_size <= 0:
            return np.empty((0, max_delay + 1), dtype=np.complex128), np.array([], dtype=int)

        starts = np.arange(0, usable - window_size - max_delay + 1, step, dtype=int)
        if len(starts) == 0:
            starts = np.array([0], dtype=int)

        profiles = []
        for start in starts:
            reference = reference_waveform[start:start + window_size]
            norm = np.linalg.norm(reference) + 1e-12
            bins = []
            for delay in range(max_delay + 1):
                segment_start = start + delay
                segment = received_signal[segment_start:segment_start + window_size]
                if len(segment) != len(reference):
                    bins.append(0j)
                else:
                    bins.append(np.vdot(reference, segment) / norm)
            profiles.append(bins)

        return np.asarray(profiles, dtype=np.complex128), starts

    def _estimate_velocity(self, event_ranges, duration_samples):
        if len(event_ranges) < 2 or duration_samples <= 0:
            return 0.0

        return (event_ranges[-1] - event_ranges[0]) / duration_samples

    def _phase_variation(self, profiles, peak_bins, event_frames):
        if len(event_frames) < 2:
            return 0.0

        phases = np.unwrap([
            np.angle(profiles[frame_idx, peak_bins[frame_idx]])
            for frame_idx in event_frames
        ])
        return float(np.std(np.diff(phases))) if len(phases) > 2 else 0.0

    def _track_loss_features(
        self,
        power,
        sample_delay_per_meter,
        snr_threshold_db,
        expected_range_m,
    ):
        if power.size == 0:
            return {
                "track_bin": 0,
                "track_range_m": 0.0,
                "track_snr_db": np.array([], dtype=float),
                "track_baseline_snr_db": 0.0,
                "track_loss_threshold_db": 0.0,
                "track_loss_frames": np.array([], dtype=int),
                "track_stable": False,
                "track_lost": False,
            }

        if expected_range_m is None:
            stable_count = min(RADAR_TRACK_STABLE_FRAMES, len(power))
            track_bin = int(np.argmax(np.median(power[:stable_count], axis=0)))
        else:
            track_bin = int(round(expected_range_m * sample_delay_per_meter))
            track_bin = int(np.clip(track_bin, 0, power.shape[1] - 1))

        noise_floor = np.median(power, axis=1) + 1e-12
        track_power = power[:, track_bin]
        track_snr_db = 10.0 * np.log10((track_power + 1e-12) / noise_floor)

        stable_count = min(RADAR_TRACK_STABLE_FRAMES, len(track_snr_db))
        baseline = float(np.median(track_snr_db[:stable_count])) if stable_count else 0.0
        loss_threshold = max(
            snr_threshold_db,
            baseline - RADAR_TRACK_LOSS_DROP_DB,
        )
        stable = stable_count >= 2 and baseline >= snr_threshold_db
        candidate_losses = np.where(track_snr_db < loss_threshold)[0]
        loss_frames = candidate_losses[candidate_losses >= stable_count]
        lost = stable and len(loss_frames) >= RADAR_TRACK_LOSS_FRAMES

        return {
            "track_bin": track_bin,
            "track_range_m": track_bin / max(sample_delay_per_meter, 1e-12),
            "track_snr_db": track_snr_db,
            "track_baseline_snr_db": baseline,
            "track_loss_threshold_db": loss_threshold,
            "track_loss_frames": loss_frames.astype(int),
            "track_stable": stable,
            "track_lost": lost,
        }

    def _empty_result(self):
        return {
            "event_detected": False,
            "range_profiles": np.empty((0, 0)),
            "window_starts": np.array([], dtype=int),
            "peak_bins": np.array([], dtype=int),
            "peak_ranges_m": np.array([], dtype=float),
            "snr_db": np.array([], dtype=float),
            "event_frames": np.array([], dtype=int),
            "event_frame_count": 0,
            "max_snr_db": 0.0,
            "range_migration_m": 0.0,
            "estimated_velocity_mps": 0.0,
            "phase_variation_rad": 0.0,
            "motion_detected": False,
            "track_bin": 0,
            "track_range_m": 0.0,
            "track_snr_db": np.array([], dtype=float),
            "track_baseline_snr_db": 0.0,
            "track_loss_threshold_db": 0.0,
            "track_loss_frames": np.array([], dtype=int),
            "track_loss_frame_count": 0,
            "track_stable": False,
            "track_lost": False,
            "duration_samples": 0,
            "threshold_snr_db": float(RADAR_SNR_THRESHOLD_DB),
        }
