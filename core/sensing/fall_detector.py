from config.settings import (
    MIN_EVENT_SAMPLES,
    RADAR_EVENT_FRAME_THRESHOLD,
    RADAR_MIN_RANGE_MIGRATION_M,
    RADAR_TRACK_LOSS_FRAMES,
)


class FallDetector:

    def detect_fall(self, analysis_result, radar_result=None):

        if radar_result is not None:
            return self._detect_from_radar(radar_result)

        if not analysis_result["event_detected"]:
            return {
                "fall_detected": False,
                "severity": "NONE",
                "confidence": 0.0,
                "basis": "RESIDUAL_ENVELOPE",
            }

        spike_count = analysis_result["spike_count"]
        peak = analysis_result["peak"]
        threshold = analysis_result["threshold"]
        confidence = min(1.0, spike_count / max(MIN_EVENT_SAMPLES, 1))

        if spike_count >= MIN_EVENT_SAMPLES and peak >= 1.5 * threshold:
            severity = "HIGH"
        elif spike_count >= MIN_EVENT_SAMPLES:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        if spike_count >= MIN_EVENT_SAMPLES:
            return {
                "fall_detected": True,
                "severity": severity,
                "confidence": float(confidence),
                "basis": "RESIDUAL_ENVELOPE",
            }

        return {
            "fall_detected": False,
            "severity": "NONE",
            "confidence": 0.0,
            "basis": "RESIDUAL_ENVELOPE",
        }

    def _detect_from_radar(self, radar_result):
        event_frames = radar_result["event_frame_count"]
        max_snr_db = radar_result["max_snr_db"]
        migration_m = radar_result["range_migration_m"]
        motion_detected = radar_result["motion_detected"]
        track_stable = radar_result["track_stable"]
        track_lost = radar_result["track_lost"]
        loss_frames = radar_result["track_loss_frame_count"]
        baseline = radar_result["track_baseline_snr_db"]
        loss_threshold = radar_result["track_loss_threshold_db"]
        track_drop_db = max(baseline - loss_threshold, 0.0)

        frame_score = min(1.0, event_frames / max(RADAR_EVENT_FRAME_THRESHOLD + 1, 1))
        snr_score = min(1.0, max(max_snr_db - radar_result["threshold_snr_db"], 0.0) / 12.0)
        motion_score = min(1.0, migration_m / max(RADAR_MIN_RANGE_MIGRATION_M, 1e-12))
        loss_score = min(1.0, loss_frames / max(RADAR_TRACK_LOSS_FRAMES + 1, 1))
        drop_score = min(1.0, track_drop_db / 12.0)
        stability_score = 1.0 if track_stable else 0.0
        confidence = (
            0.35 * stability_score
            + 0.35 * loss_score
            + 0.20 * drop_score
            + 0.10 * motion_score
        )

        fall_detected = bool(track_stable and track_lost)

        if fall_detected and confidence >= 0.78:
            severity = "HIGH"
        elif fall_detected:
            severity = "MEDIUM"
        else:
            severity = "NONE"
            confidence = 0.0

        return {
            "fall_detected": bool(fall_detected),
            "severity": severity,
            "confidence": float(min(1.0, confidence)),
            "basis": "RADAR_TRACK_LOSS",
        }
