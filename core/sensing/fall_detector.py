from config.settings import MIN_EVENT_SAMPLES


class FallDetector:

    def detect_fall(self, analysis_result):

        if not analysis_result["event_detected"]:
            return {
                "fall_detected": False,
                "severity": "NONE",
                "confidence": 0.0
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
                "confidence": float(confidence)
            }

        return {
            "fall_detected": False,
            "severity": "NONE",
            "confidence": 0.0
        }
