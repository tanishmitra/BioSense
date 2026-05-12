class FallDetector:

    def detect_fall(self, analysis_result):

        if analysis_result["event_detected"]:
            return {
                "fall_detected": True,
                "severity": "HIGH"
            }

        return {
            "fall_detected": False,
            "severity": "NONE"
        }