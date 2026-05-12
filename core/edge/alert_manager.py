class AlertManager:

    def generate_alert(self, patient_id, severity, confidence=0.0):

        return {
            "patient_id": patient_id,
            "alert": "FALL_DETECTED",
            "severity": severity,
            "confidence": confidence
        }
