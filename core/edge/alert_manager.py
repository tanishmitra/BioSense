class AlertManager:

    def generate_alert(self, patient_id, severity):

        return {
            "patient_id": patient_id,
            "alert": "FALL_DETECTED",
            "severity": severity
        }