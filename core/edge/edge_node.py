from core.sic.weak_first_sic import WeakFirstSIC
from core.sensing.residual_analyzer import ResidualAnalyzer
from core.sensing.fall_detector import FallDetector
from core.edge.alert_manager import AlertManager


class EdgeNode:

    def __init__(self):

        self.sic = WeakFirstSIC()
        self.analyzer = ResidualAnalyzer()
        self.fall_detector = FallDetector()
        self.alert_manager = AlertManager()

    def process(self,
                rx,
                h_w,
                h_s,
                a_w,
                a_s,
                bits_w,
                bits_s,
                patient_id,
                emit_alert=True):

        sic_result = self.sic.decode(
            rx,
            h_w,
            h_s,
            a_w,
            a_s,
            bits_w,
            bits_s,
        )

        analysis = self.analyzer.analyze(
            sic_result["residual"]
        )

        fall_result = self.fall_detector.detect_fall(
            analysis
        )

        if fall_result["fall_detected"]:
            alert = self.alert_manager.generate_alert(
                patient_id,
                fall_result["severity"],
                fall_result["confidence"]
            )
            if emit_alert:
                print(alert)
        else:
            alert = None

        return {
            "sic": sic_result,
            "analysis": analysis,
            "fall": fall_result,
            "alert": alert
        }
