from core.sic.weak_first_sic import WeakFirstSIC
from core.sensing.residual_analyzer import ResidualAnalyzer
from core.sensing.fall_detector import FallDetector
from core.sensing.radar_processor import RadarProcessor
from core.edge.alert_manager import AlertManager


class EdgeNode:

    def __init__(self):

        self.sic = WeakFirstSIC()
        self.analyzer = ResidualAnalyzer()
        self.radar = RadarProcessor()
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
                emit_alert=True,
                reference_waveform=None,
                expected_range_m=None):

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

        radar_analysis = None
        if reference_waveform is not None:
            radar_analysis = self.radar.analyze(
                sic_result["residual"],
                reference_waveform,
                expected_range_m=expected_range_m,
            )

        fall_result = self.fall_detector.detect_fall(
            analysis,
            radar_analysis,
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
            "radar": radar_analysis,
            "fall": fall_result,
            "alert": alert
        }
