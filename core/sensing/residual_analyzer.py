from core.sic.residual_extractor import residual_envelope
from core.sensing.spike_detector import SpikeDetector


class ResidualAnalyzer:

    def __init__(self):
        self.detector = SpikeDetector()

    def analyze(self, residual):

        env = residual_envelope(residual)

        result = self.detector.detect(env)

        return {
            "envelope": env,
            "spikes": result["spikes"],
            "threshold": result["threshold"],
            "event_detected": result["detected"]
        }