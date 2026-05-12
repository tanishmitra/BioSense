from core.sic.residual_extractor import residual_envelope
from core.sensing.spike_detector import SpikeDetector
from config.settings import SENSING_WINDOW_SIZE, SENSING_WINDOW_STEP


class ResidualAnalyzer:

    def __init__(self):
        self.detector = SpikeDetector()

    def analyze(self, residual):

        env = residual_envelope(residual)

        result = self.detector.detect(env)
        energy = env ** 2
        window_energy = self._window_energy(
            energy,
            SENSING_WINDOW_SIZE,
            SENSING_WINDOW_STEP,
        )

        return {
            "envelope": env,
            "spikes": result["spikes"],
            "threshold": result["threshold"],
            "event_detected": result["detected"],
            "peak": float(env.max()) if len(env) else 0.0,
            "peak_energy": float(energy.max()) if len(energy) else 0.0,
            "total_energy": float(energy.sum()),
            "spike_count": int(len(result["spikes"])),
            "duration_samples": int(len(result["spikes"])),
            "window_energy": window_energy,
        }

    def _window_energy(self, energy, window_size, step):
        values = []

        if len(energy) == 0:
            return values

        for start in range(0, max(1, len(energy) - window_size + 1), step):
            end = min(len(energy), start + window_size)
            values.append({
                "start": int(start),
                "end": int(end),
                "energy": float(energy[start:end].sum()),
            })

        return values
