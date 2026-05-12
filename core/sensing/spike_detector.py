import numpy as np

from config.settings import THRESHOLD_K


class SpikeDetector:

    def detect(self, signal, k=THRESHOLD_K):

        mu = np.mean(signal)
        sigma = np.std(signal)

        threshold = mu + k * sigma

        spikes = np.where(signal > threshold)[0]

        return {
            "threshold": threshold,
            "spikes": spikes,
            "detected": len(spikes) > 0
        }
