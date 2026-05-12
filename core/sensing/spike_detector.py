import numpy as np


class SpikeDetector:

    def detect(self, signal, k=4):

        mu = np.mean(signal)
        sigma = np.std(signal)

        threshold = mu + k * sigma

        spikes = np.where(signal > threshold)[0]

        return {
            "threshold": threshold,
            "spikes": spikes,
            "detected": len(spikes) > 0
        }