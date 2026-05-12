import numpy as np

class Patient:
    def __init__(self, patient_id, channel_gain, distance_m):
        self.patient_id = patient_id
        self.channel_gain = channel_gain
        self.distance_m = distance_m

    def generate_bits(self, num_bits):
        bits = np.random.randint(0, 2, size=num_bits)
        symbols = (2 * bits - 1).astype(np.float32)
        return bits, symbols