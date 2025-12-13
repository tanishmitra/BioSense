# models/patient.py
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Patient:
    patient_id: str
    channel_gain: float
    distance_m: Optional[float] = None

    def generate_bits(self, num_bits: int):
        """
        Generate random BPSK bits represented as -1 or +1 (map 0->-1, 1->+1)
        """
        bits = np.random.randint(0, 2, size=num_bits)
        symbols = (2 * bits - 1).astype(np.float32)  # use float32 for downstream math
        return bits, symbols

    def __repr__(self):
        return f"Patient(id={self.patient_id}, gain={self.channel_gain:.6g}, dist={self.distance_m})"


def make_patient(idx: int,
                 channel_gain: float = None,
                 distance_m: float = None,
                 min_distance: float = 2.0,
                 max_distance: float = 20.0,
                 pathloss_exponent: float = 3.0):
    """
    Factory to create a Patient.

    By default we use a distance-based pathloss model:
        channel_gain = 1 / (distance_m ** pathloss_exponent)

    Args:
        idx: integer index (0-based) -> patient_{idx+1}
        channel_gain: if provided, uses this value directly (overrides distance)
        distance_m: optional physical distance in meters (if provided, used directly)
        min_distance, max_distance: used to sample distance if distance_m not provided
        pathloss_exponent: pathloss exponent (alpha). Typical indoor values: 2.0-4.0.

    Returns:
        Patient instance with fields (patient_id, channel_gain, distance_m)
    """
    pid = f"patient_{idx+1}"

    if channel_gain is not None:
        # explicit override (keeps backward compatibility)
        return Patient(patient_id=pid, channel_gain=float(channel_gain), distance_m=distance_m)

    # sample or use provided distance
    if distance_m is None:
        distance_m = float(np.random.uniform(min_distance, max_distance))

    # simple power-law pathloss (linear amplitude gain model)
    # channel_gain = 1 / (distance ** alpha)
    alpha = float(pathloss_exponent)
    # avoid zero or negative distance
    distance_m = max(0.1, distance_m)
    channel_gain = float(1.0 / (distance_m ** alpha))

    return Patient(patient_id=pid, channel_gain=channel_gain, distance_m=distance_m)
