import numpy as np
from core.patients.patient import Patient
from config.settings import MIN_DISTANCE, MAX_DISTANCE, PATHLOSS_EXPONENT


def make_patient(idx, distance_m=None):
    pid = f"patient_{idx+1}"

    if distance_m is None:
        distance_m = float(np.random.uniform(MIN_DISTANCE, MAX_DISTANCE))

    channel_gain = float(1.0 / (distance_m ** PATHLOSS_EXPONENT))

    return Patient(
        patient_id=pid,
        channel_gain=channel_gain,
        distance_m=distance_m,
    )