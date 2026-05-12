import numpy as np

from config.settings import (
    SENSING_DELAY,
    SENSING_DOPPLER_HZ,
    SENSING_ECHO_AMPLITUDE,
    SENSING_PHASE_RAD,
)


def generate_fall_echo(
    waveform,
    start,
    length,
    amplitude=SENSING_ECHO_AMPLITUDE,
    doppler_hz=SENSING_DOPPLER_HZ,
    phase_rad=SENSING_PHASE_RAD,
    delay=SENSING_DELAY,
):
    echo = np.zeros_like(waveform, dtype=np.complex128)

    event_start = max(0, int(start) + int(delay))
    event_end = min(len(waveform), event_start + int(length))

    if event_start >= event_end:
        return echo

    n = np.arange(event_end - event_start)
    doppler = np.exp(1j * (2 * np.pi * doppler_hz * n + phase_rad))
    echo[event_start:event_end] = amplitude * waveform[event_start:event_end] * doppler

    return echo
