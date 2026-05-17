import numpy as np

from config.settings import (
    RADAR_CLUTTER_AMPLITUDE,
    RADAR_CLUTTER_COUNT,
    RADAR_FALL_DISPLACEMENT_M,
    RADAR_FALL_VELOCITY_MPS,
    RADAR_SAMPLE_DELAY_PER_METER,
    RADAR_TRACK_ATTENUATION_AFTER_FALL,
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


def generate_radar_echo(
    waveform,
    start,
    length,
    target_distance_m,
    amplitude=SENSING_ECHO_AMPLITUDE,
    radial_velocity_mps=RADAR_FALL_VELOCITY_MPS,
    fall_displacement_m=RADAR_FALL_DISPLACEMENT_M,
    track_loss=False,
    attenuation_after_fall=RADAR_TRACK_ATTENUATION_AFTER_FALL,
    sample_delay_per_meter=RADAR_SAMPLE_DELAY_PER_METER,
    phase_rad=SENSING_PHASE_RAD,
    clutter_count=RADAR_CLUTTER_COUNT,
    clutter_amplitude=RADAR_CLUTTER_AMPLITUDE,
):
    """Generate a radar-style reflected return from the shared ISAC waveform.

    The return is a delayed copy of the transmitted waveform. Static clutter is
    spread across several range bins, while a fall creates range migration and a
    changing phase history over the event window.
    """

    echo = np.zeros_like(waveform, dtype=np.complex128)
    waveform = np.asarray(waveform, dtype=np.complex128)

    _add_static_clutter(
        echo,
        waveform,
        target_distance_m,
        clutter_count,
        clutter_amplitude,
        sample_delay_per_meter,
    )

    _add_patient_track(
        echo,
        waveform,
        target_distance_m,
        start,
        amplitude,
        track_loss,
        attenuation_after_fall,
        sample_delay_per_meter,
        phase_rad,
    )

    event_start = max(0, int(start))
    event_end = min(len(waveform), event_start + int(length))
    if amplitude <= 0.0 or not track_loss or event_start >= event_end:
        return echo

    event_len = event_end - event_start
    base_delay = max(0, int(round(target_distance_m * sample_delay_per_meter)))
    displacement_samples = fall_displacement_m * sample_delay_per_meter
    migration = np.linspace(0.0, displacement_samples, event_len)
    velocity_phase = _velocity_phase(
        radial_velocity_mps,
        event_len,
        sample_delay_per_meter,
        phase_rad,
    )
    taper = np.hanning(max(event_len, 3))[:event_len]
    amplitude_profile = amplitude * (0.55 + taper)

    for idx in range(event_len):
        out_idx = event_start + idx
        delay = max(0, base_delay + int(round(migration[idx])))
        src_idx = out_idx - delay
        if 0 <= src_idx < len(waveform):
            echo[out_idx] += (
                0.35
                * amplitude_profile[idx]
                * waveform[src_idx]
                * np.exp(1j * velocity_phase[idx])
            )

    return echo


def _add_patient_track(
    echo,
    waveform,
    target_distance_m,
    fall_start,
    amplitude,
    track_loss,
    attenuation_after_fall,
    sample_delay_per_meter,
    phase_rad,
):
    if amplitude <= 0.0:
        return

    delay = max(0, int(round(target_distance_m * sample_delay_per_meter)))
    phase = np.exp(1j * phase_rad)
    gain = np.full(len(waveform), amplitude, dtype=float)

    if track_loss:
        fall_start = max(0, int(fall_start))
        gain[fall_start:] *= attenuation_after_fall

    if delay == 0:
        echo += gain * phase * waveform
    else:
        echo[delay:] += gain[delay:] * phase * waveform[:-delay]


def _add_static_clutter(
    echo,
    waveform,
    target_distance_m,
    clutter_count,
    clutter_amplitude,
    sample_delay_per_meter,
):
    if clutter_count <= 0 or clutter_amplitude <= 0.0:
        return

    clutter_ranges = np.linspace(
        max(1.0, target_distance_m * 0.35),
        target_distance_m * 1.35,
        clutter_count,
    )

    for idx, distance_m in enumerate(clutter_ranges):
        delay = max(0, int(round(distance_m * sample_delay_per_meter)))
        phase = np.exp(1j * 2 * np.pi * (idx + 1) / (clutter_count + 1))
        gain = clutter_amplitude / (1.0 + 0.35 * idx)
        if delay == 0:
            echo += gain * phase * waveform
        else:
            echo[delay:] += gain * phase * waveform[:-delay]


def _velocity_phase(radial_velocity_mps, sample_count, sample_delay_per_meter, phase_rad):
    if sample_count <= 0:
        return np.array([], dtype=float)

    n = np.arange(sample_count)
    phase_rate = 2 * np.pi * radial_velocity_mps * sample_delay_per_meter / sample_count
    return phase_rad + phase_rate * n
