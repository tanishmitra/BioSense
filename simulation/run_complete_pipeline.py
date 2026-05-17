import numpy as np

from config.settings import (
    BURST_LENGTH,
    NUM_BITS,
    RADAR_FALL_VELOCITY_MPS,
    SENSING_ECHO_AMPLITUDE,
)
from core.communication.awgn import generate_noise
from core.communication.channel import rayleigh_channel
from core.communication.noma import superpose_signals
from core.communication.power_allocator import allocate_power
from core.edge.edge_node import EdgeNode
from core.patients.patient_factory import make_patient
from core.sensing.echo_model import generate_radar_echo
from visualization.plot_metrics import plot_ber
from visualization.plot_radar import RadarPlotter
from visualization.plot_residual import ResidualPlotter


def run_pipeline(
    num_bits=NUM_BITS,
    inject_fall=True,
    show_plots=False,
    rng_seed=None,
    emit_alert=False,
    echo_amplitude=SENSING_ECHO_AMPLITUDE,
    radial_velocity_mps=RADAR_FALL_VELOCITY_MPS,
    save_radar_plot_path=None,
):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    strong_patient = make_patient(0, distance_m=5)
    weak_patient = make_patient(1, distance_m=20)

    bits_s, s_strong = strong_patient.generate_bits(num_bits)
    bits_w, s_weak = weak_patient.generate_bits(num_bits)

    power = allocate_power(
        strong_patient.channel_gain,
        weak_patient.channel_gain,
    )

    tx_superposed, tx_s, tx_w = superpose_signals(
        s_strong,
        s_weak,
        power,
    )

    h_s = rayleigh_channel(num_bits)
    h_w = rayleigh_channel(num_bits)
    noise = generate_noise(num_bits)

    event_start = num_bits // 2
    event_length = BURST_LENGTH if inject_fall else 0
    sensing_echo = generate_radar_echo(
        tx_superposed,
        event_start,
        event_length,
        target_distance_m=weak_patient.distance_m,
        amplitude=echo_amplitude,
        radial_velocity_mps=radial_velocity_mps if inject_fall else 0.0,
        track_loss=inject_fall,
    )

    rx = (
        h_s * tx_s +
        h_w * tx_w +
        sensing_echo +
        noise
    )

    edge = EdgeNode()

    result = edge.process(
        rx,
        h_w,
        h_s,
        np.sqrt(power["weak"]),
        np.sqrt(power["strong"]),
        bits_w,
        bits_s,
        weak_patient.patient_id,
        emit_alert=emit_alert,
        reference_waveform=tx_superposed,
        expected_range_m=weak_patient.distance_m,
    )

    result["scenario"] = {
        "num_bits": int(num_bits),
        "inject_fall": bool(inject_fall),
        "event_start": int(event_start),
        "event_length": int(event_length),
        "sensing_echo_amplitude": float(echo_amplitude if inject_fall else 0.0),
        "sensing_model": "matched_filter_radar",
        "radial_velocity_mps": float(radial_velocity_mps if inject_fall else 0.0),
        "target_distance_m": float(weak_patient.distance_m),
        "power": power,
        "strong_patient": {
            "patient_id": strong_patient.patient_id,
            "distance_m": strong_patient.distance_m,
            "channel_gain": strong_patient.channel_gain,
        },
        "weak_patient": {
            "patient_id": weak_patient.patient_id,
            "distance_m": weak_patient.distance_m,
            "channel_gain": weak_patient.channel_gain,
        },
    }

    if show_plots:
        plotter = ResidualPlotter()
        plotter.plot(
            result["analysis"]["envelope"],
            result["analysis"]["spikes"],
        )

        plot_ber(
            result["sic"]["ber_weak"],
            result["sic"]["ber_strong"],
        )

    if show_plots or save_radar_plot_path is not None:
        RadarPlotter().plot(
            result["radar"],
            result["fall"],
            save_path=save_radar_plot_path,
            show=show_plots,
        )

    return result


def run_demo():
    from simulation.stream_vitals import run_vital_stream

    run_vital_stream()


if __name__ == "__main__":
    run_demo()
