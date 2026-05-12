import numpy as np

from core.patients.patient_factory import make_patient
from core.communication.power_allocator import allocate_power
from core.communication.noma import superpose_signals
from core.communication.channel import rayleigh_channel
from core.communication.awgn import generate_noise
from core.edge.edge_node import EdgeNode
from visualization.plot_residual import ResidualPlotter
from visualization.plot_metrics import plot_ber

from config.settings import NUM_BITS


strong_patient = make_patient(0, distance_m=5)
weak_patient = make_patient(1, distance_m=20)

bits_s, s_strong = strong_patient.generate_bits(NUM_BITS)
bits_w, s_weak = weak_patient.generate_bits(NUM_BITS)

power = allocate_power(
    strong_patient.channel_gain,
    weak_patient.channel_gain,
)

print(power)

_, tx_s, tx_w = superpose_signals(
    s_strong,
    s_weak,
    power,
)

h_s = rayleigh_channel(NUM_BITS)
h_w = rayleigh_channel(NUM_BITS)

noise = generate_noise(NUM_BITS)

burst_amp = 5

t0 = NUM_BITS // 2
burst_len = 12

# Inject burst for fall event

tx_w[t0:t0 + burst_len] *= burst_amp

rx = (
    h_s * tx_s +
    h_w * tx_w +
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
    weak_patient.patient_id
)

plotter = ResidualPlotter()

plotter.plot(
    result["analysis"]["envelope"],
    result["analysis"]["spikes"]
)

plot_ber(
    result["sic"]["ber_weak"],
    result["sic"]["ber_strong"]
)