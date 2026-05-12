import numpy as np

from config.settings import MONTE_CARLO_TRIALS, NUM_BITS
from simulation.run_complete_pipeline import run_pipeline


def run_monte_carlo(trials=MONTE_CARLO_TRIALS, num_bits=NUM_BITS):
    records = []

    for trial in range(trials):
        inject_fall = trial % 2 == 0
        result = run_pipeline(
            num_bits=num_bits,
            inject_fall=inject_fall,
            show_plots=False,
        )
        records.append({
            "expected_fall": inject_fall,
            "detected_fall": result["fall"]["fall_detected"],
            "ber_weak": result["sic"]["ber_weak"],
            "ber_strong": result["sic"]["ber_strong"],
            "severity": result["fall"]["severity"],
            "confidence": result["fall"]["confidence"],
            "spike_count": result["analysis"]["spike_count"],
        })

    positives = [record for record in records if record["expected_fall"]]
    negatives = [record for record in records if not record["expected_fall"]]

    true_positives = sum(
        record["detected_fall"] for record in positives
    )
    false_positives = sum(
        record["detected_fall"] for record in negatives
    )
    missed = len(positives) - true_positives

    return {
        "trials": int(trials),
        "ber_weak_mean": float(np.mean([r["ber_weak"] for r in records])),
        "ber_strong_mean": float(np.mean([r["ber_strong"] for r in records])),
        "detection_probability": float(true_positives / max(len(positives), 1)),
        "false_alarm_rate": float(false_positives / max(len(negatives), 1)),
        "missed_detection_rate": float(missed / max(len(positives), 1)),
        "records": records,
    }
