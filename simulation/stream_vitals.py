import time
from datetime import datetime, timezone
import json

import numpy as np

from config.settings import (
    STREAM_FALL_PROBABILITY,
    STREAM_FRAME_BITS,
    STREAM_INTERVAL_SECONDS,
    STREAM_PATIENT_COUNT,
)
from config.mqtt_config import TOPIC_ALERTS, TOPIC_RESIDUAL, TOPIC_VITALS
from core.patients.patient_factory import make_patient
from simulation.run_complete_pipeline import run_pipeline


def initialize_stream_patients(count=STREAM_PATIENT_COUNT):
    distances = np.linspace(4.0, 24.0, count)
    return [
        make_patient(idx, distance_m=float(distance_m))
        for idx, distance_m in enumerate(distances)
    ]


def generate_vitals(patient, abnormal=False):
    heart_rate = np.random.normal(82 if abnormal else 74, 5)
    spo2 = np.random.normal(94 if abnormal else 98, 1.2)
    temperature_c = np.random.normal(37.4 if abnormal else 36.8, 0.25)
    respiration_rate = np.random.normal(22 if abnormal else 16, 2)

    return {
        "patient_id": patient.patient_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "distance_m": round(float(patient.distance_m), 2),
        "channel_gain": float(patient.channel_gain),
        "heart_rate_bpm": int(round(np.clip(heart_rate, 45, 140))),
        "spo2_percent": int(round(np.clip(spo2, 85, 100))),
        "temperature_c": round(float(np.clip(temperature_c, 35.0, 41.0)), 1),
        "respiration_rate_bpm": int(round(np.clip(respiration_rate, 8, 35))),
    }


def generate_fall_alert(patient_id, timestamp, fall_probability, sensing):
    confidence = float(sensing["confidence"])
    severity = sensing["severity"]

    return {
        "patient_id": patient_id,
        "alert": "FALL_DETECTED",
        "severity": severity,
        "confidence": round(confidence, 3),
        "timestamp": timestamp,
        "fall_probability": float(fall_probability),
        "reason": sensing["basis"],
    }


def generate_stream_frame(
    frame_id,
    num_bits=STREAM_FRAME_BITS,
    patients=None,
    fall_probability=STREAM_FALL_PROBABILITY,
):
    if patients is None:
        patients = initialize_stream_patients()
    if fall_probability is None:
        fall_probability = STREAM_FALL_PROBABILITY

    falling_patient_ids = {
        patient.patient_id
        for patient in patients
        if np.random.random() < fall_probability
    }

    vitals = [
        generate_vitals(
            patient,
            abnormal=patient.patient_id in falling_patient_ids,
        )
        for patient in patients
    ]

    result = run_pipeline(
        num_bits=num_bits,
        inject_fall=bool(falling_patient_ids),
        show_plots=False,
        emit_alert=False,
    )
    sensing = {
        "fall_detected": result["fall"]["fall_detected"],
        "severity": result["fall"]["severity"],
        "confidence": result["fall"]["confidence"],
        "basis": result["fall"]["basis"],
        "spike_count": result["analysis"]["spike_count"],
        "peak": result["analysis"]["peak"],
        "radar_event_frames": result["radar"]["event_frame_count"],
        "radar_max_snr_db": result["radar"]["max_snr_db"],
        "radar_range_migration_m": result["radar"]["range_migration_m"],
        "radar_estimated_velocity_mps": result["radar"]["estimated_velocity_mps"],
        "radar_track_stable": result["radar"]["track_stable"],
        "radar_track_lost": result["radar"]["track_lost"],
        "radar_track_range_m": result["radar"]["track_range_m"],
        "radar_track_loss_frames": result["radar"]["track_loss_frame_count"],
    }

    alerts = [
        generate_fall_alert(
            vital["patient_id"],
            vital["timestamp"],
            fall_probability,
            sensing,
        )
        for vital in vitals
        if (
            vital["patient_id"] in falling_patient_ids
            and sensing["fall_detected"]
        )
    ]

    return {
        "frame_id": int(frame_id),
        "patient_count": len(patients),
        "vitals": vitals,
        "fall_probability": float(fall_probability),
        "sic": {
            "ber_weak": result["sic"]["ber_weak"],
            "ber_strong": result["sic"]["ber_strong"],
        },
        "sensing": sensing,
        "alerts": alerts,
        "alert": alerts[0] if alerts else None,
    }


def stream_frames(
    interval_seconds=STREAM_INTERVAL_SECONDS,
    num_bits=STREAM_FRAME_BITS,
    max_frames=None,
    patient_count=STREAM_PATIENT_COUNT,
    fall_probability=STREAM_FALL_PROBABILITY,
):
    frame_id = 0
    patients = initialize_stream_patients(patient_count)

    while max_frames is None or frame_id < max_frames:
        frame = generate_stream_frame(
            frame_id,
            num_bits=num_bits,
            patients=patients,
            fall_probability=fall_probability,
        )
        yield frame
        frame_id += 1
        time.sleep(interval_seconds)


def publish_stream_frame(publisher, frame):
    for vital in frame["vitals"]:
        publisher.publish(TOPIC_VITALS, vital)

    publisher.publish(TOPIC_RESIDUAL, {
        "frame_id": frame["frame_id"],
        "sensing": frame["sensing"],
    })

    for alert in frame["alerts"]:
        publisher.publish(TOPIC_ALERTS, alert)


def run_vital_stream(
    max_frames=None,
    publish_mqtt=False,
    patient_count=STREAM_PATIENT_COUNT,
    fall_probability=STREAM_FALL_PROBABILITY,
):
    publisher = None

    if publish_mqtt:
        from mqtt.publisher import Publisher
        publisher = Publisher()

    patients = initialize_stream_patients(patient_count)
    print("Initialized patients:")
    for patient in patients:
        print(json.dumps({
            "patient_id": patient.patient_id,
            "distance_m": round(float(patient.distance_m), 2),
            "channel_gain": float(patient.channel_gain),
        }))

    frame_id = 0
    while max_frames is None or frame_id < max_frames:
        frame = generate_stream_frame(
            frame_id,
            patients=patients,
            fall_probability=fall_probability,
        )

        if publisher is not None:
            publish_stream_frame(publisher, frame)

        print(json.dumps(frame, indent=2))
        frame_id += 1
        time.sleep(STREAM_INTERVAL_SECONDS)


if __name__ == "__main__":
    run_vital_stream()
