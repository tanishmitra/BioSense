import threading
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.monte_carlo import run_monte_carlo
from simulation.run_complete_pipeline import run_pipeline
from simulation.stream_vitals import (
    generate_stream_frame,
    initialize_stream_patients,
    publish_stream_frame,
)

app = Flask(__name__)
CORS(app)

ALERTS = []
LAST_RESULT = None
LAST_METRICS = None
LATEST_VITALS = None
LATEST_STREAM_FRAME = None
STREAM_THREAD = None
STREAM_STOP = threading.Event()
STREAM_LOCK = threading.Lock()


def _compact_result(result):
    radar = result["radar"] or {}

    return {
        "scenario": result["scenario"],
        "sic": {
            "ber_weak": result["sic"]["ber_weak"],
            "ber_strong": result["sic"]["ber_strong"],
        },
        "analysis": {
            "event_detected": result["analysis"]["event_detected"],
            "threshold": float(result["analysis"]["threshold"]),
            "peak": result["analysis"]["peak"],
            "peak_energy": result["analysis"]["peak_energy"],
            "total_energy": result["analysis"]["total_energy"],
            "spike_count": result["analysis"]["spike_count"],
            "duration_samples": result["analysis"]["duration_samples"],
            "window_energy": result["analysis"]["window_energy"],
            "spikes": result["analysis"]["spikes"].astype(int).tolist(),
        },
        "radar": {
            "event_detected": bool(radar.get("event_detected", False)),
            "event_frame_count": int(radar.get("event_frame_count", 0)),
            "max_snr_db": float(radar.get("max_snr_db", 0.0)),
            "range_migration_m": float(radar.get("range_migration_m", 0.0)),
            "estimated_velocity_mps": float(radar.get("estimated_velocity_mps", 0.0)),
            "phase_variation_rad": float(radar.get("phase_variation_rad", 0.0)),
            "motion_detected": bool(radar.get("motion_detected", False)),
            "track_stable": bool(radar.get("track_stable", False)),
            "track_lost": bool(radar.get("track_lost", False)),
            "track_range_m": float(radar.get("track_range_m", 0.0)),
            "track_baseline_snr_db": float(radar.get("track_baseline_snr_db", 0.0)),
            "track_loss_threshold_db": float(radar.get("track_loss_threshold_db", 0.0)),
            "track_loss_frame_count": int(radar.get("track_loss_frame_count", 0)),
            "threshold_snr_db": float(radar.get("threshold_snr_db", 0.0)),
            "event_frames": radar.get("event_frames", []).astype(int).tolist()
            if hasattr(radar.get("event_frames", []), "astype") else [],
            "track_loss_frames": radar.get("track_loss_frames", []).astype(int).tolist()
            if hasattr(radar.get("track_loss_frames", []), "astype") else [],
            "peak_ranges_m": radar.get("peak_ranges_m", []).astype(float).tolist()
            if hasattr(radar.get("peak_ranges_m", []), "astype") else [],
            "snr_db": radar.get("snr_db", []).astype(float).tolist()
            if hasattr(radar.get("snr_db", []), "astype") else [],
            "track_snr_db": radar.get("track_snr_db", []).astype(float).tolist()
            if hasattr(radar.get("track_snr_db", []), "astype") else [],
        },
        "fall": result["fall"],
        "alert": result["alert"],
    }


def _stream_loop(
    interval_seconds,
    num_bits,
    publish_mqtt=False,
    fall_probability=None,
):
    global LATEST_VITALS
    global LATEST_STREAM_FRAME

    frame_id = 0
    publisher = None

    if publish_mqtt:
        try:
            from mqtt.publisher import Publisher
            publisher = Publisher()
        except Exception:
            publisher = None

    patients = initialize_stream_patients()

    while not STREAM_STOP.is_set():
        frame = generate_stream_frame(
            frame_id,
            num_bits=num_bits,
            patients=patients,
            fall_probability=fall_probability,
        )

        with STREAM_LOCK:
            LATEST_STREAM_FRAME = frame
            LATEST_VITALS = frame["vitals"]
            ALERTS.extend(frame["alerts"])

        if publisher is not None:
            publish_stream_frame(publisher, frame)

        frame_id += 1
        STREAM_STOP.wait(interval_seconds)


@app.route("/health")
def health():
    return jsonify({"status": "running"})


@app.route("/simulate", methods=["POST", "GET"])
def simulate():
    global LAST_RESULT

    payload = request.get_json(silent=True) or {}
    inject_fall = bool(payload.get("inject_fall", True))
    num_bits = int(payload.get("num_bits", 1024))

    result = run_pipeline(
        num_bits=num_bits,
        inject_fall=inject_fall,
        show_plots=False,
    )
    compact = _compact_result(result)
    LAST_RESULT = compact

    if compact["alert"] is not None:
        ALERTS.append(compact["alert"])

    return jsonify(compact)


@app.route("/metrics", methods=["POST", "GET"])
def metrics():
    global LAST_METRICS

    payload = request.get_json(silent=True) or {}
    trials = int(payload.get("trials", 50))
    num_bits = int(payload.get("num_bits", 1024))

    LAST_METRICS = run_monte_carlo(
        trials=trials,
        num_bits=num_bits,
    )

    return jsonify(LAST_METRICS)


@app.route("/stream/start", methods=["POST"])
def start_stream():
    global STREAM_THREAD

    payload = request.get_json(silent=True) or {}
    interval_seconds = float(payload.get("interval_seconds", 1.0))
    num_bits = int(payload.get("num_bits", 256))
    publish_mqtt = bool(payload.get("publish_mqtt", False))
    fall_probability = payload.get("fall_probability")
    if fall_probability is not None:
        fall_probability = float(fall_probability)

    if STREAM_THREAD is not None and STREAM_THREAD.is_alive():
        return jsonify({"status": "already_running"})

    STREAM_STOP.clear()
    STREAM_THREAD = threading.Thread(
        target=_stream_loop,
        args=(interval_seconds, num_bits, publish_mqtt, fall_probability),
        daemon=True,
    )
    STREAM_THREAD.start()

    return jsonify({
        "status": "started",
        "interval_seconds": interval_seconds,
        "num_bits": num_bits,
        "publish_mqtt": publish_mqtt,
        "fall_probability": fall_probability,
    })


@app.route("/stream/stop", methods=["POST"])
def stop_stream():
    STREAM_STOP.set()

    return jsonify({"status": "stopping"})


@app.route("/stream/status")
def stream_status():
    running = STREAM_THREAD is not None and STREAM_THREAD.is_alive()

    with STREAM_LOCK:
        latest_frame = LATEST_STREAM_FRAME

    return jsonify({
        "running": running,
        "latest_frame_id": None if latest_frame is None else latest_frame["frame_id"],
        "latest_fall": None if latest_frame is None else latest_frame["sensing"],
        "latest_alert_count": 0 if latest_frame is None else len(latest_frame["alerts"]),
    })


@app.route("/vitals/latest")
def latest_vitals():
    with STREAM_LOCK:
        vitals = LATEST_VITALS

    if vitals is None:
        return jsonify({"error": "vital stream has not produced a frame"}), 404

    return jsonify({"vitals": vitals})


@app.route("/stream/latest")
def latest_stream_frame():
    with STREAM_LOCK:
        frame = LATEST_STREAM_FRAME

    if frame is None:
        return jsonify({"error": "vital stream has not produced a frame"}), 404

    return jsonify(frame)


@app.route("/alerts")
def alerts():
    return jsonify({"alerts": ALERTS})


@app.route("/residual/latest")
def latest_residual():
    if LAST_RESULT is None:
        return jsonify({"error": "no simulation has been run"}), 404

    return jsonify({
        "analysis": LAST_RESULT["analysis"],
        "radar": LAST_RESULT["radar"],
        "fall": LAST_RESULT["fall"],
    })


if __name__ == "__main__":
    app.run(debug=True)
