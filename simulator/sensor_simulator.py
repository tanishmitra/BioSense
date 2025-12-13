# simulators/sensor_simulator.py
import os
import json
import time
import random
import argparse
import logging
import uuid
import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("simulator")


def load_config():
    # robust path: config.json at project root (one level up from simulators)
    pkg_dir = os.path.dirname(__file__)
    cfg_path = os.path.normpath(os.path.join(pkg_dir, "..", "config.json"))
    if not os.path.exists(cfg_path):
        # fallback to local config.json
        cfg_path = os.path.join(pkg_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError("Could not find config.json for simulator.")
    with open(cfg_path, "r") as f:
        return json.load(f)


def run_simulator(run_forever=True, count=100):
    cfg = load_config()
    mqtt_cfg = cfg.get("mqtt", {})
    sim_cfg = cfg.get("simulator_params", {})
    topic_prefix = mqtt_cfg.get("topic_prefix", "hospital/edge_sim")
    broker = mqtt_cfg.get("broker", "broker.hivemq.com")
    port = int(mqtt_cfg.get("port", 1883))

    client = mqtt.Client(client_id=f"simulator_{uuid.uuid4().hex[:6]}")
    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        logger.error(f"Failed to connect to MQTT broker {broker}:{port} -> {e}")
        return

    routine_topic = f"{topic_prefix}/routine"
    falls_topic = f"{topic_prefix}/falls"

    logger.info(f"Simulator publishing to: routine={routine_topic}, falls={falls_topic}")

    iterations = 0
    while True:
        # publish routine vitals message
        patient_id = f"patient_{random.randint(1, cfg.get('simulation_params', {}).get('num_patients', 4))}"
        vitals = {
            "patient_id": patient_id,
            "heart_rate": random.randint(55, 110),
            "spo2": random.randint(90, 100),
            "timestamp": int(time.time())
        }
        client.publish(routine_topic, json.dumps(vitals))
        logger.debug(f"Published routine for {patient_id}")

        # occasionally publish a fall alert (based on probability)
        # convert probability per minute to per-iteration probability using loop interval
        interval = sim_cfg.get("routine_interval_seconds", 1)
        fall_prob_per_min = sim_cfg.get("fall_probability_per_minute", 0.02)
        per_iter_prob = fall_prob_per_min * (interval / 60.0)
        if random.random() < per_iter_prob:
            alert_patient = patient_id
            alert_payload = {"patient_id": alert_patient, "event": "fall", "timestamp": int(time.time())}
            client.publish(falls_topic, json.dumps(alert_payload))
            logger.info(f"Published FALL alert for {alert_patient}")

        iterations += 1
        if not run_forever and iterations >= count:
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple sensor simulator that publishes routine vitals and occasional fall alerts.")
    parser.add_argument("--count", type=int, default=0, help="Number of iterations (0 = infinite).")
    args = parser.parse_args()
    run_forever = args.count <= 0
    run_simulator(run_forever=run_forever, count=max(0, args.count))
