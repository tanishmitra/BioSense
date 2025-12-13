# main.py
import os
import sys
import json
import time
import logging
import argparse
import paho.mqtt.client as mqtt

from models.edge_node import EdgeNode

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("main")


def load_config(cfg_path="config.json"):
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return json.load(f)


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT broker.")
        topic_prefix = userdata["config"]["mqtt"]["topic_prefix"]
        subscribe_topic = f"{topic_prefix}/#"
        client.subscribe(subscribe_topic)
        logger.info(f"Subscribed to topic: {subscribe_topic}")
    else:
        logger.error(f"Failed to connect to broker, rc={rc}")


def on_message(client, userdata, msg):
    payload = None
    try:
        payload = msg.payload.decode("utf-8")
    except Exception:
        payload = str(msg.payload)

    logger.info(f"MQTT message received on {msg.topic}: {payload}")

    # try JSON parse, otherwise treat payload as raw patient id
    patient_id = None
    try:
        data = json.loads(payload)
        # accept either {"patient_id": "patient_1"} or just a string
        if isinstance(data, dict) and "patient_id" in data:
            patient_id = data["patient_id"]
        elif isinstance(data, str):
            patient_id = data
    except Exception:
        # not JSON, maybe plain string patient id
        patient_id = payload.strip()

    # handle fall topic only (or any topic that contains '/falls' in its name)
    if "/falls" in msg.topic or "fall" in msg.topic.lower():
        # call edge node logic
        userdata["edge_node"].handle_fall_alert(patient_id)
    else:
        # routine message — we could log or ignore
        logger.debug("Routine message (ignored by main).")


def main(cfg_path="config.json"):
    config = load_config(cfg_path)

    edge = EdgeNode(config)
    edge.register_patients(config.get("simulation_params", {}).get("num_patients", 4))

    mqtt_cfg = config.get("mqtt", {})
    broker = mqtt_cfg.get("broker", "broker.hivemq.com")
    port = int(mqtt_cfg.get("port", 1883))
    client_id = mqtt_cfg.get("client_id", "edge_node_listener")

    client = mqtt.Client(client_id=client_id, userdata={"config": config, "edge_node": edge})
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(broker, port, keepalive=60)
    except Exception as e:
        logger.error(f"Could not connect to MQTT broker {broker}:{port} — {e}")
        return

    try:
        logger.info("Starting MQTT loop. Press Ctrl+C to exit.")
        client.loop_start()
        # Keep the script alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge node MQTT listener (sim).")
    parser.add_argument("--config", "-c", default="config.json", help="Path to config.json")
    args = parser.parse_args()
    main(args.config)
