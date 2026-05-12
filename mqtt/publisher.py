import json
import paho.mqtt.client as mqtt
from config.mqtt_config import BROKER, PORT


class Publisher:

    def __init__(self):
        self.client = mqtt.Client()
        self.client.connect(BROKER, PORT, 60)

    def publish(self, topic, payload):
        self.client.publish(topic, json.dumps(payload))