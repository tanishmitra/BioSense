import paho.mqtt.client as mqtt
from config.mqtt_config import BROKER, PORT


class Subscriber:

    def __init__(self, on_message):

        self.client = mqtt.Client()
        self.client.on_message = on_message

    def start(self, topic):

        self.client.connect(BROKER, PORT, 60)
        self.client.subscribe(topic)
        self.client.loop_forever()