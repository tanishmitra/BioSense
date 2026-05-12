import json


def on_message(client, userdata, msg):

    try:
        payload = json.loads(msg.payload.decode())
    except:
        payload = msg.payload.decode()

    print("MQTT Message Received")
    print(payload)