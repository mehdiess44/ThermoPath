"""
ThermoPath - Test Subscriber MQTT
==================================
Script d'écoute (Subscriber) qui se connecte au broker MQTT local
et affiche en temps réel les messages publiés sur le topic du capteur virtuel.
"""

import paho.mqtt.client as mqtt

# ── Configuration du Broker MQTT ─────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "thermopath/sensor"


# ── Callbacks ────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback déclenché lors de la connexion au broker MQTT."""
    if reason_code == 0:
        print("✅ Connecté au broker MQTT avec succès !")
        # Souscription au topic dès la connexion établie
        client.subscribe(TOPIC)
        print(f"👂 En écoute sur le topic : {TOPIC}")
    else:
        print(f"❌ Échec de connexion au broker (code : {reason_code})")


def on_message(client, userdata, msg):
    """Callback déclenché à la réception d'un message sur le topic souscrit."""
    payload = msg.payload.decode()
    print(f"📩 [{msg.topic}] {payload}")


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Instanciation du client MQTT avec l'API v2
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="thermopath-subscriber"
    )

    # Enregistrement des callbacks
    client.on_connect = on_connect
    client.on_message = on_message

    # Connexion au broker
    print(f"🔌 Connexion au broker MQTT ({BROKER_HOST}:{BROKER_PORT})...")
    client.connect(BROKER_HOST, BROKER_PORT)

    # Boucle d'écoute permanente (bloquante)
    client.loop_forever()
