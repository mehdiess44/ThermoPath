"""
ThermoPath - Test Publisher MQTT
=================================
Capteur virtuel (Publisher) qui simule l'envoi de données
de température et de g-force sur le broker MQTT local.
"""

import json
import time

import paho.mqtt.client as mqtt

# ── Configuration du Broker MQTT ─────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "thermopath/sensor"

# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Instanciation du client MQTT avec l'API v2
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="thermopath-publisher"
    )

    # Connexion au broker
    print(f"🔌 Connexion au broker MQTT ({BROKER_HOST}:{BROKER_PORT})...")
    client.connect(BROKER_HOST, BROKER_PORT)
    print("✅ Connecté au broker MQTT avec succès !")

    # Température de base avec légère variation à chaque itération
    base_temp = -70.5

    for i in range(5):
        # Création du payload avec variation progressive de température
        payload = {
            "temp": round(base_temp + i * 0.3, 2),
            "g_force": round(1.02 + i * 0.01, 2)
        }

        # Sérialisation en JSON
        payload_json = json.dumps(payload)

        # Publication sur le topic
        client.publish(TOPIC, payload_json)
        print(f"📤 [{TOPIC}] {payload_json}")

        # Pause d'une seconde avant le prochain envoi
        time.sleep(1)

    # Déconnexion propre du client
    client.disconnect()
    print("🔌 Déconnexion du broker MQTT. Fin du script.")
