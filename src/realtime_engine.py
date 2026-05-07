"""
ThermoPath - Moteur de Prédiction Temps Réel
==============================================
Écoute le flux MQTT du capteur (réel ou simulé), accumule les données
dans un buffer glissant de 5 points, calcule les features temporelles,
puis utilise le modèle Scikit-Learn pour prédire les anomalies à la volée.
"""

import json
from collections import deque

import numpy as np
import pandas as pd
import joblib
import paho.mqtt.client as mqtt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ────────────────────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "thermopath/sensor"
BUFFER_SIZE = 5

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ── Chargement de l'IA ──────────────────────────────────────────────────────
print("🧠 Chargement du modèle et du scaler...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
print("✅ Modèle et scaler chargés avec succès !")

# ── Buffer glissant (mémoire des 5 dernières lectures) ──────────────────────
temp_buffer = deque(maxlen=BUFFER_SIZE)
gforce_buffer = deque(maxlen=BUFFER_SIZE)


# ── Callbacks MQTT ───────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    """Callback déclenché lors de la connexion au broker MQTT."""
    if reason_code == 0:
        print(f"✅ Connecté au broker MQTT ({BROKER_HOST}:{BROKER_PORT})")
        client.subscribe(TOPIC)
        print(f"👂 En écoute sur le topic : {TOPIC}")
        print("─" * 60)
    else:
        print(f"❌ Échec de connexion au broker (code : {reason_code})")


def on_message(client, userdata, msg):
    """
    Callback déclenché à la réception d'un message.
    Accumule les données, calcule les features, et prédit les anomalies.
    """
    # ── 1. Extraction des données brutes ─────────────────────────────────
    payload = json.loads(msg.payload.decode())
    temp = payload["temp"]
    g_force = payload["g_force"]

    # ── 2. Ajout aux buffers ─────────────────────────────────────────────
    temp_buffer.append(temp)
    gforce_buffer.append(g_force)

    # ── 3. Vérification : buffer plein ? ─────────────────────────────────
    current_len = len(temp_buffer)

    if current_len < BUFFER_SIZE:
        print(f"⏳ Buffering en cours ({current_len}/{BUFFER_SIZE})...")
        return

    # ── 4. Calcul des features temporelles ───────────────────────────────
    temp_array = np.array(temp_buffer)
    gforce_array = np.array(gforce_buffer)

    temp_mean = np.mean(temp_array)
    temp_std = np.std(temp_array)
    g_force_mean = np.mean(gforce_array)
    g_force_std = np.std(gforce_array)
    temp_velocity = temp_array[-1] - temp_array[0]

    # ── 5. Construction du DataFrame (ordre strict des colonnes) ─────────
    features = pd.DataFrame([{
        "thermal_shipper_temp_reading": temp,
        "g_force": g_force,
        "temp_mean": temp_mean,
        "temp_std": temp_std,
        "g_force_mean": g_force_mean,
        "g_force_std": g_force_std,
        "temp_velocity": temp_velocity,
    }])

    # ── 6. Scaling + Prédiction ──────────────────────────────────────────
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    # Re-mapping : Isolation Forest renvoie -1 pour anomalie, 1 pour normal
    is_anomaly = 1 if prediction == -1 else 0

    # ── 7. Affichage console ─────────────────────────────────────────────
    print("─" * 60)
    print(f"🌡️  Temp : {temp:.2f}°C  |  ⚡ G-Force : {g_force:.2f}")
    print(f"   📊 μ_temp={temp_mean:.2f}  σ_temp={temp_std:.2f}  "
          f"μ_gf={g_force_mean:.2f}  σ_gf={g_force_std:.2f}  "
          f"Δ_temp={temp_velocity:.2f}")

    if is_anomaly:
        print(f"   🔴 ALERTE — ANOMALIE DÉTECTÉE !")
    else:
        print(f"   🟢 Statut : NORMAL")
    print("─" * 60)


# ── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Instanciation du client MQTT avec l'API v2
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="thermopath-realtime-engine"
    )

    # Enregistrement des callbacks
    client.on_connect = on_connect
    client.on_message = on_message

    # Connexion au broker
    print(f"🔌 Connexion au broker MQTT ({BROKER_HOST}:{BROKER_PORT})...")
    client.connect(BROKER_HOST, BROKER_PORT)

    # Boucle d'écoute permanente (bloquante)
    client.loop_forever()
