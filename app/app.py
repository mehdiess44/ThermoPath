"""
ThermoPath - Dashboard de Surveillance Temps Réel
===================================================
Interface Streamlit avec Threading Sécurisé.
"""

import streamlit as st
import threading
import time
import json
from collections import deque

import pandas as pd
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import base64

def play_alert_sound(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration Streamlit ──────────────────────────────────────────────────
st.set_page_config(page_title="ThermoPath Live", page_icon="🌡️", layout="wide")

# ── Configuration MQTT & Modèle ──────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "thermopath/sensor"
BUFFER_SIZE = 5

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ── LA BOÎTE AUX LETTRES (Shared State) ──────────────────────────────────────
# @st.cache_resource garantit que cette mémoire est unique et partagée 
# entre le fantôme (Thread MQTT) et l'interface (Main Thread).
@st.cache_resource
def get_shared_state():
    return {
        "temp_buffer": deque(maxlen=BUFFER_SIZE),
        "gforce_buffer": deque(maxlen=BUFFER_SIZE),
        "temp_history": deque(maxlen=100),
        "gforce_history": deque(maxlen=100),
        "last_temp": None,
        "last_gforce": None,
        "last_status": None, # None = attente, 0 = normal, 1 = anomalie
        "message_count": 0
    }

shared_state = get_shared_state()

# ── Thread MQTT en arrière-plan ──────────────────────────────────────────────
@st.cache_resource
def start_mqtt_listener():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            client.subscribe(TOPIC)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            temp = payload["temp"]
            g_force = payload["g_force"]

            # Le Thread écrit DANS LA BOÎTE AUX LETTRES (pas dans session_state)
            shared_state["temp_buffer"].append(temp)
            shared_state["gforce_buffer"].append(g_force)
            shared_state["temp_history"].append(temp)
            shared_state["gforce_history"].append(g_force)
            shared_state["last_temp"] = temp
            shared_state["last_gforce"] = g_force
            shared_state["message_count"] += 1

            current_len = len(shared_state["temp_buffer"])

            if current_len < BUFFER_SIZE:
                shared_state["last_status"] = None
                return

            # ── Calcul ML ──────────────────────────────────────────────────
            temp_array = np.array(shared_state["temp_buffer"])
            gforce_array = np.array(shared_state["gforce_buffer"])

            features = pd.DataFrame([{
                "thermal_shipper_temp_reading": temp,
                "g_force": g_force,
                "temp_mean": np.mean(temp_array),
                "temp_std": np.std(temp_array),
                "g_force_mean": np.mean(gforce_array),
                "g_force_std": np.std(gforce_array),
                "temp_velocity": temp_array[-1] - temp_array[0],
            }])

            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)[0]

            shared_state["last_status"] = 1 if prediction == -1 else 0

        except Exception as e:
            print(f"Erreur ML : {e}")

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER_HOST, BROKER_PORT)

    thread = threading.Thread(target=client.loop_forever, daemon=True)
    thread.start()
    return thread

start_mqtt_listener()

# ── INTERFACE UTILISATEUR (Streamlit lit la boîte aux lettres) ───────────────

st.markdown(
    """
    <h1 style='text-align: center;'>🌡️ ThermoPath — Surveillance Temps Réel</h1>
    <p style='text-align: center; color: gray;'>Monitoring intelligent par IA de la logistique du froid.</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

status = shared_state["last_status"]
buffer_len = len(shared_state["temp_buffer"])

if status is None:
    st.warning(f"⏳ Buffering en cours ({buffer_len}/{BUFFER_SIZE})... En attente de données IA.")
elif status == 0:
    st.success("🟢 STATUT NORMAL — Système stable.")
elif status == 1:
    st.error("🔴 ALERTE — ANOMALIE DÉTECTÉE !")
    play_alert_sound(r"C:\Users\mehdi\ThermoPath\assets\alert.mp3")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #4b0000;
            animation: blinker 1s linear infinite;
        }
        @keyframes blinker {
            50% { opacity: 0.7; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
st.markdown("")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🌡️ Température (°C)", f"{shared_state['last_temp']:.2f}" if shared_state['last_temp'] else "—")
with col2:
    st.metric("⚡ G-Force", f"{shared_state['last_gforce']:.2f}" if shared_state['last_gforce'] else "—")
with col3:
    st.metric("📨 Messages", shared_state["message_count"])

st.markdown("---")

chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("📈 Historique Température")
    if len(shared_state["temp_history"]) > 0:
        st.line_chart(pd.DataFrame(list(shared_state["temp_history"]), columns=["Température"]))
    else:
        st.info("En attente...")

with chart_col2:
    st.subheader("📈 Historique G-Force")
    if len(shared_state["gforce_history"]) > 0:
        st.line_chart(pd.DataFrame(list(shared_state["gforce_history"]), columns=["G-Force"]))
    else:
        st.info("En attente...")

# Rafraîchissement automatique
time.sleep(1)
st.rerun()