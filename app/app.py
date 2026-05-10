"""
ThermoPath - Dashboard de Surveillance Temps Réel (Version UI/UX Driver)
========================================================================
Interface Streamlit optimisée pour la cabine d'un chauffeur.
"""

import streamlit as st
import threading
import time
import json
from collections import deque
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import base64
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration Streamlit ──────────────────────────────────────────────────
st.set_page_config(page_title="ThermoPath Live", page_icon="🚛", layout="wide", initial_sidebar_state="collapsed")

# ── Fonction Audio ───────────────────────────────────────────────────────────
def play_alert_sound(file_path):
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    except FileNotFoundError:
        pass # Ignore silencieusement si le fichier son n'est pas trouvé

# ── Configuration MQTT & Modèle ──────────────────────────────────────────────
BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "thermopath/sensor"
BUFFER_SIZE = 5

MODEL_PATH = "models/model.pkl"
SCALER_PATH = "models/scaler.pkl"

# ── LA BOÎTE AUX LETTRES (Shared State) ──────────────────────────────────────
@st.cache_resource
def get_shared_state():
    return {
        "temp_buffer": deque(maxlen=BUFFER_SIZE),
        "gforce_buffer": deque(maxlen=BUFFER_SIZE),
        "temp_history": deque(maxlen=100),
        "gforce_history": deque(maxlen=100),
        "last_temp": None,
        "last_gforce": None,
        "last_status": None, 
        "message_count": 0,
        "last_update": "En attente..."
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

            shared_state["temp_buffer"].append(temp)
            shared_state["gforce_buffer"].append(g_force)
            shared_state["temp_history"].append(temp)
            shared_state["gforce_history"].append(g_force)
            shared_state["last_temp"] = temp
            shared_state["last_gforce"] = g_force
            shared_state["message_count"] += 1
            shared_state["last_update"] = datetime.now().strftime("%H:%M:%S")

            current_len = len(shared_state["temp_buffer"])

            if current_len < BUFFER_SIZE:
                shared_state["last_status"] = None
                return

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

# ── INTERFACE UTILISATEUR (Streamlit UI/UX) ──────────────────────────────────

# CSS Global pour le style industriel
st.markdown("""
    <style>
    /* Style des grandes cartes de métriques */
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 2px solid #333;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-title {
        color: #888;
        font-size: 1.5rem;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .metric-value {
        color: #FFFFFF;
        font-size: 4.5rem;
        font-weight: 900;
        margin: 0;
        line-height: 1;
    }
    .metric-unit {
        font-size: 2rem;
        color: #AAA;
    }
    /* Mode Alerte (Clignotement Rouge) */
    .alert-mode {
        background-color: #3b0000 !important;
        animation: blinker 1.5s linear infinite;
    }
    @keyframes blinker {
        50% { opacity: 0.8; background-color: #660000 !important; }
    }
    /* Instructions en cas d'alerte */
    .alert-instruction {
        background-color: #FFDDDD;
        color: #900;
        padding: 20px;
        border-radius: 10px;
        font-size: 1.8rem;
        font-weight: bold;
        text-align: center;
        border-left: 10px solid #D00;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

status = shared_state["last_status"]
buffer_len = len(shared_state["temp_buffer"])
temp_val = shared_state["last_temp"]
gforce_val = shared_state["last_gforce"]

# Déclenchement de l'alerte sonore et visuelle
if status == 1:
    play_alert_sound(r"C:\Users\mehdi\ThermoPath\assets\alert.mp3")
    st.markdown('<script>parent.document.body.classList.add("alert-mode");</script>', unsafe_allow_html=True)
    st.markdown('<style>.stApp { background-color: #3b0000; animation: blinker 1s linear infinite; }</style>', unsafe_allow_html=True)
else:
    st.markdown('<script>parent.document.body.classList.remove("alert-mode");</script>', unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────────────────────
col_logo, col_time = st.columns([3, 1])
with col_logo:
    st.markdown("<h2 style='margin:0; color:#00BFFF;'>🚛 ThermoPath - Terminal Cabine</h2>", unsafe_allow_html=True)
with col_time:
    st.markdown(f"<h4 style='text-align:right; color:#888; margin:0;'>Dernière synchro: {shared_state['last_update']}</h4>", unsafe_allow_html=True)

st.markdown("<hr style='margin-top:5px; margin-bottom:20px;'>", unsafe_allow_html=True)

# ── BANDEAU STATUT ───────────────────────────────────────────────────────────
if status is None:
    st.info(f"🔄 Initialisation des capteurs... ({buffer_len}/{BUFFER_SIZE} s)")
elif status == 0:
    st.success("✅ SYSTÈME OPÉRATIONNEL - CARGAISON SÉCURISÉE")
elif status == 1:
    st.error("⚠️ ALERTE RUPTURE CHAÎNE DU FROID OU CHOC ⚠️")

# ── CARTES DE MÉTRIQUES GÉANTES ──────────────────────────────────────────────
col1, col2 = st.columns(2)

# Formatage des valeurs
temp_display = f"{temp_val:.1f}" if temp_val is not None else "--"
gforce_display = f"{gforce_val:.2f}" if gforce_val is not None else "--"

# Couleur dynamique pour la température (Rouge si chaud, Bleu si froid)
temp_color = "#FF4444" if (temp_val is not None and temp_val > -60) else "#00BFFF"
# Couleur dynamique pour la force G (Rouge si gros choc)
gforce_color = "#FF4444" if (gforce_val is not None and gforce_val > 2.0) else "#00FF00"

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🌡️ TEMPÉRATURE INTERNE</div>
            <p class="metric-value" style="color: {temp_color};">{temp_display}<span class="metric-unit">°C</span></p>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ STABILITÉ (G-FORCE)</div>
            <p class="metric-value" style="color: {gforce_color};">{gforce_display}<span class="metric-unit"> G</span></p>
        </div>
    """, unsafe_allow_html=True)

# ── INSTRUCTIONS CHAUFFEUR (Seulement si Alerte) ─────────────────────────────
if status == 1:
    st.markdown("""
        <div class="alert-instruction">
            🛑 ACTIONS REQUISES IMMÉDIATEMENT :<br>
            1. Garez-vous en sécurité dès que possible.<br>
            2. Contrôlez l'intégrité de la palette et le groupe frigorifique.<br>
            3. Contactez le superviseur logistique.
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── GRAPHIQUES HISTORIQUES (Secondaires pour le chauffeur) ───────────────────
with st.expander("📊 Afficher l'historique détaillé (Supervision)", expanded=(status==1)):
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        if len(shared_state["temp_history"]) > 0:
            st.line_chart(pd.DataFrame(list(shared_state["temp_history"]), columns=["Température (°C)"]), height=200)
    with chart_col2:
        if len(shared_state["gforce_history"]) > 0:
            st.line_chart(pd.DataFrame(list(shared_state["gforce_history"]), columns=["G-Force"]), height=200)

# ── Boucle de rafraîchissement automatique ───────────────────────────────────
time.sleep(1)
st.rerun()