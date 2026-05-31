"""
ThermoPath - Dashboard de Surveillance Temps Réel (Version Industrialisée)
========================================================================
Application Streamlit de jumeau numérique IoT pour la chaîne du froid.
- Wildcard 1 : Résilience Réseau Absolue (Auto-reconnect & Health Check)
- Wildcard 2 : Explicabilité IA (Score de Risque Thermique via decision_function)
- UX Chauffeur : Cartes géantes, jauge de risque, flash d'alerte, son
"""

import streamlit as st
import threading
import time
import json
import base64
from collections import deque
from datetime import datetime
from threading import Lock
import pandas as pd
import numpy as np
import joblib
import paho.mqtt.client as mqtt
import warnings
import os

# ── Suppression des avertissements Scikit-Learn ──────────────────────────────
warnings.filterwarnings("ignore")

# ── Configuration Streamlit ──────────────────────────────────────────────────
st.set_page_config(
    page_title="ThermoPath Live",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Configuration MQTT & Modèle ─────────────────────────────────────────────
# Variables d'environnement pour Docker, localhost par défaut pour les tests
BROKER_HOST = os.getenv("BROKER_HOST", "localhost")
BROKER_PORT = int(os.getenv("BROKER_PORT", "1883"))
TOPIC = "thermopath/sensor"
BUFFER_SIZE = 5  # Taille du tampon glissant pour le calcul des features IA

# Utilisation de chemins absolus pour éviter les problèmes de répertoire de travail (CWD)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ALERT_SOUND_PATH = os.path.join(BASE_DIR, "assets", "alert.mp3")


# ── Fonction utilitaire : Explicabilité IA (XAI) ─────────────────────────────
def generate_xai_explanation(current_temp: float, current_gforce: float, temp_velocity: float) -> str:
    """
    Génère une explication heuristique pour l'alerte d'anomalie.
    Priorité 1: Choc pur (gforce > 3.0)
    Priorité 2: Fuite/Panne thermique (vélocité > 2.0)
    Priorité 3: Dérive lente (température > -65.0)
    Défaut: Instabilité complexe
    """
    if current_gforce is not None and current_gforce > 3.0:
        return "Impact mécanique détecté. Vérifiez l'intégrité de la palette."
    elif temp_velocity is not None and temp_velocity > 2.0:
        return "Hausse thermique anormale rapide. Vérifiez l'alimentation du groupe frigorifique."
    elif current_temp is not None and current_temp > -65.0:
        return "Température critique atteinte par dérive. Vérifiez l'étanchéité des portes."
    else:
        return "Instabilité thermodynamique complexe détectée par l'IA. Contrôle visuel recommandé."


import streamlit.components.v1 as components

# ── Fonction utilitaire : Alerte sonore ──────────────────────────────────────
def play_alert_sound(file_path: str):
    """
    Joue un fichier audio d'alerte en arrière-plan sans afficher de lecteur.
    Utilise components.html pour injecter un iframe invisible qui gère l'autoplay.
    """
    if not os.path.isfile(file_path):
        return

    # Identifiant unique de l'alerte pour s'assurer que l'iframe
    # n'est rechargé qu'une seule fois par nouvelle alerte.
    alert_id = shared_state.get("alert_until", 0)
    
    try:
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
            
            # Injection via components.html qui crée un iframe (invisible avec width/height 0)
            # L'ID d'alerte dans le commentaire garantit que Streamlit recharge l'iframe
            # si et seulement si c'est une nouvelle alerte.
            html_string = f"""
            <!-- Alert ID: {alert_id} -->
            <audio id="audio" autoplay="autoplay">
                <source src="data:audio/mp3;base64,{b64_audio}" type="audio/mp3">
            </audio>
            <script>
                var audio = document.getElementById("audio");
                audio.play().catch(function(e) {{
                    console.log('Autoplay blocked:', e);
                }});
            </script>
            """
            components.html(html_string, width=0, height=0)
    except Exception as e:
        print(f"[ERREUR] Impossible de jouer le son : {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. ÉTAT PARTAGÉ (Boîte aux lettres inter-threads)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_shared_state() -> dict:
    """
    Initialise le dictionnaire global partagé entre le thread MQTT et Streamlit.
    - deque(maxlen=5) : tampon glissant pour le calcul des 7 features IA.
    - deque(maxlen=100) : historique pour les graphiques temps réel.
    """
    return {
        # Tampons glissants pour le calcul IA (fenêtre de 5 points)
        "temp_buffer": deque(maxlen=BUFFER_SIZE),
        "gforce_buffer": deque(maxlen=BUFFER_SIZE),
        # Historiques pour les graphiques (100 derniers points)
        "temp_history": deque(maxlen=100),
        "gforce_history": deque(maxlen=100),
        # Dernières valeurs instantanées
        "last_temp": None,
        "last_gforce": None,
        # Résultat de l'IA : 0 = normal, 1 = anomalie, None = pas encore assez de données
        "last_status": None,
        # Wildcard 2 : Indice de Risque Thermique (0 à 100)
        "risk_score": 0,
        # Latch d'alerte XAI (10 secondes)
        "alert_until": 0,
        "latched_xai_message": "",
        # Compteur de messages reçus
        "message_count": 0,
        # Horodatage de la dernière mise à jour
        "last_update": "En attente...",
        # Wildcard 1 : État de connexion MQTT
        "mqtt_connected": False,
        # Wildcard 3 : Verrou de sécurité critique pour éviter les écritures concurrentes (Race Conditions)
        "lock": Lock()
    }


shared_state = get_shared_state()


# ══════════════════════════════════════════════════════════════════════════════
# 2. THREAD MQTT (Résilience Réseau + IA en temps réel)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def start_mqtt_listener():
    """
    Démarre le client MQTT dans un thread daemon séparé.
    - Charge le modèle Isolation Forest et le scaler.
    - Implémente on_connect / on_disconnect pour la résilience réseau.
    - Implémente on_message pour le calcul IA en temps réel.
    - Gère l'échec de connexion initiale avec un thread de reconnexion.
    """
    # Chargement du modèle et du scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # ── Callback : Connexion établie ─────────────────────────────────────
    def on_connect(client, userdata, flags, reason_code, properties):
        """Appelé quand la connexion MQTT est établie."""
        if reason_code == 0:
            shared_state["mqtt_connected"] = True
            client.subscribe(TOPIC)
            print(f"[OK] Connecte au broker MQTT ({BROKER_HOST}:{BROKER_PORT})")
        else:
            shared_state["mqtt_connected"] = False
            print(f"[ERREUR] Echec de connexion MQTT, code : {reason_code}")

    # ── Callback : Connexion perdue ──────────────────────────────────────
    def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
        """Appelé quand la connexion MQTT est perdue."""
        shared_state["mqtt_connected"] = False
        print("[WARN] Perte de connexion MQTT. Tentative de reconnexion en cours...")

    # ── Callback : Message reçu ──────────────────────────────────────────
    def on_message(client, userdata, msg):
        """
        Traite chaque message MQTT reçu :
        1. Décode le payload JSON (temp, g_force).
        2. Alimente les tampons et historiques (sous Mutex).
        3. Calcule les 7 features de l'IA (incluant la vélocité corrigée).
        4. Applique le scaler puis le modèle Isolation Forest.
        5. Utilise decision_function() pour obtenir le score brut.
        6. Transforme le score en Indice de Risque Thermique (0-100%).
        """
        try:
            payload = json.loads(msg.payload.decode())
            temp = float(payload["temp"])
            g_force = float(payload["g_force"])

            # Verrouillage complet du bloc d'écriture et d'inférence pour la Thread-Safety
            with shared_state["lock"]:
                # Mise à jour des tampons glissants et historiques
                shared_state["temp_buffer"].append(temp)
                shared_state["gforce_buffer"].append(g_force)
                shared_state["temp_history"].append(temp)
                shared_state["gforce_history"].append(g_force)
                shared_state["last_temp"] = temp
                shared_state["last_gforce"] = g_force
                shared_state["message_count"] += 1
                shared_state["last_update"] = datetime.now().strftime("%H:%M:%S")

                # On attend d'avoir assez de données pour le calcul IA
                if len(shared_state["temp_buffer"]) < BUFFER_SIZE:
                    shared_state["last_status"] = None
                    return

                # Calcul des arrays
                temp_array = np.array(shared_state["temp_buffer"])
                gforce_array = np.array(shared_state["gforce_buffer"])
                
                # ── Correction MLOps : Vélocité calculée strictement sur 1 période
                current_velocity = temp_array[-1] - temp_array[-2]

                features = pd.DataFrame(
                    [
                        {
                            "thermal_shipper_temp_reading": temp,
                            "g_force": g_force,
                            "temp_mean": np.mean(temp_array),
                            "temp_std": np.std(temp_array),
                            "g_force_mean": np.mean(gforce_array),
                            "g_force_std": np.std(gforce_array),
                            "temp_velocity": current_velocity,
                        }
                    ]
                )

                # Normalisation avec le scaler entraîné
                features_scaled = scaler.transform(features)

                # ── WILDCARD 2 : Explicabilité Mathématique ──────────────────
                raw_score = model.decision_function(features_scaled)[0]
                prediction = model.predict(features_scaled)[0]

                # Transformation du score brut en Indice de Risque (0% à 100%)
                risk_index = 50 - (raw_score * 200)
                shared_state["risk_score"] = int(np.clip(risk_index, 0, 100))

                # Statut binaire et gestion du verrouillage de l'alerte XAI (Latch de 10s)
                if prediction == -1:
                    shared_state["last_status"] = 1
                    # Calcul immédiat de la vélocité thermique
                    current_velocity = temp_array[-1] - temp_array[-2]
                    new_explanation = generate_xai_explanation(temp, g_force, current_velocity)
                    
                    # CORRECTION DU "GHOST EFFECT"
                    default_msg = "Instabilité thermodynamique complexe détectée par l'IA. Contrôle visuel recommandé."
                    
                    # Mise à jour du message SEULEMENT si c'est le début d'une nouvelle alerte 
                    # OU si l'explication est spécifique (et non le message par défaut).
                    if time.time() >= shared_state["alert_until"] or new_explanation != default_msg:
                        shared_state["latched_xai_message"] = new_explanation
                    
                    # Prolongation du timer de 10 secondes
                    shared_state["alert_until"] = time.time() + 10
                else:
                    shared_state["last_status"] = 0

        except json.JSONDecodeError as e:
            print(f"[ERREUR] Erreur de decodage JSON : {e}")
        except KeyError as e:
            print(f"[ERREUR] Cle manquante dans le payload : {e}")
        except Exception as e:
            print(f"[ERREUR] Erreur dans le traitement ML : {e}")

    # ── Création et configuration du client MQTT ─────────────────────────
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="thermopath-dashboard",
    )
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    # ── WILDCARD 1 : Résilience Réseau Absolue ───────────────────────────
    client.reconnect_delay_set(min_delay=1, max_delay=120)

    # Tentative de connexion initiale dans un thread séparé.
    def connect_with_retry():
        retry_delay = 1  # Délai initial en secondes
        max_delay = 30  # Délai maximum entre les tentatives
        while True:
            try:
                client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
                client.loop_forever()
                break
            except OSError as e:
                shared_state["mqtt_connected"] = False
                print(
                    f"[WARN] Broker MQTT injoignable ({BROKER_HOST}:{BROKER_PORT}). "
                    f"Nouvelle tentative dans {retry_delay}s... ({e})"
                )
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)
            except Exception as e:
                shared_state["mqtt_connected"] = False
                print(f"[ERREUR] Erreur inattendue MQTT : {e}. Nouvelle tentative dans {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_delay)

    thread = threading.Thread(target=connect_with_retry, daemon=True)
    thread.start()
    return thread


# Démarrage du thread MQTT (une seule fois grâce à cache_resource)
start_mqtt_listener()


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTERFACE UTILISATEUR (Streamlit UI/UX "Chauffeur")
# ══════════════════════════════════════════════════════════════════════════════

# ── Lecture des valeurs actuelles depuis l'état partagé ───────────────────────
# Gestion du verrouillage d'alerte (10 secondes)
if time.time() < shared_state["alert_until"]:
    status = 1
    explanation = shared_state["latched_xai_message"]
else:
    status = 0
    explanation = ""

with shared_state["lock"]:
    temp_val = shared_state["last_temp"]
    gforce_val = shared_state["last_gforce"]
    risk_val = shared_state["risk_score"]
    is_connected = shared_state["mqtt_connected"]
    msg_count = shared_state["message_count"]
    last_update = shared_state["last_update"]

# ── CSS Personnalisé : Design "Terminal Cabine" ──────────────────────────────
alert_css = ""
if status == 1:
    alert_css = """
    .stApp {
        background-color: #3b0000 !important;
        animation: blinker 1s linear infinite !important;
    }
    """

st.markdown(
    f"""
    <style>
    /* ── Import Google Fonts ─────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');

    /* ── Variables globales ──────────────────────────────────────────── */
    .stApp {{
        font-family: 'Inter', sans-serif;
    }}

    /* ── Animation de clignotement pour les alertes ──────────────────── */
    @keyframes blinker {{
        0%   {{ background-color: #1a0000; }}
        50%  {{ background-color: #660000; }}
        100% {{ background-color: #1a0000; }}
    }}

    @keyframes pulse {{
        0%   {{ transform: scale(1); }}
        50%  {{ transform: scale(1.03); }}
        100% {{ transform: scale(1); }}
    }}

    @keyframes glow {{
        0%   {{ box-shadow: 0 0 5px rgba(0, 191, 255, 0.3); }}
        50%  {{ box-shadow: 0 0 20px rgba(0, 191, 255, 0.6); }}
        100% {{ box-shadow: 0 0 5px rgba(0, 191, 255, 0.3); }}
    }}

    /* ── Cartes de métriques géantes ─────────────────────────────────── */
    .metric-card {{
        background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
        animation: glow 3s ease-in-out infinite;
    }}
    .metric-card:hover {{
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 191, 255, 0.2);
    }}
    .metric-title {{
        color: #8899AA;
        font-size: 1.1rem;
        margin-bottom: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
    }}
    .metric-value {{
        color: #FFFFFF;
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 20px rgba(0, 191, 255, 0.3);
    }}
    .metric-unit {{
        font-size: 1.8rem;
        color: #6699BB;
        font-weight: 400;
    }}

    /* ── Carte en alerte ─────────────────────────────────────────────── */
    .metric-card-alert {{
        background: linear-gradient(145deg, #2d0a0a 0%, #3b0000 100%);
        border-radius: 20px;
        padding: 30px 20px;
        text-align: center;
        border: 2px solid #ff4444;
        box-shadow: 0 0 30px rgba(255, 0, 0, 0.3);
        animation: pulse 1s ease-in-out infinite;
    }}
    .metric-card-alert .metric-value {{
        color: #FF4444;
        text-shadow: 0 0 20px rgba(255, 0, 0, 0.5);
    }}

    /* ── Badge de connexion ──────────────────────────────────────────── */
    .badge-connected {{
        background: linear-gradient(135deg, #0a3d0a 0%, #1a5c1a 100%);
        color: #00FF88;
        padding: 8px 18px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #00FF88;
        box-shadow: 0 0 12px rgba(0, 255, 136, 0.3);
    }}
    .badge-disconnected {{
        background: linear-gradient(135deg, #3d0a0a 0%, #5c1a1a 100%);
        color: #FF4444;
        padding: 8px 18px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        border: 1px solid #FF4444;
        box-shadow: 0 0 12px rgba(255, 68, 68, 0.3);
        animation: blinker 1.5s linear infinite;
    }}

    /* ── Jauge de risque ─────────────────────────────────────────────── */
    .risk-label {{
        text-align: center;
        font-size: 1.6rem;
        font-weight: 900;
        margin-top: 8px;
    }}
    .risk-low {{ color: #00FF88; }}
    .risk-medium {{ color: #FFA500; }}
    .risk-high {{ color: #FF4444; text-shadow: 0 0 10px rgba(255,0,0,0.5); }}

    /* ── En-tête ─────────────────────────────────────────────────────── */
    .header-title {{
        margin: 0;
        color: #00BFFF;
        font-size: 1.8rem;
        font-weight: 900;
        letter-spacing: 1px;
    }}
    .header-subtitle {{
        margin: 0;
        color: #556677;
        font-size: 0.9rem;
    }}

    /* ── Séparateur ──────────────────────────────────────────────────── */
    .custom-hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #334455, transparent);
        margin: 15px 0 25px 0;
    }}

    /* ── Stats footer ────────────────────────────────────────────────── */
    .stats-bar {{
        color: #556677;
        font-size: 0.85rem;
        text-align: center;
        padding: 10px;
    }}
    .stats-bar span {{
        margin: 0 15px;
    }}

    {alert_css}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Alerte sonore si anomalie détectée ───────────────────────────────────────
if status == 1:
    play_alert_sound(ALERT_SOUND_PATH)

# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE & INDICATEUR RÉSEAU
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_net = st.columns([3, 1])

with col_logo:
    st.markdown(
        """
        <p class="header-title">🚛 ThermoPath — Terminal Cabine</p>
        <p class="header-subtitle">Surveillance temps réel de la chaîne du froid</p>
        """,
        unsafe_allow_html=True,
    )

with col_net:
    # Wildcard 1 : Badge de connexion dynamique
    if is_connected:
        st.markdown(
            '<div style="text-align:right; margin-top:5px;">'
            '<span class="badge-connected">🟢 MQTT : Connecté</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="text-align:right; margin-top:5px;">'
            '<span class="badge-disconnected">🔴 MQTT : Déconnecté (Reconnexion...)</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<p style="text-align:right; color:#556677; margin:4px 0 0 0; font-size:0.85rem;">'
        f"Dernière activité : {last_update}</p>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

# ── LOGIQUE XAI (Explicabilité IA) ───────────────────────────────────────────
if status == 1:
    # Affichage bien visible de l'explication verrouillée (Composant Streamlit)
    st.warning(f"**🔍 Analyse Heuristique (XAI) :** {explanation}", icon="⚠️")

# ══════════════════════════════════════════════════════════════════════════════
# CARTES DE MÉTRIQUES GÉANTES
# ══════════════════════════════════════════════════════════════════════════════
col1, col2 = st.columns(2)

temp_display = f"{temp_val:.1f}" if temp_val is not None else "—"
gforce_display = f"{gforce_val:.2f}" if gforce_val is not None else "—"

card_class = "metric-card-alert" if status == 1 else "metric-card"

with col1:
    st.markdown(
        f'<div class="{card_class}">'
        f'<div class="metric-title">🌡️ Température Interne</div>'
        f'<p class="metric-value">{temp_display}<span class="metric-unit"> °C</span></p>'
        f"</div>",
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f'<div class="{card_class}">'
        f'<div class="metric-title">⚡ Stabilité (G-Force)</div>'
        f'<p class="metric-value">{gforce_display}<span class="metric-unit"> G</span></p>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown("", unsafe_allow_html=True)  # Espacement

# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES D'HISTORIQUE TEMPS RÉEL
# ══════════════════════════════════════════════════════════════════════════════
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.markdown(
        "<h4 style='color:#8899AA; text-align:center;'>📈 Historique Température (°C)</h4>",
        unsafe_allow_html=True,
    )
    with shared_state["lock"]:
        temp_data = list(shared_state["temp_history"])
        if len(temp_data) > 1:
            st.line_chart(pd.DataFrame(temp_data, columns=["Température (°C)"]))
        else:
            st.info("⏳ En attente de données de température...")

with col_chart2:
    st.markdown(
        "<h4 style='color:#8899AA; text-align:center;'>📈 Historique G-Force (G)</h4>",
        unsafe_allow_html=True,
    )
    with shared_state["lock"]:
        gforce_data = list(shared_state["gforce_history"])
        if len(gforce_data) > 1:
            st.line_chart(pd.DataFrame(gforce_data, columns=["G-Force (G)"]))
        else:
            st.info("⏳ En attente de données de G-Force...")

# ══════════════════════════════════════════════════════════════════════════════
# BARRE DE STATISTIQUES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="stats-bar">'
    f"<span>📩 Messages reçus : <b>{msg_count}</b></span>"
    f"<span>🕐 Dernière MAJ : <b>{last_update}</b></span>"
    f"<span>📡 Broker : <b>{BROKER_HOST}:{BROKER_PORT}</b></span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# RAFRAÎCHISSEMENT AUTOMATIQUE (Boucle de polling Streamlit)
# ══════════════════════════════════════════════════════════════════════════════
time.sleep(1)
st.rerun()