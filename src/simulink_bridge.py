"""
Ce script est appelé par MATLAB Simulink à chaque seconde.
Il prend la température et la force G en arguments, les package en JSON, 
et les envoie au Broker MQTT.
"""
import sys
import json
import paho.mqtt.publish as publish

if len(sys.argv) == 3:
    try:
        temp = float(sys.argv[1])
        g_force = float(sys.argv[2])
        
        # Formatage JSON
        payload = json.dumps({"temp": temp, "g_force": g_force})
        
        # Envoi "One-Shot" vers l'IP locale (Force l'IPv4 pour contourner le blocage WSL)
        publish.single("thermopath/sensor", payload, hostname="127.0.0.1", port=1884)
        
        # Message de confirmation visuelle
        print(f"✅ SUCCÈS : {payload} envoyé sur 127.0.0.1:1884")
        
    except Exception as e:
        print(f"❌ Erreur d'envoi : {e}")