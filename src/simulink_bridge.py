"""
Ce script est appelé par MATLAB Simulink à chaque seconde.
Il prend la température et la force G en arguments, les package en JSON, 
et les envoie au Broker MQTT.
"""
import sys
import json
import paho.mqtt.publish as publish

# Vérifie que MATLAB a bien envoyé les 2 arguments
if len(sys.argv) == 3:
    try:
        temp = float(sys.argv[1])
        g_force = float(sys.argv[2])
        
        # Formatage JSON
        payload = json.dumps({"temp": temp, "g_force": g_force})
        
        # Envoi "One-Shot" ultra-rapide (pas besoin de maintenir une connexion)
        publish.single("thermopath/sensor", payload, hostname="localhost", port=1883)
        
    except Exception as e:
        print(f"Erreur d'envoi : {e}")