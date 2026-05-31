# 🧊 ThermoPath
**Le Jumeau Numérique Prédictif pour la Logistique du Dernier Kilomètre**

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License MIT](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)

---

## 🎯 À Propos du Projet

Dans l'industrie de la chaîne du froid, les ruptures thermiques sur le dernier kilomètre sont souvent catalysées par des chocs mécaniques invisibles qui altèrent l'isolation. **ThermoPath** est une solution d'IA prédictive qui croise la dynamique thermodynamique avec l'accélérométrie (Force G) pour détecter ces micro-défaillances en temps réel. Grâce à notre moteur de Machine Learning, le système anticipe et alerte avant que le seuil critique de température ne soit atteint, sauvant ainsi la marchandise.

## ⚡ Fonctionnalités Clés

- **Boucle Cyber-Physique Temps Réel** : Simulation accélérée 60x via Simulink pour modéliser le comportement thermodynamique.
- **Moteur Machine Learning MLOps** : Algorithme *Isolation Forest* spécialisé dans la détection prédictive d'anomalies rares sur des fenêtres glissantes.
- **Dashboard Industriel UI/UX** : Interface Streamlit thread-safe, offrant une tour de contrôle interactive avec alertes visuelles et sonores.
- **Explicabilité de l'IA (XAI)** : Diagnostic physique de la panne permettant de comprendre la cause profonde de l'alerte (corrélation choc/température).
- **Infrastructure Docker Résiliente** : Déploiement conteneurisé incluant un broker MQTT Mosquitto avec bypass de port réseau pour une connectivité Edge robuste.

## 🛠️ Stack Technique

- **Langages & Data** : Python, Pandas, Numpy
- **Machine Learning** : Scikit-Learn (Isolation Forest)
- **Frontend & UI** : Streamlit
- **IoT & Connectivité** : Paho-MQTT, Broker Mosquitto
- **Simulation** : MATLAB Simulink
- **Déploiement** : Docker, Docker Compose

## 📚 Documentation Officielle

L'architecture détaillée, le manuel de déploiement et l'ensemble des spécifications techniques sont documentés en profondeur sur notre plateforme dédiée.

👉 **[Lire la documentation complète sur ReadTheDocs](https://thermopath.readthedocs.io/fr/latest/)**

## 🚀 Quick Start

Déployez l'ensemble de l'infrastructure (Dashboard, Broker MQTT, Moteur IA) en quelques secondes via Docker Compose :

```bash
# 1. Cloner le dépôt
git clone https://github.com/mehdiess44/ThermoPath.git
cd ThermoPath

# 2. Lancer l'infrastructure
docker-compose up -d --build
```
