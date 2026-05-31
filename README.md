# ThermoPath
**Jumeau Numérique IoT Prédictif pour la Surveillance de la Chaîne du Froid**

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-app-red.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## À Propos du Projet
Les ruptures thermiques liées aux chocs physiques lors du transport sur le dernier kilomètre constituent un enjeu majeur pour l'intégrité de la chaîne du froid. **ThermoPath** apporte une solution performante grâce à la détection prédictive par IA et la corrélation thermodynamique, permettant d'identifier et de diagnostiquer la cause des pannes en temps réel.

## Fonctionnalités Clés
- **Boucle Cyber-Physique Temps Réel** : Simulation accélérée 60x via Simulink.
- **Moteur Machine Learning MLOps** : Isolation Forest pour la détection d'anomalies rares.
- **Dashboard Industriel UI/UX** : Interface Streamlit thread-safe avec alertes visuelles et sonores.
- **Explicabilité de l'IA (XAI)** : Fournit un diagnostic physique de la panne.
- **Infrastructure Docker Résiliente** : Broker MQTT Mosquitto avec bypass de port réseau.

## Stack Technique
- **Langage** : Python
- **Data & Modélisation** : Pandas, Scikit-Learn
- **Dashboard** : Streamlit
- **IoT & Connectivité** : Paho-MQTT, Eclipse Mosquitto
- **Infrastructure** : Docker, Docker Compose
- **Simulation** : MATLAB Simulink

## Documentation Complète

> [!IMPORTANT]
> La documentation technique complète, l'architecture détaillée et les guides de déploiement sont disponibles sur notre site officiel :
> 
> 👉 **[Lire la documentation complète sur ReadTheDocs](https://thermopath.readthedocs.io/fr/latest/)**

## Quick Start (Aperçu)

Lancez l'infrastructure complète en quelques commandes :

```bash
git clone https://github.com/mehdiess44/ThermoPath.git
cd ThermoPath
docker-compose up -d --build
```
