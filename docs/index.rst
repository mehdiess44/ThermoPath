ThermoPath
==========

ThermoPath est un jumeau numérique IoT expérimental pour la surveillance
prédictive de la chaîne du froid sur le dernier kilomètre. Le projet relie une
simulation cyber-physique MATLAB Simulink, un bridge Python, un broker MQTT
Eclipse Mosquitto, une application Streamlit et un modèle d'anomalie fondé sur
Isolation Forest.

Objectif du projet
------------------

L'objectif est de détecter plus tôt les situations où un colis frigorifique
risque de sortir de sa plage thermique acceptable. Le système ne se limite pas
à observer une température trop élevée : il combine la dynamique thermique et
les chocs mécaniques afin d'identifier des signaux faibles, comme une fatigue
d'isolant ou une dérive progressive après impact.

Architecture globale
--------------------

Le flux nominal suit une boucle simple et démontrable :

1. Simulink simule l'environnement physique et produit une température et une
   force mécanique.
2. ``src/simulink_bridge.py`` reçoit ces valeurs en arguments, les sérialise en
   JSON et les publie sur MQTT.
3. Mosquitto transporte le message sur le topic ``thermopath/sensor``.
4. Streamlit consomme le flux, maintient un état temps réel et affiche une
   interface chauffeur.
5. Isolation Forest calcule un statut d'anomalie et un score de risque humain.

.. important::

   ThermoPath est un projet académique et expérimental. La documentation et le
   prototype montrent une démarche industrialisable, mais ne constituent pas une
   validation industrielle, médicale, pharmaceutique ou réglementaire réelle.

Table des matières
------------------

.. toctree::
   :maxdepth: 2
   :caption: Démarrage

   0_getting_started/installation

.. toctree::
   :maxdepth: 2
   :caption: Boucle cyber-physique

   1_cyber_physical_loop/overview
   1_cyber_physical_loop/simulink_model
   1_cyber_physical_loop/python_bridge

.. toctree::
   :maxdepth: 2
   :caption: Données et MLOps

   2_data_mlops/prep_pipeline
   2_data_mlops/fault_injection
   2_data_mlops/feature_engineering
   2_data_mlops/isolation_forest

.. toctree::
   :maxdepth: 2
   :caption: Infrastructure

   3_infrastructure/docker_compose
   3_infrastructure/mosquitto_config
   3_infrastructure/image_optimization

.. toctree::
   :maxdepth: 2
   :caption: Dashboard Streamlit

   4_streamlit_dashboard/architecture
   4_streamlit_dashboard/concurrency_control
   4_streamlit_dashboard/driver_ui

.. toctree::
   :maxdepth: 2
   :caption: Réseau et API

   5_networking_api/proxy_routing
   5_networking_api/troubleshooting_windows_wsl
   5_networking_api/api_reference
