Architecture Streamlit
======================

L'application ``app/app.py`` fournit l'interface temps réel de ThermoPath. Elle
combine une interface Streamlit, un client MQTT en arrière-plan, le modèle
Isolation Forest et des composants d'alerte visuelle et sonore.

Thread principal UI
-------------------

Streamlit exécute le script pour produire l'interface utilisateur. Le thread
principal lit l'état courant, affiche les métriques, les graphiques et les
alertes, puis déclenche un rafraîchissement périodique avec ``st.rerun``.

Ce modèle est adapté à une démonstration temps réel : l'interface est
reconstruite régulièrement à partir du dernier état disponible.

Thread MQTT en arrière-plan
---------------------------

Le client MQTT est lancé dans un thread ``daemon`` séparé. Ce thread se connecte
au broker, s'abonne au topic ``thermopath/sensor`` et traite chaque message
reçu. Il met ensuite à jour l'état partagé utilisé par l'interface.

Cette séparation évite de bloquer l'affichage Streamlit pendant l'attente réseau.

Utilisation de @st.cache_resource
---------------------------------

``@st.cache_resource`` est utilisé pour initialiser une seule fois les ressources
globales :

* l'état partagé ;
* le thread MQTT ;
* le chargement du modèle et du scaler.

Sans cache, Streamlit pourrait recréer ces objets à chaque rerun, ce qui
multiplierait les connexions MQTT et les threads.

Calcul temps réel
-----------------

À chaque message, l'application :

1. décode le payload JSON ;
2. met à jour les buffers et historiques ;
3. calcule les features glissantes ;
4. applique le ``StandardScaler`` ;
5. interroge Isolation Forest ;
6. convertit le score brut en score de risque.

.. note::

   Le dashboard concentre plusieurs responsabilités pour rester simple dans un
   MVP. Dans une version industrielle, le calcul ML pourrait être séparé dans un
   service backend dédié.

Lancement
---------

En mode conteneurise, le service ``dashboard`` est lance par Docker Compose et
expose Streamlit sur :

.. code-block:: text

   http://localhost:8501

En mode bare metal, lancez l'application depuis la racine du depot :

.. code-block:: powershell

   $env:BROKER_HOST = "127.0.0.1"
   $env:BROKER_PORT = "1884"
   streamlit run app/app.py

Le port ``1884`` cible le proxy hote expose par ``docker-compose.yml``. Dans
Docker Compose, ces variables valent ``mqtt_broker`` et ``1883`` afin d'utiliser
le reseau interne ``thermopath_net``.
