Docker Compose
==============

L'infrastructure Docker de ThermoPath assemble deux services principaux :
``mqtt_broker`` et ``dashboard``. Le fichier ``docker-compose.yml`` permet de
démarrer l'ensemble de la démonstration avec une topologie réseau reproductible.

Services
--------

``mqtt_broker`` utilise l'image officielle ``eclipse-mosquitto``. Il expose le
broker MQTT interne sur le port ``1883`` du conteneur et mappe ce port vers
``1884`` côté machine hôte.

``dashboard`` construit l'image applicative à partir du ``Dockerfile`` local. Il
lance Streamlit sur le port ``8501`` et reçoit les variables d'environnement :

.. code-block:: yaml

   BROKER_HOST: mqtt_broker
   BROKER_PORT: 1883

Ces valeurs indiquent au dashboard de joindre le broker par son nom de service
Docker, et non par ``localhost``.

Réseau thermopath_net
---------------------

Les deux services sont attachés au réseau bridge ``thermopath_net``. Docker
fournit alors une résolution DNS interne : depuis le conteneur ``dashboard``, le
nom ``mqtt_broker`` résout vers l'adresse du conteneur Mosquitto.

Résolution DNS entre conteneurs
-------------------------------

Dans Docker Compose, ``localhost`` désigne le conteneur courant. Si le dashboard
utilisait ``localhost:1883``, il chercherait un broker à l'intérieur de son
propre conteneur. L'utilisation de ``mqtt_broker:1883`` est donc nécessaire pour
atteindre le service Mosquitto.

.. note::

   Le port ``1884`` est utile depuis la machine hôte, notamment pour le bridge
   Simulink. Entre conteneurs, le port interne ``1883`` est utilisé directement.
