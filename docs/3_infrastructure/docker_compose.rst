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

Commandes d'exploitation
------------------------

Depuis la racine du depot, reconstruisez et lancez l'ensemble en arriere-plan :

.. code-block:: powershell

   docker compose up --build -d

Verifiez ensuite l'etat des services :

.. code-block:: powershell

   docker compose ps
   docker compose logs -f mqtt_broker
   docker compose logs -f dashboard

Pour arreter proprement l'infrastructure :

.. code-block:: powershell

   docker compose down

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

Role du port proxy 1884
-----------------------

Le port hote ``1884`` redirige vers ``1883`` dans le conteneur Mosquitto. Ce
choix evite les collisions avec un broker MQTT local souvent installe sur
``1883`` sous Windows, et rend le chemin Simulink plus explicite :

.. code-block:: text

   Simulink / Python hote -> 127.0.0.1:1884 -> mqtt_broker:1883

Le dashboard conteneurise ne doit pas utiliser ``1884`` : il reste sur le reseau
Docker ``thermopath_net`` et joint directement ``mqtt_broker:1883``.
