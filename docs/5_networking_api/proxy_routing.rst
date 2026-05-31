Routage et réseau MQTT
======================

Le routage MQTT de ThermoPath dépend du contexte d'exécution : conteneur Docker,
machine hôte Windows, WSL2 ou Simulink.

Port interne 1883
-----------------

Dans le conteneur Mosquitto, le broker écoute sur ``1883``. C'est le port utilisé
par le dashboard lorsqu'il s'exécute dans Docker Compose :

.. code-block:: text

   mqtt_broker:1883

Port hôte 1884
--------------

Le fichier ``docker-compose.yml`` mappe le port hôte ``1884`` vers le port
interne ``1883`` :

.. code-block:: yaml

   ports:
     - "1884:1883"

Ainsi, un processus lancé sur la machine hôte, comme le bridge appelé par
Simulink, publie vers ``127.0.0.1:1884``.

Pourquoi forcer 127.0.0.1
-------------------------

``src/simulink_bridge.py`` utilise explicitement ``127.0.0.1`` au lieu de
``localhost``. Cette décision force IPv4 et évite certaines ambiguïtés de
résolution DNS sur Windows ou WSL2.

Problème possible avec IPv6
---------------------------

Sur certaines machines, ``localhost`` peut résoudre en priorité vers ``::1``,
l'adresse IPv6 locale. Si Mosquitto ou le mapping Docker n'écoute pas comme
prévu sur IPv6, la connexion échoue malgré un service actif côté IPv4.

Forcer ``127.0.0.1`` rend le chemin réseau plus déterministe pour la
démonstration.

Diagnostic rapide
-----------------

Sous Windows, verifiez les ports MQTT exposes avec :

.. code-block:: powershell

   netstat -ano | findstr :1884
   netstat -ano | findstr :1883

``1884`` doit correspondre au port hote Docker de ThermoPath. ``1883`` peut etre
occupe par un Mosquitto local sans bloquer la demonstration, car le projet
utilise justement ``1884`` cote hote.

.. warning::

   En production, le routage devrait être explicite via un nom DNS, une adresse
   de service ou une configuration d'environnement, pas codé en dur dans le
   bridge.
