Référence API MQTT
==================

ThermoPath expose une API de messages très simple sur MQTT. Le flux principal
contient les mesures simulées issues de Simulink.

Topic
-----

Le topic utilisé est :

.. code-block:: text

   thermopath/sensor

Payload JSON
------------

Le payload est un objet JSON avec deux champs :

``temp``
   ``float`` représentant la température mesurée ou simulée.

``g_force``
   ``float`` représentant la force mécanique simulée.

Exemple de payload
------------------

.. code-block:: json

   {
     "temp": -72.4,
     "g_force": 1.03
   }

Exemple de publication Python
-----------------------------

.. code-block:: python

   import json
   import paho.mqtt.publish as publish

   payload = json.dumps({"temp": -72.4, "g_force": 1.03})
   publish.single(
       "thermopath/sensor",
       payload,
       hostname="127.0.0.1",
       port=1884,
   )

Pseudo-code Simulink
--------------------

.. code-block:: text

   à chaque seconde réelle:
       lire temperature_simulee
       lire g_force_simulee
       appeler python src/simulink_bridge.py temperature_simulee g_force_simulee

Contrat minimal
---------------

Tout consommateur du topic doit supposer que les deux champs sont requis et
convertibles en ``float``. En cas de champ absent ou de JSON invalide, le
dashboard journalise l'erreur et ignore la trame.

.. important::

   Le schéma est volontairement minimal pour le MVP. Une version industrielle
   devrait ajouter un timestamp, un identifiant de véhicule, un identifiant de
   colis et une version de schéma.
