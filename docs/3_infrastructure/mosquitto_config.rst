Configuration Mosquitto
=======================

Le fichier ``mosquitto/mosquitto.conf`` configure le broker MQTT utilisé par le
prototype.

Listener
--------

La directive suivante ouvre le broker sur le port MQTT standard du conteneur :

.. code-block:: text

   listener 1883 0.0.0.0

``1883`` est le port MQTT non chiffré conventionnel. ``0.0.0.0`` indique que le
service écoute sur toutes les interfaces réseau disponibles dans le conteneur.
Cette configuration permet au dashboard Docker de se connecter depuis le réseau
``thermopath_net``.

Accès anonyme
-------------

La directive suivante autorise les connexions sans authentification :

.. code-block:: text

   allow_anonymous true

Elle simplifie fortement la démonstration locale : aucun utilisateur, mot de
passe ou certificat n'est nécessaire pour publier et consommer les messages.

.. warning::

   Cette configuration est acceptable pour un prototype local ou une soutenance
   académique. Elle ne doit pas être utilisée en production réelle. Un broker
   industriel devrait activer l'authentification, le chiffrement TLS, des ACL par
   topic et une politique réseau restrictive.
