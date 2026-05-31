Bridge Python Simulink
======================

Le fichier ``src/simulink_bridge.py`` est le connecteur minimal entre Simulink
et le broker MQTT. Il est conçu comme un processus court, appelé par Simulink à
chaque seconde réelle de simulation.

Entrées en ligne de commande
----------------------------

Le script lit les valeurs via ``sys.argv``. Il attend exactement deux arguments
après le nom du programme :

* ``sys.argv[1]`` : température ;
* ``sys.argv[2]`` : force mécanique ``g_force``.

Ces valeurs sont converties en ``float`` avant publication. Cette interface est
simple à piloter depuis un bloc MATLAB/Simulink capable d'appeler un processus
externe.

Sérialisation JSON
------------------

Le message publié respecte le format suivant :

.. code-block:: json

   {
     "temp": -72.4,
     "g_force": 1.03
   }

La sérialisation est réalisée avec ``json.dumps``. Le choix JSON facilite le
débogage, la lecture humaine et l'intégration avec d'autres consommateurs MQTT.

Publication MQTT unique
-----------------------

Le bridge utilise ``paho.mqtt.publish.single`` pour envoyer un seul message sur
le topic ``thermopath/sensor``. Dans le dépôt actuel, la publication cible
``127.0.0.1`` sur le port hôte ``1884``.

.. code-block:: python

   publish.single("thermopath/sensor", payload, hostname="127.0.0.1", port=1884)

Processus léger et éphémère
---------------------------

Le processus ne maintient pas de boucle permanente. Il démarre, publie une
trame, affiche un statut, puis se termine. Ce design est volontaire :

* il limite l'état interne du bridge ;
* il réduit les risques d'accumulation mémoire ;
* il garde Simulink maître de la cadence temporelle ;
* il rend chaque publication indépendante.

Référence du module
-------------------

.. automodule:: src.simulink_bridge
   :members:
   :undoc-members:
