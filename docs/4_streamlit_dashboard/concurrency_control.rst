Contrôle de concurrence
=======================

Le dashboard manipule un état partagé entre deux contextes d'exécution : le
thread MQTT, qui écrit les nouvelles mesures, et le thread Streamlit, qui lit les
valeurs pour afficher l'interface.

Problème de concurrence
-----------------------

Une structure de données modifiée pendant sa lecture peut provoquer des erreurs
ou des affichages incohérents. Le cas typique est une itération sur un ``deque``
pendant qu'un autre thread y ajoute une valeur.

L'erreur associée peut ressembler à :

.. code-block:: text

   RuntimeError: deque mutated during iteration

shared_state
------------

``shared_state`` est un dictionnaire global mis en cache par Streamlit. Il
contient les dernières valeurs, les historiques, les buffers glissants, le score
de risque et l'état de connexion MQTT.

Cette approche centralise l'information et simplifie le passage de données entre
le listener MQTT et l'interface.

deque
-----

Les ``deque`` de Python sont utilisés pour conserver uniquement les derniers
points utiles :

* ``deque(maxlen=5)`` pour les buffers ML ;
* ``deque(maxlen=100)`` pour les historiques graphiques.

Le paramètre ``maxlen`` évite une croissance mémoire illimitée.

threading.Lock
--------------

Un ``threading.Lock`` est le mécanisme classique pour protéger les sections
critiques. Il agit comme un mutex : un seul thread peut entrer dans la zone
protégée à un instant donné.

Une version renforcée du dashboard pourrait encadrer les écritures MQTT et les
lectures UI avec un verrou :

.. code-block:: python

   with state_lock:
       temp_data = list(shared_state["temp_history"])

Pourquoi le mutex évite l'erreur
--------------------------------

Le mutex empêche la mutation d'un ``deque`` pendant que Streamlit le convertit
en liste ou l'envoie à un graphique. L'interface lit alors un instantané stable,
tandis que le thread MQTT attend brièvement son tour pour écrire.

.. important::

   Le dépôt actuel reste léger pour le MVP. Si la fréquence MQTT augmente ou si
   plusieurs consommateurs sont ajoutés, l'ajout explicite d'un lock devient une
   amélioration prioritaire.
