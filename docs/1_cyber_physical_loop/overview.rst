Vue d'ensemble cyber-physique
=============================

Un jumeau numérique est une représentation logicielle d'un système physique. Il
permet de simuler, observer et tester le comportement d'un système sans devoir
attendre des incidents réels. Dans ThermoPath, le jumeau numérique représente
un colis ou conteneur froid soumis aux contraintes du dernier kilomètre :
vibrations, chocs, inertie thermique et dérive progressive.

Contexte chaîne du froid
------------------------

La chaîne du froid impose de maintenir un produit dans une plage de température
contrôlée pendant le transport. Sur le dernier kilomètre, les véhicules légers,
les arrêts fréquents et les routes urbaines peuvent générer des perturbations
brèves mais répétées. Une simple lecture de température peut arriver trop tard :
elle constate souvent le dépassement après le début de la dégradation.

Le triptyque de risque
----------------------

ThermoPath modélise trois phénomènes liés :

* ``choc mécanique`` : un pic de ``g_force`` représente un impact ou une forte
  vibration.
* ``fatigue d'isolant`` : l'impact peut détériorer l'étanchéité ou créer un
  pont thermique.
* ``dérive thermique`` : la température interne augmente progressivement après
  le choc, même si le seuil critique n'est pas immédiatement franchi.

Cette lecture est utile pour la prédiction : le système cherche une combinaison
de signaux, pas seulement une valeur instantanée.

Cycle global
------------

Le cycle opérationnel suit la chaîne suivante :

``Simulink -> Python Bridge -> MQTT -> Streamlit -> ML``

Simulink joue le rôle de monde physique simulé. Le bridge Python transforme les
valeurs numériques en message JSON. Mosquitto assure le transport léger du flux
IoT. Streamlit reçoit les données, alimente des buffers glissants, calcule les
features et applique le modèle Isolation Forest.

.. note::

   Cette architecture sépare volontairement simulation, transport, interface et
   intelligence artificielle. Chaque bloc peut ainsi être testé ou remplacé plus
   facilement.
