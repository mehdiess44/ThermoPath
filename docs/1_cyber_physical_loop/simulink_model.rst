Modèle Simulink
===============

Simulink représente la partie cyber-physique de ThermoPath. Son rôle est de
produire un flux contrôlé de mesures simulées, suffisamment réaliste pour tester
le reste de la chaîne sans dépendre de capteurs matériels.

Rôle dans le prototype
----------------------

Le modèle Simulink agit comme source de données temps réel. À chaque pas, il
fournit deux grandeurs principales :

* la température interne simulée du conteneur ;
* la force mécanique simulée, exprimée en ``g_force``.

Ces valeurs sont transmises au bridge Python afin d'être publiées sur MQTT.

Ouverture du modele
-------------------

Le modele se trouve dans ``app/publish_mqtt.slx``. Ouvrez-le depuis MATLAB
Simulink, puis verifiez que le bloc charge d'appeler le systeme externe pointe
vers le bridge Python du depot.

Si le modele est lance depuis le dossier ``app``, la commande de reference est :

.. code-block:: text

   python ../src/simulink_bridge.py <temp> <g_force>

Les deux arguments doivent correspondre aux signaux Simulink representant la
temperature et la force mecanique instantanee. Le bridge convertit ces valeurs
en ``float`` avant publication MQTT.

Modèle thermodynamique à haut niveau
------------------------------------

Le modèle peut être lu comme une approximation du comportement thermique d'un
emballage froid. Il prend en compte l'inertie du système : la température ne
change pas instantanément, mais évolue progressivement sous l'effet des pertes,
des perturbations et des défauts.

Dans le cadre académique du projet, la priorité n'est pas de certifier un modèle
physique industriel complet. L'objectif est de produire une dynamique cohérente
pour vérifier l'alignement entre simulation, données d'entraînement et
inférence.

Accélération temporelle 60x
---------------------------

ThermoPath utilise une accélération temporelle :

``1 seconde réelle = 1 minute simulée``

Ce choix permet de rejouer rapidement des phénomènes qui seraient trop lents à
observer en temps réel strict. Une dérive de 30 minutes simulées peut ainsi être
observée en 30 secondes de démonstration.

Cette dilatation temporelle est aussi une contrainte ML : le dashboard conserve
une fenetre glissante de ``5`` messages, donc ``5`` secondes reelles. Avec le
facteur ``60x``, cette fenetre represente ``5`` minutes simulees, soit la meme
granularite que les donnees d'entrainement.

Alignement avec l'entraînement
------------------------------

Les données d'entraînement sont préparées à une fréquence d'une ligne par
minute, notamment avec ``asfreq('1min')``. En production simulée, Simulink émet
une trame chaque seconde réelle. Comme cette seconde représente une minute
simulée, la cadence d'inférence reste compatible avec les features apprises.

.. important::

   Cette convention 60x est centrale : elle donne un sens physique à la
   différence de température calculée entre deux trames successives.
