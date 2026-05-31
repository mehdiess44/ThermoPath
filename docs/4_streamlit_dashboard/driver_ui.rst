Interface chauffeur
===================

L'interface Streamlit est pensée comme un terminal cabine : peu de texte, des
valeurs lisibles rapidement et une alerte très visible en cas d'anomalie.

Métriques principales
---------------------

Le chauffeur ou démonstrateur observe deux grandeurs centrales :

* la température interne ;
* la stabilité mécanique exprimée en G-Force.

Ces métriques sont affichées dans de grands blocs visuels, avec un contraste
élevé pour faciliter la lecture rapide.

Alertes visuelles
-----------------

Lorsqu'Isolation Forest signale une anomalie, l'interface bascule en mode alerte.
Le fond et les cartes deviennent rouges, avec un effet de clignotement. Le
message XAI explique la cause probable : impact mécanique, hausse thermique
rapide, dérive lente ou instabilité complexe.

Latch d'alerte de 10 secondes
-----------------------------

L'application conserve l'alerte pendant 10 secondes via ``alert_until``. Ce
latch évite qu'un événement bref disparaisse immédiatement au rerun suivant.

Ce choix est ergonomique : un conducteur ne regarde pas l'écran en permanence.
Maintenir l'alerte quelques secondes augmente la probabilité qu'elle soit vue.

Signal sonore en base64
-----------------------

Le fichier ``assets/alert.mp3`` est lu, encodé en base64 et injecté dans un
élément HTML ``audio`` avec ``autoplay``. Cette stratégie évite de devoir servir
un fichier statique séparé et reste compatible avec le rendu Streamlit.

Dégradation gracieuse
---------------------

Si le fichier audio est absent ou illisible, la fonction ne bloque pas
l'application. L'alerte visuelle reste disponible.

.. note::

   Pour une exploitation réelle, l'ergonomie sonore devrait être validée avec
   les contraintes du véhicule, le bruit ambiant et les règles de sécurité.
