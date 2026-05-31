Feature engineering
===================

Le feature engineering transforme le flux brut en variables interprétables par
le modèle. Dans ThermoPath, les features combinent valeurs instantanées,
statistiques glissantes et vitesse thermique.

Features utilisées
------------------

Le modèle utilise les sept colonnes suivantes :

* ``thermal_shipper_temp_reading`` ;
* ``g_force`` ;
* ``temp_mean`` ;
* ``temp_std`` ;
* ``g_force_mean`` ;
* ``g_force_std`` ;
* ``temp_velocity``.

Fenêtre glissante
-----------------

Les moyennes et écarts-types sont calculés sur une fenêtre glissante de taille
``5``. En entraînement, cela correspond à cinq points temporels successifs après
resampling à la minute. En inférence, le dashboard maintient deux ``deque`` de
taille 5 pour la température et la G-Force.

La fenêtre glissante donne au modèle une mémoire courte : il ne voit pas
seulement la mesure courante, mais aussi le contexte immédiat.

Vélocité thermique
------------------

La feature ``temp_velocity`` mesure la variation de température. En notation
simple :

.. math::

   temp\_velocity = T_t - T_{t-1}

Dans ``src/features.py``, l'entraînement utilise ``diff()`` sur la série
temporelle resamplée. Dans l'application temps réel, la logique exploite la
différence entre les valeurs du buffer pour caractériser la dynamique récente.

Alignement entraînement / inférence
-----------------------------------

L'alignement temporel est un point critique :

* en entraînement, la différence est calculée sur des données espacées d'une
  minute ;
* en production simulée, deux trames sont reçues chaque seconde réelle ;
* comme Simulink accélère le temps 60x, une seconde réelle correspond à une
  minute simulée.

Ainsi, la cadence observée par le modèle reste cohérente entre apprentissage et
démonstration.

.. important::

   Si la cadence Simulink change, les features dynamiques doivent être revues.
   Sinon, le modèle pourrait interpréter une vitesse thermique avec une mauvaise
   unité temporelle.

Référence des fonctions
-----------------------

.. automodule:: src.features
   :members:
   :undoc-members:
