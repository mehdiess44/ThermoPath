Isolation Forest
================

ThermoPath utilise Isolation Forest pour détecter des anomalies sans devoir
entraîner un classifieur supervisé complet. Ce choix correspond bien au contexte
du projet : les anomalies logistiques réelles sont rares, diverses et difficiles
à annoter exhaustivement.

Principe d'isolation
--------------------

Isolation Forest construit plusieurs arbres aléatoires. Une observation normale,
située dans une région dense de l'espace des features, demande généralement plus
de coupes pour être isolée. Une observation atypique est isolée plus rapidement,
car elle se trouve loin du comportement majoritaire.

Score d'anomalie
----------------

Dans scikit-learn, ``predict`` renvoie :

* ``1`` pour une observation normale ;
* ``-1`` pour une anomalie.

La méthode ``decision_function`` fournit un score brut relatif à la frontière de
décision. Plus le score devient négatif, plus l'observation est anormale.

Rôle du StandardScaler
----------------------

Les features n'ont pas les mêmes unités : degrés Celsius, G-Force, moyennes,
écarts-types et variations. ``StandardScaler`` centre et réduit les variables
pour éviter qu'une feature domine artificiellement les distances et les coupes
aléatoires du modèle.

Le scaler est ajusté uniquement sur le jeu d'entraînement, puis appliqué au jeu
de test et à l'inférence. Cette séparation évite le data leakage.

Paramètre contamination
-----------------------

Le modèle du dépôt utilise :

.. code-block:: python

   IsolationForest(contamination=0.0065, random_state=42)

``contamination`` indique la proportion attendue d'anomalies. Une valeur faible
correspond à l'hypothèse que les défauts restent rares dans le flux normal.

Score de risque humain
----------------------

Le dashboard transforme le score brut en indice lisible de ``0`` à ``100`` :

.. code-block:: python

   risk_index = 50 - (raw_score * 200)
   risk_score = int(np.clip(risk_index, 0, 100))

Cette transformation rend l'information exploitable par un conducteur ou un
opérateur. Un score proche de 0 indique une situation rassurante ; un score
proche de 100 indique une situation fortement atypique.

.. warning::

   Ce score est un indicateur d'aide à la décision. Il ne doit pas être présenté
   comme une probabilité réglementaire de défaillance.

Référence du module
-------------------

.. automodule:: src.model
   :members:
   :undoc-members:
