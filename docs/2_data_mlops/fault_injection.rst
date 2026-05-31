Injection synthétique de défauts
================================

L'injection synthétique de défauts est implémentée dans ``src/features.py`` via
``inject_synthetic_faults``. Elle enrichit les données préparées avec des chocs
mécaniques et leurs conséquences thermiques.

Pourquoi injecter des défauts ?
-------------------------------

Les anomalies réelles sont rares, parfois mal annotées et coûteuses à collecter.
Pour tester un modèle non supervisé comme Isolation Forest, le projet crée donc
des scénarios contrôlés. Cette approche permet de vérifier que le système réagit
à des événements plausibles sans attendre un incident terrain.

Chocs mécaniques
----------------

Le pipeline initialise une force mécanique proche de ``1.0 G`` avec un bruit
faible. Il sélectionne ensuite plusieurs positions temporelles et remplace la
valeur par un pic entre ``3.0`` et ``5.0 G``. Ces pics représentent des
événements tels qu'un nid-de-poule, une chute locale ou un choc de manutention.

La colonne ``is_shock`` sert de repère d'évaluation pendant les expérimentations.
Elle n'est pas une vérité terrain industrielle, mais une annotation synthétique
utile pour mesurer la réaction du modèle.

Dérive thermique progressive
----------------------------

Après chaque choc, le code ajoute une pénalité thermique progressive pendant une
fenêtre de 60 minutes. Cette dérive simule l'idée qu'un impact peut endommager
l'isolation et provoquer une hausse graduelle de température.

.. code-block:: text

   choc mécanique -> fatigue locale -> dérive thermique

Intérêt pour l'apprentissage non supervisé
------------------------------------------

Isolation Forest n'apprend pas une classe explicite comme un classifieur
supervisé traditionnel. Il apprend plutôt la structure des observations
majoritaires et isole les points rares. Les défauts synthétiques permettent de
tester si les points combinant choc, variabilité et dérive sont effectivement
considérés comme atypiques.

.. note::

   Les défauts synthétiques ne remplacent pas une campagne de validation réelle.
   Ils constituent un banc d'essai contrôlé pour un prototype académique.
