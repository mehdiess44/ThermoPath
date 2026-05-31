Pipeline de préparation des données
===================================

Le pipeline de préparation est implémenté dans ``src/data_prep.py``. Il convertit
un fichier brut en série temporelle régulière, exploitable pour les fenêtres
glissantes et l'entraînement du modèle.

Chargement et filtrage
----------------------

La fonction ``load_and_resample`` lit un fichier CSV et filtre le trajet
``batch001`` par défaut. Ce filtrage isole un scénario cohérent, ce qui évite de
mélanger plusieurs lots ou trajets dans une même dynamique temporelle.

Parsing temporel
----------------

La colonne ``date`` est convertie avec ``pd.to_datetime`` puis utilisée comme
index. Le DataFrame est trié chronologiquement et les doublons temporels sont
supprimés en conservant la première occurrence.

Cette étape est indispensable pour les opérations temporelles de pandas :
resampling, interpolation et calculs glissants.

Resampling à la minute
----------------------

Le code applique :

.. code-block:: python

   df = df.asfreq("1min")

Cette instruction impose une grille temporelle régulière d'une ligne par minute.
Les minutes absentes deviennent explicites, ce qui rend possible une
interpolation contrôlée.

Interpolation linéaire
----------------------

La température ``thermal_shipper_temp_reading`` est interpolée avec la méthode
``time`` :

.. code-block:: python

   df["thermal_shipper_temp_reading"] = (
       df["thermal_shipper_temp_reading"].interpolate(method="time")
   )

Ce choix est physiquement raisonnable pour un système thermique inertiel :
entre deux mesures proches, la température évolue généralement de manière
progressive plutôt que par sauts arbitraires.

Justification physique
----------------------

La chaîne du froid est un phénomène continu. Même si les capteurs n'échantillonnent
pas chaque minute, l'état thermique existe entre deux mesures. L'interpolation
linéaire ne prétend pas reconstruire parfaitement le monde réel ; elle crée une
approximation stable pour entraîner et tester le modèle.

.. warning::

   L'interpolation doit rester réservée à des intervalles raisonnables. Des
   lacunes très longues dans les données nécessiteraient une analyse qualité
   séparée.

Référence du module
-------------------

.. automodule:: src.data_prep
   :members:
   :undoc-members:
