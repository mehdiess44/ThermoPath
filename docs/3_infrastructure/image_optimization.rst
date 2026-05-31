Optimisation de l'image Docker
==============================

Le ``Dockerfile`` construit l'image du dashboard Streamlit. Il privilégie une
structure simple et lisible, adaptée à un prototype démontrable.

Image Python
------------

L'image de base est :

.. code-block:: dockerfile

   FROM python:3.10-slim

La variante ``slim`` réduit la taille par rapport à une image complète tout en
gardant un environnement Python officiel. Le projet applicatif fonctionne ainsi
dans un conteneur plus léger.

Ordre des layers
----------------

Le Dockerfile copie d'abord ``requirements.txt``, installe les dépendances, puis
copie le reste du projet :

.. code-block:: dockerfile

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   COPY . .

Cet ordre améliore le cache Docker. Tant que les dépendances ne changent pas,
Docker peut réutiliser le layer d'installation, même si le code applicatif est
modifié.

Dépendances système
-------------------

``build-essential`` est installé pour permettre la compilation éventuelle de
paquets Python ayant des extensions natives. Le nettoyage de
``/var/lib/apt/lists`` réduit ensuite la taille finale de l'image.

Rôle du .dockerignore
---------------------

Le fichier ``.dockerignore`` exclut notamment les environnements virtuels, les
caches Python, Git et les checkpoints Jupyter. Cela évite de copier des fichiers
inutiles dans l'image et accélère le build.

.. important::

   Un ``.dockerignore`` propre améliore à la fois la sécurité, la vitesse de
   build et la reproductibilité de l'image.
