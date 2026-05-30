# Utilise une image Python officielle, version allégée
FROM python:3.10-slim

# Définit le dossier de travail dans le conteneur
WORKDIR /app

# Met à jour le système et installe les outils de compilation de base
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copie d'abord le fichier des dépendances (pour optimiser le cache Docker)
COPY requirements.txt .

# Installe les librairies Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le reste du projet (le code, les modèles, les assets)
COPY . .

# Expose le port de Streamlit
EXPOSE 8501

# Commande de lancement
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]