# Utiliser une image Python officielle comme image de base
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application dans le conteneur
COPY . .

# Exposer le port que l'application FastAPI utilise
EXPOSE 63055

# Commande pour démarrer l'application FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "63055"]
