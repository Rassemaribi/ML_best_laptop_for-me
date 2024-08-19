# Utiliser une image de base Python
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de votre application dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Installer ngrok
RUN pip install pyngrok

# Exposer le port sur lequel l'application s'exécute
EXPOSE 63055

# Copier le fichier de configuration ngrok
COPY ngrok.yml /app/ngrok.yml

# Ajouter un script pour configurer et lancer ngrok et l'application
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Commande pour démarrer ngrok et l'application FastAPI
CMD ["/app/start.sh"]
