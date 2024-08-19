#!/bin/sh

# Configurer ngrok avec le token d'authentification
ngrok authtoken 2k0RNs4E1sBXP1yotUVGJkDNRNa_2266eWTTxYkQSyPRHZogL

# Lancer ngrok avec la configuration
ngrok http 63055 --domain upright-vast-stallion.ngrok-free.app &

# Lancer l'application FastAPI sur le port 63055
uvicorn app:app --host 0.0.0.0 --port 63055
