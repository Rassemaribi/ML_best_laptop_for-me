from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from pydantic import BaseModel
import os

# Chemins relatifs
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Charger les modèles enregistrés
price_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_laptop_price_model.keras'))
review_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'best_laptop_review_model.keras'))

# Charger vos données de laptop
laptop = pd.read_csv(os.path.join(BASE_DIR, 'predicted_laptops.csv'))

# Encoder les variables catégorielles
le_processor = LabelEncoder()
le_ram = LabelEncoder()
le_storage = LabelEncoder()

# Ajuster les encodeurs avec les valeurs uniques du dataset
le_processor.fit(laptop['processor'].unique())
le_ram.fit(laptop['ram'].unique())
le_storage.fit(laptop['storage'].unique())

# Définir la structure du corps de la requête
class LaptopRequest(BaseModel):
    min_processor: str
    min_ram: str
    min_storage: str
    min_display_in_inch: float

# Initialiser l'application FastAPI
app = FastAPI()

@app.post("/recommend")
def recommend(request: LaptopRequest):
    try:
        # Fonction pour gérer les labels non vus
        def transform_label(label, encoder):
            if label in encoder.classes_:
                return encoder.transform([label])[0]
            else:
                new_classes = np.append(encoder.classes_, label)
                encoder.classes_ = new_classes
                return encoder.transform([label])[0]

        # Encoder l'entrée de l'utilisateur
        user_input_encoded = np.array([
            transform_label(request.min_processor, le_processor),
            transform_label(request.min_ram, le_ram),
            transform_label(request.min_storage, le_storage),
            request.min_display_in_inch,
            0,  # Placeholder pour rating
            0   # Placeholder pour no_of_ratings
        ]).reshape(1, -1)

        # Prédire le prix et l'avis pour les critères d'entrée
        predicted_price = price_model.predict(user_input_encoded).astype(float)[0][0]
        predicted_review = review_model.predict(user_input_encoded).astype(float)[0][0]

        # Filtrer les laptops en fonction des critères d'entrée
        filtered_laptops = laptop[
            (laptop['rating'] >= 3.9) &
            (laptop['rating'] <= 5) &
            (laptop['no_of_ratings'] >= 100) &
            (laptop['storage'].str.contains(request.min_storage)) &
            (laptop['display_in_inch'] >= request.min_display_in_inch) &
            (laptop['processor'].str.contains(request.min_processor)) &
            (laptop['ram'].str.contains(request.min_ram))
        ]

        # Prédire les prix et les avis pour les laptops filtrés
        if not filtered_laptops.empty:
            filtered_laptops_encoded = filtered_laptops[['processor_encoded', 'ram_encoded', 'storage_encoded', 'display_in_inch', 'rating', 'no_of_ratings']]

            # Convertir en numpy array pour la prédiction
            filtered_laptops_encoded = filtered_laptops_encoded.to_numpy()

            # Prédire les prix et les avis
            filtered_laptops['predicted_price'] = price_model.predict(filtered_laptops_encoded).astype(float)
            filtered_laptops['predicted_review'] = review_model.predict(filtered_laptops_encoded).astype(float)

            # Trier par avis prédit et prix
            recommended_laptops = filtered_laptops.sort_values(by=['predicted_review', 'predicted_price'], ascending=[False, True])

            # Retourner les laptops recommandés avec les critères d'entrée et les valeurs prédites
            return {
                "input_criteria": {
                    "min_processor": request.min_processor,
                    "min_ram": request.min_ram,
                    "min_storage": request.min_storage,
                    "min_display_in_inch": request.min_display_in_inch,
                    "predicted_review": predicted_review
                },
                "recommended_laptops": recommended_laptops[['name', 'ram', 'processor', 'display_in_inch', 'storage', 'rating', 'predicted_price']].to_dict('records')
            }
        else:
            return {"message": "Aucun laptop ne correspond aux critères spécifiés."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
