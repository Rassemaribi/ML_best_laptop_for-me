from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from pydantic import BaseModel

# Load the saved models
price_model = tf.keras.models.load_model('best_laptop_price_model.keras')
review_model = tf.keras.models.load_model('best_laptop_review_model.keras')

# Load your laptop data
laptop = pd.read_csv("predicted_laptops.csv")

# Encode categorical variables
le_processor = LabelEncoder()
le_ram = LabelEncoder()
le_storage = LabelEncoder()

# Fit the label encoders with the unique values from the dataset
le_processor.fit(laptop['processor'].unique())
le_ram.fit(laptop['ram'].unique())
le_storage.fit(laptop['storage'].unique())

# Define the request body structure
class LaptopRequest(BaseModel):
    min_processor: str
    min_ram: str
    min_storage: str
    min_display_in_inch: float

# Initialize FastAPI app
app = FastAPI()

@app.post("/recommend")
def recommend(request: LaptopRequest):
    try:
        # Function to handle unseen labels gracefully
        def transform_label(label, encoder):
            if label in encoder.classes_:
                return encoder.transform([label])[0]
            else:
                new_classes = np.append(encoder.classes_, label)
                encoder.classes_ = new_classes
                return encoder.transform([label])[0]

        # Encode user input
        user_input_encoded = np.array([
            transform_label(request.min_processor, le_processor),
            transform_label(request.min_ram, le_ram),
            transform_label(request.min_storage, le_storage),
            request.min_display_in_inch,
            0,  # Placeholder for rating
            0   # Placeholder for no_of_ratings
        ]).reshape(1, -1)

        # Predict price and review for the input criteria
        predicted_price = price_model.predict(user_input_encoded).astype(float)[0][0]
        predicted_review = review_model.predict(user_input_encoded).astype(float)[0][0]

        # Filter laptops based on input criteria
        filtered_laptops = laptop[
            (laptop['rating'] >= 3.9) &
            (laptop['rating'] <= 5) &
            (laptop['no_of_ratings'] >= 100) &
            (laptop['storage'].str.contains(request.min_storage)) &
            (laptop['display_in_inch'] >= request.min_display_in_inch) &
            (laptop['processor'].str.contains(request.min_processor)) &
            (laptop['ram'].str.contains(request.min_ram))
        ]

        # Predict prices and reviews for filtered laptops
        if not filtered_laptops.empty:
            filtered_laptops_encoded = filtered_laptops[['processor_encoded', 'ram_encoded', 'storage_encoded', 'display_in_inch', 'rating', 'no_of_ratings']]
            filtered_laptops_encoded = filtered_laptops_encoded.to_numpy()

            # Predict prices and reviews
            filtered_laptops['predicted_price'] = price_model.predict(filtered_laptops_encoded).astype(float)
            filtered_laptops['predicted_review'] = review_model.predict(filtered_laptops_encoded).astype(float)

            # Sort by predicted review and price
            recommended_laptops = filtered_laptops.sort_values(by=['predicted_review', 'predicted_price'], ascending=[False, True])

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
            return {"message": "No laptops match the specified criteria."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
