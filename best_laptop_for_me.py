# -*- coding: utf-8 -*-
"""Best_Laptop_For_Me.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EK0MOipi9DzrXU8IOnRQWOKLWPr3ZY6b
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from google.colab import drive
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier  # Exemple de modèle
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# mode
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Monter Google Drive
drive.mount('/content/drive', force_remount=True)

import pandas as pd
import csv

# Path to your CSV file
file_path = '/content/drive/MyDrive/laptops_reviws.csv'

# Read the CSV file with error handling
try:
    laptop = pd.read_csv(
        file_path,
        delimiter=';',  # Adjust the delimiter if necessary
        on_bad_lines='skip',  # Skip problematic lines
        encoding='utf-8',  # Ensure correct encoding
        quoting=csv.QUOTE_MINIMAL  # Handle quoted fields correctly
    )
    print("Data loaded successfully")
    display(laptop.head())  # Display the first few rows of the data in a table format
except pd.errors.ParserError as e:
    print(f"ParserError: {e}")
    # Optionally, inspect the problematic lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if len(line.split(',')) != 11:  # Adjust based on expected number of columns
                print(f"Problematic line {i+1}: {line}")
except Exception as e:
    print(f"An error occurred: {e}")

# Assuming you have your laptop data loaded in a pandas DataFrame

# Rename 'price.in.Rs..' to 'price_in_rs'
laptop['price_in_rs'] = laptop['price(in Rs.)']
del laptop['price(in Rs.)']  # Remove the old column

# Rename 'display.in.inch.' to 'display_in_inch'
laptop['display_in_inch'] = laptop['display(in inch)']
del laptop['display(in inch)']  # Remove the old column

# Print column names
print(laptop.columns)

laptop.describe()

# Remove duplicate rows
laptop = laptop.drop_duplicates()

# Remove rows with missing values
laptop = laptop.dropna()

laptop.describe()

best_laptops = laptop[
    (laptop['rating'] >= 4) &
    (laptop['rating'] <= 5) &
    (laptop['no_of_ratings'] > 100) &
    (laptop['storage'].str.contains(r'512 GB SSD|1 TB SSD')) &
    (laptop['display_in_inch'] >= 15) &
    (laptop['processor'].str.contains(r'Intel Core i5|Intel Core i7|AMD Ryzen 5|AMD Ryzen 7|Apple M1|Apple M2|.*Processor.*'))  # Include any processor
]


# Get the first 10 best laptops
top_10_laptops = best_laptops.head(10)

print(top_10_laptops)

print(best_laptops['no_of_ratings'])

# Filter laptops with more than 100 ratings
best_laptops = best_laptops[best_laptops['no_of_ratings'] > 100]

# Display the first few rows
print(best_laptops.head())

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'laptop' (with a 'price_in_rs' column)

# Conversion rate (adjust as needed)
conversion_rate = 0.014  # Rupees to USD (example rate, find an appropriate source)

# Filter laptops based on your criteria
best_laptops = laptop[(laptop['ram'] == '16 GB DDR4 RAM') &
                      (laptop['processor'] == 'Intel Core i7 Processor (11th Gen)') &
                      (laptop['display_in_inch'] >= 15) &
                      (laptop['storage'] == '1 TB SSD') &
                      (laptop['rating'] >= 4) &
                      (laptop['rating'] <= 5) &
                      (laptop['no_of_ratings'] > 100)]

# Get the first 10 best laptops
top_10_laptops = best_laptops.head(10)

# Convert price to USD (assuming 'price_in_rs' is numerical)
top_10_laptops['price_in_usd'] = top_10_laptops['price_in_rs'] * conversion_rate

# First plot (price in USD)
plt.figure(figsize=(10, 6))
top_10_laptops.plot(x='name', y='price_in_usd', kind='barh', color='#85e0e0')
plt.axhline(y=2397.93 * conversion_rate, color='gray', linestyle='--')  # Adjust based on desired USD value
plt.title('My best options')
plt.xlabel('Price (USD)')
plt.ylabel('Laptop Brand')
plt.show()

# Second plot (rating)
plt.figure(figsize=(10, 6))
top_10_laptops.plot(x='name', y='rating', kind='barh', color='#85e0e0')
plt.axhline(y=5, color='gray', linestyle='--')
plt.title('My best options')
plt.xlabel('Rating')
plt.ylabel('Laptop Brand')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'laptop' (with a 'price_in_rs' column)

# Conversion rate (adjust as needed)
conversion_rate = 0.014  # Rupees to USD (example rate, find an appropriate source)

# Filter laptops based on your criteria
best_laptops = laptop[
    (laptop['ram'] == '16 GB DDR4 RAM') &
    (
        (laptop['processor'].str.contains(r'Intel Core i7 Processor \(1[0-3]th Gen\)')) |
        (laptop['processor'].str.contains(r'Intel Core i5 Processor \(1[0-3]th Gen\)'))
    ) &
    (laptop['display_in_inch'] >= 15) &
    (laptop['storage'].str.contains(r'512 GB SSD|1 TB SSD')) &
    (laptop['rating'] >= 4) &
    (laptop['rating'] <= 5) &
    (laptop['no_of_ratings'] > 100)
]

# Get the first 10 best laptops
top_10_laptops = best_laptops.head(20)

# Convert price to USD (assuming 'price_in_rs' is numerical)
top_10_laptops['price_in_usd'] = top_10_laptops['price_in_rs'] * conversion_rate

# Create a table of the best laptops for you (optional)
best_laptops_for_me = top_10_laptops[['name', 'ram', 'processor', 'display_in_inch',
                                      'storage', 'rating', 'price_in_usd']]  # Select desired columns

print("Best Laptops for You:")
print(best_laptops_for_me.to_string(index=False))  # Print the table without index

# Check if there are any laptops in the filtered DataFrame before plotting
if not top_10_laptops.empty:
    # First plot (price in USD)
    plt.figure(figsize=(10, 6))
    top_10_laptops.plot(x='name', y='price_in_usd', kind='barh', color='#85e0e0')
    plt.axhline(y=2397.93 * conversion_rate, color='gray', linestyle='--')  # Adjust based on desired USD value
    plt.title('My best options')
    plt.xlabel('Price (USD)')
    plt.ylabel('Laptop Brand')
    plt.show()

    # Second plot (rating)
    plt.figure(figsize=(10, 6))
    top_10_laptops.plot(x='name', y='rating', kind='barh', color='#85e0e0')
    plt.axhline(y=5, color='gray', linestyle='--')
    plt.title('My best options')
    plt.xlabel('Rating')
    plt.ylabel('Laptop Brand')
    plt.show()
else:
    print("No laptops match the specified criteria.")

#pip install sqlalchemy pyodbc

#pip install pandas sqlalchemy pyodbc

#!apt-get update
#!apt-get install -y unixodbc-dev
#!curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
#!curl https://packages.microsoft.com/config/debian/10/prod.list > /etc/apt/sources.list.d/mssql-release.list
#!apt-get update
#!ACCEPT_EULA=Y apt-get install -y msodbcsql18

#!apt-get install -y unixodbc-dev
#!pip install pyodbc sqlalchemy

#!apt-get update
#!apt-get install -y unixodbc-dev
#!apt-get install -y msodbcsql18
#!pip install pyodbc sqlalchemy

#!apt-get install -y unixodbc-dev
#!apt-get install -y msodbcsql18

#!pip install pyodbc

!pip install pymssql

import pandas as pd
import matplotlib.pyplot as plt
import pymssql

# Assuming you have a DataFrame named 'laptop' (with relevant columns)

# Conversion rate (adjust as needed)
conversion_rate = 0.014  # Rupees to USD (example rate, find an appropriate source)

# Filter laptops based on your criteria
best_laptops = laptop[
    (laptop['rating'] >= 4) &
    (laptop['rating'] <= 5) &
    (laptop['no_of_ratings'] > 100) &
    (laptop['storage'].str.contains(r'512 GB SSD|1 TB SSD')) &
    (laptop['display_in_inch'] >= 15) &
    (laptop['processor'].str.contains(r'Intel Core i5|Intel Core i7|AMD Ryzen 5|AMD Ryzen 7|Apple M1|Apple M2|.*Processor.*')) &  # Include any processor
    (laptop['ram'].str.contains(r'16 GB|32 GB|64 GB|128 GB'))  # RAM greater than 16 GB
]

# Get the first 20 best laptops
top_20_laptops = best_laptops.head(20)

# Convert price to USD (assuming 'price_in_rs' is numerical)
top_20_laptops.loc[:, 'price_in_usd'] = top_20_laptops['price_in_rs'] * conversion_rate

# Create a table of the best laptops for you (optional)
best_laptops_for_me = top_20_laptops[['name', 'ram', 'processor', 'display_in_inch',
                                      'storage', 'rating', 'price_in_usd']]  # Select desired columns

print("Meilleurs ordinateurs portables pour vous:")
print(best_laptops_for_me.to_string(index=False))  # Print the table without index

# **Database Connection Details (replace with your actual values)**
server = 'entraidfaresserver.database.windows.net'
database = 'entraidfaresbase'
username = 'Adminparcinformatique'
password = 'Racem26091862'  # Replace with your actual password

def replace_data(data, table_name):
    """Deletes old data and inserts new data into the specified table using pymssql.

    Args:
        data (list): A list of dictionaries, each representing a row to insert.
        table_name (str): The name of the target table in the database.

    Returns:
        None
    """

    try:
        with pymssql.connect(server=server, user=username, password=password, database=database) as conn:
            with conn.cursor() as cursor:
                # Delete old data
                delete_query = f"DELETE FROM {table_name}"
                cursor.execute(delete_query)

                # Construct the INSERT query with placeholders for security
                columns = ', '.join(data[0].keys())  # Get column names from the first dictionary
                placeholders = ', '.join(['%s' for _ in range(len(data[0]))])  # Match placeholders to columns
                insert_query = f"""INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"""

                for row in data:
                    cursor.execute(insert_query, tuple(row.values()))  # Extract values from each dictionary

                conn.commit()
                print("Data successfully replaced in the database.")
    except pymssql.OperationalError as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Prepare data for database insertion
data = best_laptops_for_me.to_dict('records')  # Convert DataFrame to list of dictionaries

# Choose the table name for insertion
table_name = 'BestLaptops'  # Replace with your desired table name

# Check if there are any laptops before plotting and database insertion
if not top_20_laptops.empty:
    # Data Visualization (optional)
    # ... (your visualization code remains the same)

    # Database Insertion
    replace_data(data, table_name)
else:
    print("No laptops match the specified criteria.")

"""-------------------------------------------------------------------------------

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



# Convert price to USD
conversion_rate = 0.014  # Example rate
laptop['price_in_usd'] = laptop['price_in_rs'] * conversion_rate

# Encode categorical variables
le_processor = LabelEncoder()
le_ram = LabelEncoder()
le_storage = LabelEncoder()

laptop['processor_encoded'] = le_processor.fit_transform(laptop['processor'])
laptop['ram_encoded'] = le_ram.fit_transform(laptop['ram'])
laptop['storage_encoded'] = le_storage.fit_transform(laptop['storage'])

# Select relevant features and target
features = laptop[['processor_encoded', 'ram_encoded', 'storage_encoded', 'display_in_inch', 'rating', 'no_of_ratings']]
target = laptop['price_in_usd']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Build the model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(features.columns)]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=1)

# Save the model in .h5 format
model.save('laptop_recommendation_model.h5', save_format='tf')

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras

# Load the saved model
model = keras.models.load_model('laptop_recommendation_model.h5')

# Load your laptop data (assuming the DataFrame is named 'laptop')


# Conversion rate (adjust as needed)
conversion_rate = 0.014  # Example rate
laptop['price_in_usd'] = laptop['price_in_rs'] * conversion_rate

# Encode categorical variables
le_processor = LabelEncoder()
le_ram = LabelEncoder()
le_storage = LabelEncoder()

laptop['processor_encoded'] = le_processor.fit_transform(laptop['processor'])
laptop['ram_encoded'] = le_ram.fit_transform(laptop['ram'])
laptop['storage_encoded'] = le_storage.fit_transform(laptop['storage'])

# User input
user_input = {
    'processor': 'Intel Core i5 Processor (11th Gen)',
    'ram': '16 GB DDR4 RAM',
    'storage': '1 TB SSD',
    'display_in_inch': 15.6,
    'rating': 4.5,
    'no_of_ratings': 150
}

# Preprocess user input
user_input_encoded = [
    le_processor.transform([user_input['processor']])[0],
    le_ram.transform([user_input['ram']])[0],
    le_storage.transform([user_input['storage']])[0],
    user_input['display_in_inch'],
    user_input['rating'],
    user_input['no_of_ratings']
]

# Filter laptops based on input criteria
filtered_laptops = laptop[
    (laptop['rating'] >= 4) &
    (laptop['rating'] <= 5) &
    (laptop['no_of_ratings'] > 100) &
    (laptop['storage'].str.contains(r'512 GB SSD|1 TB SSD')) &
    (laptop['display_in_inch'] >= 15) &
    (laptop['processor'].str.contains(r'Intel Core i5|Intel Core i7|AMD Ryzen 5|AMD Ryzen 7|Apple M1|Apple M2|.*Processor.*')) &  # Include any processor
    (laptop['ram'].str.contains(r'16 GB|32 GB|64 GB|128 GB'))  # RAM greater than 16 GB
]

# Predict prices for filtered laptops
if not filtered_laptops.empty:
    filtered_laptops_encoded = filtered_laptops[['processor_encoded', 'ram_encoded', 'storage_encoded', 'display_in_inch', 'rating', 'no_of_ratings']]
    filtered_laptops['predicted_price'] = model.predict(filtered_laptops_encoded)

    # Sort by predicted price
    recommended_laptops = filtered_laptops.sort_values(by='predicted_price')

    # Display the recommended laptops
    print("Recommended laptops for you:")
    print(recommended_laptops[['name', 'ram', 'processor', 'display_in_inch', 'storage', 'rating', 'price_in_usd', 'predicted_price']].to_string(index=False))
else:
    print("No laptops match the specified criteria.")

import tensorflow as tf
from tensorflow import keras

# Load the saved model (if not already loaded)
model = keras.models.load_model('laptop_recommendation_model.h5')

# Save the model to Google Drive
model.save('/content/drive/My Drive/laptop_recommendation_model.h5', save_format='tf')

print("Model saved to Google Drive successfully.")