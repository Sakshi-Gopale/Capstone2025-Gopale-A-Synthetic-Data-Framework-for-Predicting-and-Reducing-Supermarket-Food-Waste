import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from prophet.serialize import model_from_json

#Manual input 
manual_input = {
    'product_id': ['PN028'],
    'product': ['Milk'],
    'category': ['Dairy'],
    'quantity_sold': [np.nan],  # Target (ignored)
    'date': ['2025-08-02'],
    'expiry_date': ['2025-08-06'],
    'unit_price': [60],
    'discount': [10.0],  # Ensure this is float, not '10%'
    'discount_price': [54],
    'total': [1080],
    'shelf_life': [4],
    'holiday': [0],
    'event': [0],
    'temperature_min': [0],
    'temperature_max': [7.5],
    'temperature_mean': [15],
    'rain_mm': [1.0],
    'snowfall_cm': [0.0]
}

# Create DataFrame
new_data = pd.DataFrame(manual_input)

# Preprocessing
new_data['date'] = pd.to_datetime(new_data['date'])
new_data['expiry_date'] = pd.to_datetime(new_data['expiry_date'])
new_data['days_to_expiry'] = (new_data['expiry_date'] - new_data['date']).dt.days
new_data['month'] = new_data['date'].dt.month
new_data['day'] = new_data['date'].dt.day
new_data['dayofweek'] = new_data['date'].dt.dayofweek

# Drop unused
X_new = new_data.drop(columns=['product_id', 'product', 'category', 'date', 'expiry_date', 'quantity_sold'])

# One-hot encoding
X_new = pd.get_dummies(X_new, columns=['holiday', 'event'], drop_first=True)

# Align with training features
model_features = joblib.load("../models/linear_model.pkl").feature_names_in_
X_new = X_new.reindex(columns=model_features, fill_value=0)

# Load models
lr_model = joblib.load("../models/linear_model.pkl")
rf_model = joblib.load("../models/rf_model.pkl")
# dl_model = load_model("../models/dl_model.h5")

# Load Prophet
with open("../models/prophet_model.json", "r") as f:
    prophet_model = model_from_json(f.read())

# Predict
lr_pred = lr_model.predict(X_new)[0]
rf_pred = rf_model.predict(X_new)[0]

print("\nPredicted Sales Quantity:")
print(f"Linear Regression:     {lr_pred:.0f}")
print(f"Random Forest:         {rf_pred:.0f}")
