import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

#Create models folder
os.makedirs('../models', exist_ok=True)

#Load data
df = pd.read_csv("../data/sales_weather_data.csv")

#Convert date fields
df['date'] = pd.to_datetime(df['date'])
df['expiry_date'] = pd.to_datetime(df['expiry_date'])

#Feature Engineering
df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

#Clean percentage fields if any
def clean_percentage(val):
    if isinstance(val, str) and '%' in val:
        return float(val.replace('%', ''))
    return float(val)

df['discount'] = df['discount'].apply(clean_percentage)

#Define features and target
features = df.drop(columns=['product_id', 'product', 'category', 'date', 'expiry_date', 'quantity_sold'])
target = df['quantity_sold']

#One-hot encoding for categorical variables
features = pd.get_dummies(features, columns=['holiday', 'event'], drop_first=True)

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)

#Train Random Forest
rf_model = RandomForestRegressor(n_estimators=8, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, '../models/rf_model.pkl')

#Accuracy Metrics
def print_metrics(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} Metrics:")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

#Predictions
rf_pred = rf_model.predict(X_test)

# Print metrics
print_metrics(y_test, rf_pred, "Random Forest")

print("\n Model trained successfully!")
