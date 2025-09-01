import requests
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

#Setting coordinates and weather API Url
LAT = 53.27472923790981
LON = -9.048561251875782
URL = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&daily=temperature_2m_min,temperature_2m_max,temperature_2m_mean,rain_sum,snowfall_sum&forecast_days=5&timezone=auto"

#Loading pre-trained Random Forest model
rf_model = joblib.load("models/rf_model.pkl")
#Storing the feature names used during training to align new data
model_features = rf_model.feature_names_in_

#Fetching 5-day weather forecast
resp = requests.get(URL)
weather = resp.json()

#Extracting daily weather features
dates = weather["daily"]["time"]
tmin = weather["daily"]["temperature_2m_min"]
tmax = weather["daily"]["temperature_2m_max"]
tmean = weather["daily"]["temperature_2m_mean"]
rain = weather["daily"]["rain_sum"]
snow = weather["daily"]["snowfall_sum"]

#Predicting sales for next 5 days
print("\n PN028 Predicted Sales Quantity (next 5 days):\n")
for i in range(len(dates)):
    date_str = dates[i]
    date = pd.to_datetime(date_str)

    #Preparing manual input for prediction
    manual_input = {
        'product_id': ['PN028'],
        'product': ['Milk'],
        'category': ['Dairy'],
        'quantity_sold': [np.nan],  #Target
        'date': [date],
        'expiry_date': [date + timedelta(days=5)], #Expected expiry date
        'unit_price': [1.29],
        'discount': [0.0],
        'discount_price': [1.29],
        'total': [20],
        'shelf_life': [7],
        'holiday': [0],
        'event': [0],
        'temperature_min': [tmin[i]],
        'temperature_max': [tmax[i]],
        'temperature_mean': [tmean[i]],
        'rain_mm': [rain[i]],
        'snowfall_cm': [snow[i]]
    }

    #Converting dictionary to DataFrame
    new_data = pd.DataFrame(manual_input)

    #Feature engineering
    new_data['days_to_expiry'] = (new_data['expiry_date'] - new_data['date']).dt.days
    new_data['month'] = new_data['date'].dt.month
    new_data['day'] = new_data['date'].dt.day
    new_data['dayofweek'] = new_data['date'].dt.dayofweek

    #Droping columns not used by the model
    X_new = new_data.drop(columns=['product_id', 'product', 'category', 'date', 'expiry_date', 'quantity_sold'])

    #Converting categorical variables (holiday, event) to one-hot encoding
    X_new = pd.get_dummies(X_new, columns=['holiday', 'event'], drop_first=True)

    #Aligning columns with the features used during model training
    X_new = X_new.reindex(columns=model_features, fill_value=0)

    #Predicting sales
    rf_pred = rf_model.predict(X_new)[0]

    #Printing forecasted quantity for the day
    print(f"{date.strftime('%d-%b-%Y')} -> Predicted Quantity: {rf_pred:.0f}")
