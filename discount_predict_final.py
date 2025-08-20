import pandas as pd
import joblib
import numpy as np
import tensorflow as tf
from datetime import datetime

def pred_dis(new_data):
    #Taking in new data in same format as training data and returing the predicted discount using a saved model
    # Feature engineering
    #Converting the date columns into datetime objects
    new_data['date'] = pd.to_datetime(new_data['date'])
    new_data['expiry_date'] = pd.to_datetime(new_data['expiry_date'])
    
    #Creating extra features which model expects
    new_data['days_until_expiry'] = (new_data['expiry_date'] - new_data['date']).dt.days
    new_data['day_of_week'] = new_data['date'].dt.dayofweek #Mon=0, Sun=0
    new_data['month'] = new_data['date'].dt.month
    
    #Dropping columns which are not used during training
    new_data = new_data.drop(['date', 'expiry_date', 'product'], axis=1)
    
    #Choosing the model
    mod_type = 'XGBoost'
    
    if mod_type in ['LinearRegression', 'RandomForest', 'XGBoost']:
        pipeline = joblib.load(f'../models/{mod_type}_pipeline.pkl') #Loading the trained pipeline
        return pipeline.predict(new_data)
    
if __name__ == "__main__":
    new_data = pd.DataFrame([{
        'date': '2025-08-24',
        'product': 'Milk',
        'category': 'Dairy',
        'expiry_date': '2025-08-25',
        'shelf_life': 7,
        'holiday': 0,
        'event': 0
    }])
    
    preds = pred_dis(new_data) #Running the prediction function
    print(f"Predicted Discount: {preds[0]:.2f}%")