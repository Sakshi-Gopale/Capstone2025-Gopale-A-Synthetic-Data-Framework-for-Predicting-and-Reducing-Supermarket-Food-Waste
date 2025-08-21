import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

#Loading data
df = pd.read_csv("data/Sales_Dataset.csv")

#Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['expiry_date'] = pd.to_datetime(df['expiry_date'], dayfirst=True, errors='coerce')
df['days_until_expiry'] = (df['expiry_date'] - df['date']).dt.days
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

#Dropping original date columns
df = df.drop(['date', 'expiry_date', 'product'], axis=1)

if df['discount'].dtype == 'object':
    df['discount'] = df['discount'].str.replace('%', '', regex=False).astype(float)

#Features and target
X = df.drop('discount', axis=1)
y = df['discount']

#Classification target (example: high discount >= 20%)
y_class = (y >= 20).astype(int)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_class, y_test_class = train_test_split(y_class, test_size=0.2, random_state=42)

#Preprocessing
num_features = ['shelf_life', 'days_until_expiry', 'day_of_week', 'month']
cat_features = ['category', 'holiday', 'event']

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), #Fills missing values in numeric columns with the median
    ('scaler', StandardScaler()) #Scales numeric columns so that everything is on a similar range
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), #Replace missing categories with the word "missing"
    ('onehot', OneHotEncoder(handle_unknown='ignore')) #Turns categories into binary columns
])

#Combines both numeris and categorical preprocessing into one transformer
preprocessed_feat = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ])

#Saving the test set so we can easily reload it later
joblib.dump((X_test, y_test), 'models/test_data.pkl') #Useful when we want to evaluate models without re-splitting every time

#Building a pipeline that first preprocesses the data and then appies the given model
def train_eval_mod(mod, mod_name):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessed_feat),
        ('regressor', mod)
    ])
    pipeline.fit(X_train, y_train) #Training the model
    y_pred = pipeline.predict(X_test) #Using the trained pipeline to predict discounts

    #Classification metrics (using threshold 15)
    y_pred_class = (y_pred >= 15).astype(int)
    acc = accuracy_score(y_test_class, y_pred_class)
    print(f"\n{mod_name} Classification Performance:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision_score(y_test_class, y_pred_class):.4f}")
    print(f"Recall: {recall_score(y_test_class, y_pred_class):.4f}")
    print(f"F1-Score: {f1_score(y_test_class, y_pred_class):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test_class, y_pred_class))
    
    #Regression metrics
    print(f"\n{mod_name} Regression Performance:")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    
    joblib.dump(pipeline, f'models/{mod_name}_pipeline.pkl') #Saving the whole pipeline (preprocessing and model) for reusing it 
    return pipeline,acc

#Training models
ranfor, ranfor_acc = train_eval_mod(RandomForestRegressor(n_estimators=100, random_state=42), "RandomForest")
xgboost, xgb_acc = train_eval_mod(xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100), "XGBoost")

import matplotlib.pyplot as plt

model_names = ["RandomForest", "XGBoost"]
accuracies = [ranfor_acc, xgb_acc]

plt.figure(figsize=(6, 4))
plt.bar(model_names, accuracies, color=["skyblue", "pink"])
plt.ylim(0, 1) #Y-axis from 0 to 1 with step 0.1
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.savefig("accuracy_comparison.png", dpi=300, bbox_inches="tight") #Saving the png
plt.show() #Displaying the graph



