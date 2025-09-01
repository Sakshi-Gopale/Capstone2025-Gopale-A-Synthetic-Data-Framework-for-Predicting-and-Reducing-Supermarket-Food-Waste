import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

#Creating models folder
os.makedirs('models', exist_ok=True)

#Loading data
df = pd.read_csv("data/sales_weather_data.csv")

#Converting date fields
df['date'] = pd.to_datetime(df['date'])
df['expiry_date'] = pd.to_datetime(df['expiry_date'])

#Feature Engineering
df['days_to_expiry'] = (df['expiry_date'] - df['date']).dt.days
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek

#Cleaning percentage fields if any
def clean_percentage(val):
    if isinstance(val, str) and '%' in val:
        return float(val.replace('%', ''))
    return float(val)

df['discount'] = df['discount'].apply(clean_percentage)

#Defining features and target
features = df.drop(columns=['product_id', 'product', 'category', 'date', 'expiry_date', 'quantity_sold'])
target = df['quantity_sold']

#One-hot encoding for categorical variables
features = pd.get_dummies(features, columns=['holiday', 'event'], drop_first=True)

#Splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.5, random_state=42)

#Training Random Forest
rf_model = RandomForestRegressor(n_estimators=8, random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/rf_model.pkl')

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

#Training Linear Regression 
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
joblib.dump(lin_model, 'models/linreg_model.pkl') #Saving the trained linear regression model to a file using joblib

#Printing Linear Regression metrics
print_metrics(y_test, lin_pred, "Linear Regression")

def regression_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred) #Calculating mean absolute error
    denom = np.mean(np.abs(y_true)) + 1e-8 #Normalizing by average true value (avoid div by 0)
    acc = 1.0 - (mae / denom) #Converting error to accuracy
    return float(np.clip(acc, 0.0, 1.0)) #Ensuring accuracy is between 0 and 1

#Collecting metrics for both models
lin_acc = regression_accuracy(y_test, lin_pred)
lin_r2  = r2_score(y_test, lin_pred)

rf_acc  = regression_accuracy(y_test, rf_pred)
rf_r2   = r2_score(y_test, rf_pred)

#Preparing grouped bar chart
metrics_labels = ["R²"]
linear_values  = [lin_r2]
rf_values      = [rf_r2]

x = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
#Plotting bars for Linear Regression
bars_lin = ax.bar(x - width/2, linear_values, width,
                  label='Linear Regression', color='pink')
#Plotting bars for Random Forest
bars_rf  = ax.bar(x + width/2, rf_values, width,
                  label='Random Forest', color='skyblue')

ax.set_xticks(x)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(0, 1.0)
ax.set_ylabel('Score')
ax.set_title('R² Comparison of Linear Regression vs Random Forest')
ax.legend()

plt.tight_layout()
plt.savefig('models/accuracy_r2_comparison.png', dpi=150)
plt.show()

#Printing metrics
print_metrics(y_test, rf_pred, "Random Forest")
print("\n Model trained successfully!")
