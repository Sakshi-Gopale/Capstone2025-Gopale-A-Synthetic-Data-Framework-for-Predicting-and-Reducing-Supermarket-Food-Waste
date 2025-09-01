import json
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

#Opneneing and loading the JSON file
with open("training_parameters_by_product.json", "r") as f:
    raw_params = json.load(f)

#Defining a function to normalize and map JSON parameters into the expected format for XGBoost / sklearn models
def norm_params(tp: dict) -> dict:
    #Mapping your JSON keys to XGBoost / sklearn names and provide defaults
    return {
        "test_size": float(tp.get("test_size", 0.1)),  #Default test size
        "n_estimators": int(tp.get("estimators", 120)),   #XGBoost expects 'n_estimators', so mapping JSON key 'estimators' to 'n_estimators'
        "learning_rate": float(tp.get("learning_rate", 0.01)), #Stepping size shrinkage
        "max_depth": int(tp.get("max_depth", 5)), #Maximum depth of each tree (how complex each tree can get)
        "subsample": float(tp.get("subsample", 0.6)), #Fraction of the data used to build each tree
        "colsample_bytree": float(tp.get("colsample_bytree", 0.9)), #Fraction of features (columns) used per tree
        "objective": "reg:squarederror", #Solving a regression problem
        "n_jobs": -1,   #Using all CPU cores available to speed things up
        "random_state": 42,
    }

#Mapping product_id to its training parameters from JSON
PARAMS_BY_PID = {
    item["product_id"]: norm_params(item.get("tran_parameter", {}))
    for item in raw_params
}

#Default parameters if product_id is missing in JSON
GLOBAL_DEFAULTS = norm_params({})

#Helper functions to use later
def safe_mape(y_true, y_pred):
    #Calculating MAPE safely (ignoring zero values in y_true)
    y_true = np.asarray(y_true, dtype=float) 
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0 #Ignoring places where the actual value is zero
    if mask.sum() == 0: #If all actual values are zero, MAPE cannot be computed
        return np.nan #Returning "not a number" to signal it's invalid
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0) #Absolute % error

df = pd.read_csv("Sales_Dataset_Train.csv")

#Preprocessing the data
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df['expiry_date'] = pd.to_datetime(df['expiry_date'], dayfirst=True)
df['discount'] = df['discount'].str.replace('%', '', regex=False).astype(float) / 100

#Filling event/holiday columns
if 'event' in df.columns:
    df['event'] = df['event'].fillna(0)
if 'holiday' in df.columns:
    df['holiday'] = df['holiday'].fillna(0)

#Extracting time-based features
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

#Encoding category column into numeric values
le_cat = LabelEncoder()
df['cat_enc'] = le_cat.fit_transform(df['category'])

#Defining features for training
feature_cols = [
    'dayofweek', 'month', 'day',
    'unit_price', 'discount', 'discount_price',
    'shelf_life', 'cat_enc'
]

#Training per product using JSON parameters
results = {}
forecast_duration = 10 

for pid, group in df.groupby("product_id"): #Looping over each product
    group = group.sort_values("date") #Sorting by date

    #Skipping products with insufficient history
    if len(group) < 5:
        continue

    p = PARAMS_BY_PID.get(pid, GLOBAL_DEFAULTS) #Pulling parameters for this product

    X = group[feature_cols] #Features
    y = group['quantity_sold'] #Targets

    test_size = p.get("test_size", 0.1) #Using test_size from JSON
    test_size = min(max(test_size, 0.05), 0.5) #Bounding it slightly to avoid edge cases

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False #No shuffle for time series
    )

    #Building the model from JSON parameters
    model = XGBRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        objective=p["objective"],
        n_jobs=p["n_jobs"],
        random_state=p["random_state"]
    )

    model.fit(X_train, y_train) #Training the model on this product's data

    #Evaluating performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test, y_pred)
    mape = safe_mape(y_test, y_pred)

    #Forecasting next N days 
    future_dates = pd.date_range(pd.to_datetime("2025-09-01") + pd.Timedelta(days=1), periods=forecast_duration) #Date can be changed 

#Building a future dataframe using the last known product attributes
    future_df = pd.DataFrame({
        'date': future_dates,
        'dayofweek': future_dates.dayofweek,
        'month': future_dates.month,
        'day': future_dates.day,
        'unit_price': group['unit_price'].iloc[-1], #Using last known unit price
        'discount': group['discount'].iloc[-1], #Using last known discount
        'discount_price': group['discount_price'].iloc[-1], #Computing last known discount price 
        'shelf_life': group['shelf_life'].iloc[-1], #Using last known shelf life
        'cat_enc': group['cat_enc'].iloc[-1], #Last known category encoded value
    })

    forecast = model.predict(future_df[feature_cols]) #Forecasting future quantites

    #Storing forecast results
    forecast_df = pd.DataFrame({
        'date': future_dates,
        'product_id': pid,
        'forecast_quantity': np.round(forecast, 0).astype(int),
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    })

    #Saving evaluation metrics, forecast, and parameters used for this product
    results[pid] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
        "forecast": forecast_df,
        "params_used": p
    }

#Saving all product forecasts
if results:
    all_forecasts = pd.concat([res['forecast'] for res in results.values()], ignore_index=True)
    all_forecasts.to_csv("products_forecast.csv", index=False)
    print("\nSaved: products_forecast.csv") #Confirming save
else:
    print("No products were trained; check data or minimum history length.") #Fallback message

#Picking a product to visualize 
sample_pid = "PN002"
res = results[sample_pid]

#Plotting Actual vs Predicted sale 
prod_df = df[df['product_id'] == sample_pid].sort_values("date")  #Full product history

X = prod_df[feature_cols]
y = prod_df['quantity_sold']
dates = prod_df['date']

test_size = res['params_used']['test_size']
test_size = min(max(test_size, 0.05), 0.5)

#Calculating split index manually (number of rows to use for training)
split_idx = int(len(X) * (1 - test_size))

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:] #Splitting features (X) into training and testing sets
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:] #Splitting target/labels (y) into training and testing sets
test_dates = dates.iloc[split_idx:]  #Keeping the dates corresponding to the test set

#Training model again with same params
model = XGBRegressor(**res['params_used'])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#Plotting with real dates
plt.figure(figsize=(10, 5))
plt.plot(test_dates, y_test.values, label="Actual")
plt.plot(test_dates, y_pred, label="Predicted", linestyle="--")
plt.title(f"Actual vs Predicted Sales for Product {sample_pid}")
plt.xlabel("Date")
plt.ylabel("Quantity Sold")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.close()

#Forecasting of duration given
forecast_df = res["forecast"]

#Plotting graph for forecast
plt.figure(figsize=(10, 5))
plt.plot(forecast_df["date"], forecast_df["forecast_quantity"], marker="o")
plt.title(f"10-Day Forecast for Product {sample_pid}")
plt.xlabel("Date")
plt.ylabel("Forecast Quantity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("forecast_10days.png", dpi=300)
plt.close()

#Saving the plots in png
print("\nSaved plots: actual_vs_predicted.png and forecast_30days.png")
