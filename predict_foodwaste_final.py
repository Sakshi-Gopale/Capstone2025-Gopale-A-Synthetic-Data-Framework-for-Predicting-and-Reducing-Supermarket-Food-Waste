import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

try:
    import tensorflow as tf  #Imported but not used directly here
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
try:
    from xgboost import XGBRegressor  #Imported but not used directly here
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

def add_time_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame: #Expanding a datetime column into multiple useful time-based features.
    df = df.copy() #Work on a copy to avoid modifying the original dataframe
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dow"] = df[date_col].dt.dayofweek
        df["weekofyear"] = df[date_col].dt.isocalendar().week.astype("Int64")
        df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
        df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
        df["quarter"] = df[date_col].dt.quarter
    return df

#Creating lag and rolling mean features for a future dataset using historical data.
def add_lag_rolling_with_history(future_df: pd.DataFrame, history_df: pd.DataFrame,
                                 date_col: str, id_cols, target: str,
                                 lags=(1, 7, 28), rolls=(7, 28)) -> pd.DataFrame:
    if history_df is None or not date_col or not id_cols:
        return future_df
    #Combining history and future to ensure lags/rolling can reference past values
    df = pd.concat([history_df, future_df], ignore_index=True, sort=False)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(id_cols + [date_col]) #Sorting by group and date for proper lag/rolling calculation
    grp = df.groupby(id_cols, sort=False)
    #Creating lag features
    for L in lags:
        df[f"lag_{target}_{L}"] = grp[target].shift(L)
    #Creating rolling mean features
    for W in rolls:
        df[f"rmean_{target}_{W}"] = grp[target].rolling(W, min_periods=max(1, W//2))[target].mean().reset_index(level=id_cols, drop=True)
    #Merging only the future_df rows with calculated features
    key_cols = id_cols + [date_col] if date_col else id_cols
    future_df = future_df.copy()
    future_df["_merge_key"] = np.arange(len(future_df))
    merged = df.merge(
        future_df[key_cols + ["_merge_key"]] if date_col else future_df[id_cols + ["_merge_key"]],
        on=key_cols, how="right"
    )
    #Restoring original order and drop the temporary key
    merged = merged.sort_values("_merge_key").drop(columns=["_merge_key"])
    return merged

def main():
    p = argparse.ArgumentParser() #Parsing command-line arguments
    p.add_argument("--model_type", required=False, default="xgb", choices=["rf", "xgb", "dl"])
    p.add_argument("--model_path", required=False, default="xgb.joblib")
    p.add_argument("--preprocessor", required=False, default="preprocessor.pkl")
    p.add_argument("--schema", required=False, default="schema.json")
    p.add_argument("--manual_csv", required=False,  )
    p.add_argument("--history_csv", required=False)
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    pre = joblib.load(args.preprocessor) #Loading preprocessing pipeline
    schema = json.loads(Path(args.schema).read_text()) #Loading schema metadata

    Xf = pd.read_csv(args.manual_csv) #Loading new input data
    Xh = pd.read_csv(args.history_csv) if args.history_csv else None  #Loading historical data if provided

    date_col = schema.get("date_col")
    if date_col and date_col in Xf.columns:
        Xf = add_time_parts(Xf, date_col) #Adding datetime features to future data
        if Xh is not None and date_col in Xh.columns: #Adding datetime features to history data
            Xh = add_time_parts(Xh, date_col)

    id_cols = schema.get("id_cols") or [] #ID columns for grouping
    target = schema.get("target") #Target column name

    generated_lags = [c for c in schema.get("generated_lag_parts", []) if isinstance(c, str)]
    if generated_lags:
        if Xh is None:
            print("[WARN] Model trained with lag/rolling features but --history_csv not provided. Imputing; accuracy may degrade.")
        else:
            if target not in Xh.columns:
                raise SystemExit(f"History file must contain the target column '{target}' to compute lags.") #Ensuring target exists
            Xf = add_lag_rolling_with_history(Xf, Xh, date_col=date_col, id_cols=id_cols, target=target) #Computing lag/rolling features

    expected_cols = pre.feature_names_in_.tolist() if hasattr(pre, "feature_names_in_") else None  #Getting expected feature order
    if expected_cols:
        for c in expected_cols:
            if c not in Xf.columns:
                Xf[c] = np.nan
        Xf = Xf[expected_cols] #Reordering columns

    Xt = pre.transform(Xf)

    #Loading model and predicting
    if args.model_type == "rf":
        model = joblib.load(args.model_path)
        preds = model.predict(Xt)
    elif args.model_type == "xgb":
        model = joblib.load(args.model_path)
        preds = model.predict(Xt)
    else:
        if not TF_AVAILABLE:
            raise SystemExit("TensorFlow/Keras not installed; cannot load 'dl' model.")
        model = tf.keras.models.load_model(args.model_path)
        preds = model.predict(Xt, verbose=0).ravel()

    out_path = Path(args.out)
    out_df = Xf.copy()
    out_df["predicted_wastage"] = preds #Adding predictions column
    out_df.to_csv(out_path, index=False) #Saving predictions CSV
    print(f"[OK] Wrote predictions to {out_path}")

if __name__ == "__main__":
    main()
