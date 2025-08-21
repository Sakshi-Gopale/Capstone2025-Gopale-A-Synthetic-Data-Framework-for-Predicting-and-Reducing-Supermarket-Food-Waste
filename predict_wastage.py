#!/usr/bin/env python3
# predict_wastage.py — Use a trained model + preprocessor to predict wastage.

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

try:
    from xgboost import XGBRegressor  # noqa: F401
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def add_time_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
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


def add_lag_rolling_with_history(future_df: pd.DataFrame, history_df: pd.DataFrame,
                                 date_col: str, id_cols, target: str,
                                 lags=(1, 7, 28), rolls=(7, 28)) -> pd.DataFrame:
    if history_df is None or not date_col or not id_cols:
        return future_df
    df = pd.concat([history_df, future_df], ignore_index=True, sort=False)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(id_cols + [date_col])
    grp = df.groupby(id_cols, sort=False)
    for L in lags:
        df[f"lag_{target}_{L}"] = grp[target].shift(L)
    for W in rolls:
        df[f"rmean_{target}_{W}"] = grp[target].rolling(W, min_periods=max(1, W//2))[target].mean().reset_index(level=id_cols, drop=True)
    key_cols = id_cols + [date_col] if date_col else id_cols
    future_df = future_df.copy()
    future_df["_merge_key"] = np.arange(len(future_df))
    merged = df.merge(
        future_df[key_cols + ["_merge_key"]] if date_col else future_df[id_cols + ["_merge_key"]],
        on=key_cols, how="right"
    )
    merged = merged.sort_values("_merge_key").drop(columns=["_merge_key"])
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", required=True, choices=["rf", "xgb", "dl"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--preprocessor", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--manual_csv", required=True)
    p.add_argument("--history_csv", required=False)
    p.add_argument("--out", default="predictions.csv")
    args = p.parse_args()

    pre = joblib.load(args.preprocessor)
    schema = json.loads(Path(args.schema).read_text())

    Xf = pd.read_csv(args.manual_csv)
    Xh = pd.read_csv(args.history_csv) if args.history_csv else None

    date_col = schema.get("date_col")
    if date_col and date_col in Xf.columns:
        Xf = add_time_parts(Xf, date_col)
        if Xh is not None and date_col in Xh.columns:
            Xh = add_time_parts(Xh, date_col)

    id_cols = schema.get("id_cols") or []
    target = schema.get("target")

    generated_lags = [c for c in schema.get("generated_lag_parts", []) if isinstance(c, str)]
    if generated_lags:
        if Xh is None:
            print("[WARN] Model trained with lag/rolling features but --history_csv not provided. Imputing; accuracy may degrade.")
        else:
            if target not in Xh.columns:
                raise SystemExit(f"History file must contain the target column '{target}' to compute lags.")
            Xf = add_lag_rolling_with_history(Xf, Xh, date_col=date_col, id_cols=id_cols, target=target)

    expected_cols = pre.feature_names_in_.tolist() if hasattr(pre, "feature_names_in_") else None
    if expected_cols:
        for c in expected_cols:
            if c not in Xf.columns:
                Xf[c] = np.nan
        Xf = Xf[expected_cols]

    Xt = pre.transform(Xf)

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
    out_df["predicted_wastage"] = preds.astype(int)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions to {out_path}")


if __name__ == "__main__":
    main()
