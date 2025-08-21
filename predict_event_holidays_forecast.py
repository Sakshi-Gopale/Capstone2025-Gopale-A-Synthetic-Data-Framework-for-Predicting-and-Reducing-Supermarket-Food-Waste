#!/usr/bin/env python3
# event_window_forecast_manual_start.py

import json
import re
import warnings
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=UserWarning)

def save_actual_vs_pred_plot(dates, y_true, y_pred, product_id, event_name, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Quantity Sold")
    plt.title(f"Actual vs Predicted — {product_id} — {event_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def normalize_params(tp: dict) -> dict:
    return {
        "test_size": float(tp.get("test_size", 0.1)),
        "n_estimators": int(tp.get("estimators", 120)),
        "learning_rate": float(tp.get("learning_rate", 0.01)),
        "max_depth": int(tp.get("max_depth", 5)),
        "subsample": float(tp.get("subsample", 0.6)),
        "colsample_bytree": float(tp.get("colsample_bytree", 0.9)),
        "objective": "reg:squarederror",
        "n_jobs": -1,
        "random_state": 42,
    }

def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = np.nan
    mape = safe_mape(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def _canonize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def _set_year_if_provided(date_series: pd.Series, year_series: pd.Series) -> pd.Series:
    out = date_series.copy()
    mask = date_series.notna() & year_series.notna()
    if mask.any():
        tmp = pd.DataFrame({
            "year": year_series[mask].astype(int),
            "month": date_series[mask].dt.month,
            "day": date_series[mask].dt.day
        })
        out.loc[mask] = pd.to_datetime(tmp, errors="coerce")
    return out

def _load_calendar_flex(calendar_path: str) -> pd.DataFrame:
    
    if calendar_path.lower().endswith((".xlsx", ".xls")):
        cal = pd.read_excel(calendar_path)
    else:
        cal = pd.read_csv(calendar_path)

    canon_cols = {_canonize(c): c for c in cal.columns}

    ev_candidates = ["event_name", "event", "holiday", "holidays", "holidays_event", "holidays_event_name"]
    st_candidates = ["start_date", "start", "start_dt", "from_date", "from", "startdate"]
    en_candidates = ["end_date", "end", "end_dt", "to_date", "to", "enddate"]
    dur_candidates = ["duration_days", "duration", "duration_in_days", "duration__days", "duration_days_", "days", "no_of_days"]
    yr_candidates = ["year"]

    ev_col = canon_cols.get("holidays_event", None) or next((canon_cols[k] for k in ev_candidates if k in canon_cols), None)
    st_col = next((canon_cols[k] for k in st_candidates if k in canon_cols), None)
    en_col = next((canon_cols[k] for k in en_candidates if k in canon_cols), None)
    du_col = next((canon_cols[k] for k in dur_candidates if k in canon_cols), None)
    yr_col = next((canon_cols[k] for k in yr_candidates if k in canon_cols), None)

    if ev_col is None or st_col is None:
        raise ValueError(f"Calendar must have an event name and start date column. Got: {list(cal.columns)}")

    rename_map = {ev_col: "event_name", st_col: "start_date"}
    if en_col: rename_map[en_col] = "end_date"
    if du_col: rename_map[du_col] = "duration_days"
    if yr_col: rename_map[yr_col] = "year"
    cal = cal.rename(columns=rename_map)

    cal["event_name"] = cal["event_name"].astype(str).str.strip()
    cal["start_date"] = pd.to_datetime(cal["start_date"], dayfirst=True, errors="coerce")
    if "end_date" in cal.columns:
        cal["end_date"] = pd.to_datetime(cal["end_date"], dayfirst=True, errors="coerce")

    if "year" in cal.columns:
        cal["start_date"] = _set_year_if_provided(cal["start_date"], cal["year"])
        if "end_date" in cal.columns:
            cal["end_date"] = _set_year_if_provided(cal["end_date"], cal["year"])

    #Ensuring duration column
    if "duration_days" not in cal.columns:
        cal["duration_days"] = np.nan

    #If missing durations, compute from dates where possible (inclusive)
    if "end_date" in cal.columns:
        has_both = cal["start_date"].notna() & cal["end_date"].notna()
        cal.loc[has_both, "duration_days"] = (cal.loc[has_both, "end_date"] - cal.loc[has_both, "start_date"]).dt.days + 1

    cal["duration_days"] = pd.to_numeric(cal["duration_days"], errors="coerce").fillna(1).astype(int)

    if cal["start_date"].isna().any():
        bad = cal[cal["start_date"].isna()]
        raise ValueError(f"Some calendar rows have invalid start dates:\n{bad}")

    return cal[["event_name", "start_date", "end_date", "duration_days"]]

def _pick_event_duration_days(cal: pd.DataFrame, ev_lc: str) -> int:
    
    dur = cal.loc[cal["event_name"].str.lower().str.strip() == ev_lc, "duration_days"]
    if len(dur) == 0:
        return 1
    mode_vals = dur.mode(dropna=True)
    if len(mode_vals) > 0:
        return int(mode_vals.iloc[0])
    return int(round(float(dur.median()))) if len(dur) > 0 else 1

def forecast_for_event(
    product_id: str,
    event_name: str,
    calendar_path: str = "HolidaysEventsEntries.xlsx",
    sales_path: str = "Sales_with_HolidayEvents_true.csv",
    params_path: Optional[str] = "training_parameters_event_holiday.json",
    out_csv: Optional[str] = None,
    out_metrics_csv: Optional[str] = None,
    manual_start_date: Optional[str] = None,  #optional override
    return_metrics: bool = False,             #NEW: return (forecast_df, metrics_row)
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, object]]:
    
    #Parameters
    PARAMS_BY_PID: Dict[str, dict] = {}
    GLOBAL_DEFAULTS = normalize_params({})
    if params_path:
        try:
            with open(params_path, "r") as f:
                raw_params = json.load(f)
            PARAMS_BY_PID = {
                item["product_id"]: normalize_params(item.get("tran_parameter", {}))
                for item in raw_params
            }
        except FileNotFoundError:
            print(f"[info] Params file '{params_path}' not found. Using defaults for all products.")

    #Sales
    df = pd.read_csv(sales_path)
    needed_cols = {"product_id", "date", "quantity_sold", "event_name"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Sales file is missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], dayfirst=True, errors="coerce")
    if "discount" in df.columns:
        df["discount"] = df["discount"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    if "event" in df.columns:
        df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0.0)
    if "holiday" in df.columns:
        df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").fillna(0.0)

    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    if "shelf_life" not in df.columns:
        df["shelf_life"] = 30.0

    if "category" in df.columns:
        le_cat = LabelEncoder()
        df["category"] = df["category"].astype(str).fillna("unknown")
        df["category_enc"] = le_cat.fit_transform(df["category"])
    else:
        df["category_enc"] = 0

    df["event_name"] = df["event_name"].fillna("none").astype(str)
    df["event_name_lc"] = df["event_name"].str.lower().str.strip()
    le_event = LabelEncoder()
    df["event_name_enc"] = le_event.fit_transform(df["event_name_lc"])
    EVENT_NAME_TO_CODE = {name: int(code) for code, name in enumerate(le_event.classes_)}
    CODE_TO_EVENT_NAME = {v: k for k, v in EVENT_NAME_TO_CODE.items()}

    feature_cols = ["dayofweek", "month", "day", "shelf_life", "category_enc", "event_name_enc"]
    if "discount" in df.columns:
        feature_cols.append("discount")
    if "event" in df.columns:
        feature_cols.append("event")
    if "holiday" in df.columns:
        feature_cols.append("holiday")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #Product slicing
    g = df[df["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    if len(g) < 5:
        raise ValueError(f"Not enough history (<5 rows) for product_id='{product_id}'.")

    p = PARAMS_BY_PID.get(product_id, GLOBAL_DEFAULTS)
    test_size = min(max(p.get("test_size", 0.1), 0.05), 0.5)

    X = g[feature_cols].copy()
    y = g["quantity_sold"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    model = XGBRegressor(
        n_estimators=p["n_estimators"],
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        objective=p["objective"],
        n_jobs=p["n_jobs"],
        random_state=p["random_state"],
    )
    model.fit(X_train, y_train)

    #Test metrics
    y_pred = model.predict(X_test)
    overall = compute_metrics(y_test, y_pred)

    test_len = len(y_test)
    test_dates = g["date"].iloc[-test_len:]  #Aligns with X_test (shuffle=False)
    plot_file = f"{product_id}_{_canonize(event_name)}_actual_vs_pred.png"
    if PRODUCT_ID is product_id:
        save_actual_vs_pred_plot(
            dates=test_dates, y_true=np.asarray(y_test), y_pred=np.asarray(y_pred),
            product_id=product_id, event_name=event_name, out_path=plot_file
        )
        #print(f"[{product_id}] saved plot: {plot_file}")

    ev_lc = str(event_name).lower().strip()
    #Mapping test rows to event names for event-specific metrics
    test_df = X_test.copy()
    test_df = test_df.assign(
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        event_name_lc=test_df["event_name_enc"].round().astype(int).map(CODE_TO_EVENT_NAME)
    )
    sub = test_df[test_df["event_name_lc"] == ev_lc]
    if len(sub) >= 1:
        event_metrics = compute_metrics(sub["y_true"], sub["y_pred"])
    else:
        event_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}

    #print(f"[{product_id}] overall test metrics:", overall)
    #print(f"[{product_id}] '{event_name}' test metrics:", event_metrics, f"(n={len(sub)} rows in test)")

    #Calendar and duration
    cal = _load_calendar_flex(calendar_path)
    duration_days = _pick_event_duration_days(cal, ev_lc)
    #print(f"[{product_id}] learned duration for '{event_name}': {duration_days} day(s)")

    #Encoding for this event name (fallback to 'none' if unseen during training)
    if ev_lc in EVENT_NAME_TO_CODE:
        ev_code = EVENT_NAME_TO_CODE[ev_lc]
        event_flag = 1.0 if ev_lc not in ("none", "nan", "") else 0.0
    else:
        ev_code = EVENT_NAME_TO_CODE.get("none", 0)
        event_flag = 1.0  # treat as active event

    event_name_canonical = CODE_TO_EVENT_NAME.get(ev_code, ev_lc)
    #print(f"[{product_id}] event requested='{event_name}' -> encoded={ev_code}, canonical='{event_name_canonical}'")

    #Build event windows
    windows: List[Dict] = []
    if manual_start_date:
        #Manual single window using learned duration
        start_dt = pd.to_datetime(manual_start_date, dayfirst=True, errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"Could not parse manual_start_date='{manual_start_date}'. Use DD/MM/YYYY or YYYY-MM-DD.")
        end_dt = start_dt + pd.Timedelta(days=duration_days - 1)
        windows.append({"start_date": start_dt, "end_date": end_dt})
        #print(f"[{product_id}] manual window: {start_dt.date()} -> {end_dt.date()} ({duration_days} day(s))")
    else:
        #Default: use all windows from calendar for this event
        cal_ev = cal[cal["event_name"].str.lower().str.strip() == ev_lc]
        if cal_ev.empty:
            raise ValueError(f"Event '{event_name}' not found in calendar.")
        for _, r in cal_ev.iterrows():
            sdt = r["start_date"]
            edt = r["end_date"] if pd.notna(r["end_date"]) else r["start_date"] + pd.Timedelta(days=int(r["duration_days"]) - 1)
            windows.append({"start_date": sdt, "end_date": edt})
        #print(f"[{product_id}] using {len(windows)} window(s) from calendar for '{event_name}'.")

    #Forecast each window
    base_vals = {
        "shelf_life": float(g["shelf_life"].iloc[-1]),
        "category_enc": int(g["category_enc"].iloc[-1]),
        "discount": float(g["discount"].iloc[-1]) if "discount" in g.columns else 0.0,
        "holiday": float(g["holiday"].iloc[-1]) if "holiday" in g.columns else 0.0,
    }

    all_frames: List[pd.DataFrame] = []
    for idx, win in enumerate(windows, start=1):
        dts = pd.date_range(win["start_date"], win["end_date"], freq="D")
        if len(dts) == 0:
            continue

        future_df = pd.DataFrame({
            "date": dts,
            "dayofweek": dts.dayofweek,
            "month": dts.month,
            "day": dts.day,
            "shelf_life": base_vals["shelf_life"],
            "category_enc": base_vals["category_enc"],
            "event_name_enc": ev_code,
        })
        if "discount" in feature_cols:
            future_df["discount"] = base_vals["discount"]
        if "event" in feature_cols:
            future_df["event"] = event_flag
        if "holiday" in feature_cols:
            future_df["holiday"] = 0.0

        preds = model.predict(future_df[feature_cols])

        out = pd.DataFrame({
            "product_id": product_id,
            "event_name": event_name,
            "window_index": idx,
            "window_start": win["start_date"].date(),
            "window_end": win["end_date"].date(),
            "date": dts,
            "forecast_quantity_sold": np.round(preds, 5),

            #Attach accuracy columns (same values repeated on all rows)
            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
            "R2": overall["R2"],
            "MAPE": overall["MAPE"],
        })
        all_frames.append(out)

    if not all_frames:
        raise ValueError(f"No valid dates generated for event '{event_name}'.")

    result = pd.concat(all_frames, ignore_index=True)

    #Per-window totals
    totals = (
        result.groupby(
            ["product_id", "window_index", "window_start", "window_end"]
        )["forecast_quantity_sold"]
        .sum().reset_index()
        .rename(columns={"forecast_quantity_sold": "window_total_forecast"})
    )
    result = result.merge(
        totals,
        on=["product_id", "window_index", "window_start", "window_end"],
        how="left",
    )

    #Build metrics row (used for optional CSV and return)
    metrics_row = {
        "product_id": product_id,
        "event_name": event_name,
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        "MAPE": overall["MAPE"],
    }

    #Optional saves
    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"Saved forecasts: {out_csv}")

    if out_metrics_csv:
        pd.DataFrame([metrics_row]).to_csv(out_metrics_csv, index=False)
        print(f"Saved metrics: {out_metrics_csv}")

    if return_metrics:
        return result, metrics_row
    return result

#Product summary
def summarize_metrics_all_products(
    event_name: str,
    calendar_path: str = "HolidaysEventsEntries.xlsx",
    sales_path: str = "Sales_with_HolidayEvents_true.csv",
    params_path: Optional[str] = "training_parameters_event_holiday.json",
    manual_start_date: Optional[str] = None,
    out_csv: Optional[str] = "ALL_PRODUCTS_metrics.csv",
) -> pd.DataFrame:
    
    sales = pd.read_csv(sales_path)
    if "product_id" not in sales.columns:
        raise ValueError("sales_path must contain a 'product_id' column.")
    products = (
        sales["product_id"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    )

    rows = []
    for pid in products:
        try:
            _, mrow = forecast_for_event(
                product_id=pid,
                event_name=event_name,
                calendar_path=calendar_path,
                sales_path=sales_path,
                params_path=params_path,
                out_csv=None,                 #Per-product forecasts not saved here
                out_metrics_csv=None,         #Collecting in one table
                manual_start_date=manual_start_date,
                return_metrics=True,          #Pulling metrics row back
            )
            rows.append(mrow)
        except Exception as e:
            rows.append({
                "product_id": pid,
                "event_name_requested": event_name,
                "error": str(e),
            })

    summary = pd.DataFrame(rows)

    #Putting the important metric columns side-by-side in a clean order
    desired_cols = [
        "product_id", "event_name",        
        "MAE", "RMSE", "R2", "MAPE",
    ]

    for c in desired_cols:
        if c not in summary.columns:
            summary[c] = np.nan

    ordered = [c for c in desired_cols if c in summary.columns]
    if "error" in summary.columns:
        ordered = ordered + ["error"]
    extra = [c for c in summary.columns if c not in ordered]
    summary = summary[ordered + extra]

    if out_csv:
        summary.to_csv(out_csv, index=False)
        print(f"[all-products] saved metrics summary -> {out_csv}")

    return summary

#Example
if __name__ == "__main__":
    CAL_PATH = "HolidaysEventsEntries.xlsx"
    SALES_PATH = "Sales_with_HolidayEvents_true.csv"
    PARAMS_PATH = "training_parameters_event_holiday.json"  

    #Example 1: single product with manual window 
    PRODUCT_ID = "PN002"  #Change as needed
    EVENT_NAME = "Galway International Arts Festival"  #Change as needed
    MANUAL_START = "17/05/2026"  #DD/MM/YYYY or YYYY-MM-DD; set to None to use calendar windows

    out_file = f"{PRODUCT_ID}_{_canonize(EVENT_NAME)}_manual_start_forecast.csv"
    out_metrics = f"{PRODUCT_ID}_{_canonize(EVENT_NAME)}_manual_start_metrics.csv"

    df_out = forecast_for_event(
        product_id=PRODUCT_ID,
        event_name=EVENT_NAME,
        calendar_path=CAL_PATH,
        sales_path=SALES_PATH,
        params_path=PARAMS_PATH,
        out_csv=out_file,
        out_metrics_csv=out_metrics,
        manual_start_date=MANUAL_START,
    )
    print(df_out.head(10))

    #Example 2: metrics summary for ALL products
    all_metrics_file = f"ALL_PRODUCTS_{_canonize(EVENT_NAME)}_metrics.csv"
    summary_df = summarize_metrics_all_products(
        event_name=EVENT_NAME,
        calendar_path=CAL_PATH,
        sales_path=SALES_PATH,
        params_path=PARAMS_PATH,
        manual_start_date=MANUAL_START,     
        out_csv=all_metrics_file,
    )
    print(summary_df.head(10))
    
    
