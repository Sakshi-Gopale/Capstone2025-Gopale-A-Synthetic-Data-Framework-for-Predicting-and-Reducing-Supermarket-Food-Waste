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
    #Plotting a line chart to compare actual vs predicted sales
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
        "n_estimators": int(tp.get("estimators", 120)), #No. of trees in XGB
        "learning_rate": float(tp.get("learning_rate", 0.01)),
        "max_depth": int(tp.get("max_depth", 5)),
        "subsample": float(tp.get("subsample", 0.6)), #Percent of samples used per tree
        "colsample_bytree": float(tp.get("colsample_bytree", 0.6)), #Percent of features used per tree
        "objective": "reg:squarederror", #Standard regression objective
        "n_jobs": -1,
        "random_state": 42,
    }

def safe_mape(y_true, y_pred):
    #Calculating mean absolute percentage error, but avoiding division by zero errors
    #and skipping the rows where the ture value is 0 so the metric does not explode
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0: #Edge case: if all true values are 0
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

def compute_metrics(y_true, y_pred) -> Dict[str, float]:
    #Putting together multiple preformance metrics to evaluate the model with one call
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    try:
        r2 = float(r2_score(y_true, y_pred)) #Variance
    except Exception:
        r2 = np.nan
    mape = safe_mape(y_true, y_pred) #Relative error
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def _canonize(s: str) -> str:
    #converting any string into a clean identifier
    s = s.strip().lower()
    s = re.sub(r"[^\w]+", "_", s) #Replacing non-alphanumeric chars with undescores
    s = re.sub(r"_+", "_", s).strip("_") #Avoiding multiple uncerscores
    return s

def _set_year_if_provided(date_series: pd.Series, year_series: pd.Series) -> pd.Series:
    #If the dates don't have a year this function patches in the year from another column
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
    #Accepts columns (any casing): 'Holidays/Event' (or 'Event Name'), 'Start Date','End Date' and/or 'Duration (days)', optional 'Year'.
    #Returns: ['event_name', 'start_date', 'end_date', 'duration_days'] (duration filled).
    if calendar_path.lower().endswith((".xlsx", ".xls")): #Loading the file
        cal = pd.read_excel(calendar_path)
    else:
        cal = pd.read_csv(calendar_path)

    canon_cols = {_canonize(c): c for c in cal.columns} #Mapping column names to a standard form

    #Possible column names people might use for each required field
    ev_candidates  = ["event_name", "event", "holiday", "holidays", "holidays_event", "holidays_event_name"]
    st_candidates  = ["start_date", "start", "start_dt", "from_date", "from", "startdate"]
    en_candidates  = ["end_date", "end", "end_dt", "to_date", "to", "enddate"]
    dur_candidates = ["duration_days", "duration", "duration_in_days", "duration__days", "duration_days_", "days", "no_of_days"]
    yr_candidates  = ["year"]

    #Trying to find the actual columns from the dataset
    ev_col = canon_cols.get("holidays_event", None) or next((canon_cols[k] for k in ev_candidates if k in canon_cols), None)
    st_col = next((canon_cols[k] for k in st_candidates if k in canon_cols), None)
    en_col = next((canon_cols[k] for k in en_candidates if k in canon_cols), None)
    du_col = next((canon_cols[k] for k in dur_candidates if k in canon_cols), None)
    yr_col = next((canon_cols[k] for k in yr_candidates if k in canon_cols), None)

    #Raising an error if event name and start date is not provided
    if ev_col is None or st_col is None:
        raise ValueError(f"Calendar must have an event name and start date column. Got: {list(cal.columns)}")

    #Renaming the columns to standard names
    rename_map = {ev_col: "event_name", st_col: "start_date"}
    if en_col: rename_map[en_col] = "end_date"
    if du_col: rename_map[du_col] = "duration_days"
    if yr_col: rename_map[yr_col] = "year"
    cal = cal.rename(columns=rename_map)

    #Cleaning the text and converting the dates
    cal["event_name"] = cal["event_name"].astype(str).str.strip()
    cal["start_date"] = pd.to_datetime(cal["start_date"], dayfirst=True, errors="coerce")
    if "end_date" in cal.columns:
        cal["end_date"] = pd.to_datetime(cal["end_date"], dayfirst=True, errors="coerce")

    #If the year column is provided, settign the year for start and end date
    if "year" in cal.columns:
        cal["start_date"] = _set_year_if_provided(cal["start_date"], cal["year"])
        if "end_date" in cal.columns:
            cal["end_date"] = _set_year_if_provided(cal["end_date"], cal["year"])

    #Ensuring the duration column
    if "duration_days" not in cal.columns:
        cal["duration_days"] = np.nan

    #If start and end dates are given, calculating the duration (end-start+1)
    if "end_date" in cal.columns:
        has_both = cal["start_date"].notna() & cal["end_date"].notna()
        cal.loc[has_both, "duration_days"] = (cal.loc[has_both, "end_date"] - cal.loc[has_both, "start_date"]).dt.days + 1

    #Filling the missing duration with 1day
    cal["duration_days"] = pd.to_numeric(cal["duration_days"], errors="coerce").fillna(1).astype(int)

    if cal["start_date"].isna().any():
        bad = cal[cal["start_date"].isna()]
        raise ValueError(f"Some calendar rows have invalid start dates:\n{bad}")

    return cal[["event_name", "start_date", "end_date", "duration_days"]]

def _pick_event_duration_days(cal: pd.DataFrame, ev_lc: str) -> int:
    #Picking a typical duration for an event: 1st choice is mode, if no clear mode then select median and if nothing else then deault to 1 day
    dur = cal.loc[cal["event_name"].str.lower().str.strip() == ev_lc, "duration_days"]
    if len(dur) == 0: #If no records, assume 1 day
        return 1
    mode_vals = dur.mode(dropna=True) #Using most frequent value (mode)
    if len(mode_vals) > 0:
        return int(mode_vals.iloc[0])
    return int(round(float(dur.median()))) if len(dur) > 0 else 1 #Otherwise use median (middle value)

def forecast_for_event(
    product_id: str,
    event_name: str,
    calendar_path: str = "HolidaysEventsDataset.xlsx",
    sales_path: str = "Sales_with_HolidayEvents_true.csv",
    params_path: Optional[str] = "training_parameters_by_product.json",
    out_csv: Optional[str] = None,
    out_metrics_csv: Optional[str] = None,
    manual_start_date: Optional[str] = None,  #Allowing optional overriding of the event start
    return_metrics: bool = False,   #If TRUE, return both forecast+ metrics
) -> pd.DataFrame | tuple[pd.DataFrame, Dict[str, object]]:
    #Forecasting sales for a product only during the event period
    #If manual start date is given, the end date is calculated from the typical event duration
    
    #Loading model parameters
    PARAMS_BY_PID: Dict[str, dict] = {}
    GLOBAL_DEFAULTS = normalize_params({})
    if params_path:
        try:
            with open(params_path, "r") as f:
                raw_params = json.load(f)
            #Mapping product IDs to their training parameters
            PARAMS_BY_PID = {
                item["product_id"]: normalize_params(item.get("tran_parameter", {}))
                for item in raw_params
            }
        except FileNotFoundError:
            print(f"[info] Params file '{params_path}' not found. Using defaults for all products.")

    #Loading and preparing sales data
    df = pd.read_csv(sales_path)
    needed_cols = {"product_id", "date", "quantity_sold", "event_name"} #Ensuring the key columns are present
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(f"Sales file is missing columns: {missing}")

    #Converting dates and cleaning optional columns
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    if "expiry_date" in df.columns:
        df["expiry_date"] = pd.to_datetime(df["expiry_date"], dayfirst=True, errors="coerce")
    #Converting discount from 20% to 0.20
    if "discount" in df.columns:
        df["discount"] = df["discount"].astype(str).str.replace("%", "", regex=False).astype(float) / 100.0
    #Ensuring event/holiday columns are numeric (0/1 flags)
    if "event" in df.columns:
        df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0.0)
    if "holiday" in df.columns:
        df["holiday"] = pd.to_numeric(df["holiday"], errors="coerce").fillna(0.0)

    #Adding useful features: day of week, month and day of month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    #If the dataset does not contain shelf life, assuming it as 30 days
    if "shelf_life" not in df.columns: 
        df["shelf_life"] = 30.0

    #I product category exists, convert it into numbers
    if "category" in df.columns:
        le_cat = LabelEncoder()
        df["category"] = df["category"].astype(str).fillna("unknown")
        df["category_enc"] = le_cat.fit_transform(df["category"])
    else:
        df["category_enc"] = 0 #If no category, set everything to 0

    #Cleaning event names
    df["event_name"] = df["event_name"].fillna("none").astype(str)
    df["event_name_lc"] = df["event_name"].str.lower().str.strip()
    
    #Encoding event names as numbers too
    le_event = LabelEncoder()
    df["event_name_enc"] = le_event.fit_transform(df["event_name_lc"])
    
    #Storing helpful dictionaries to map events back and forth
    EVENT_NAME_TO_CODE = {name: int(code) for code, name in enumerate(le_event.classes_)}
    CODE_TO_EVENT_NAME = {v: k for k, v in EVENT_NAME_TO_CODE.items()}

    #Features which the model will use to predict sales
    feature_cols = ["dayofweek", "month", "day", "shelf_life", "category_enc", "event_name_enc"]
    #Adding extra features if they exist in the dataset
    if "discount" in df.columns:
        feature_cols.append("discount")
    if "event" in df.columns:
        feature_cols.append("event")
    if "holiday" in df.columns:
        feature_cols.append("holiday")
    for c in feature_cols: #Making sure all selected feature columns are numeric
        df[c] = pd.to_numeric(df[c], errors="coerce")

    #Filtering data for the given product
    g = df[df["product_id"] == product_id].sort_values("date").reset_index(drop=True)
    if len(g) < 5:  #If product doesn’t have enough sales history, stop (need at least 5 records)
        raise ValueError(f"Not enough history (<5 rows) for product_id='{product_id}'.")

    #Geting training parameters for this product (or fall back to defaults)
    p = PARAMS_BY_PID.get(product_id, GLOBAL_DEFAULTS)
    test_size = min(max(p.get("test_size", 0.1), 0.05), 0.5)

    #Splittng into inputs (X = features) and output (y = sales numbers)
    X = g[feature_cols].copy()
    y = g["quantity_sold"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    #Building the prediction model (XGBoost regressor) with chosen parameters
    model = XGBRegressor(
        n_estimators=p["n_estimators"], #Number of trees 
        learning_rate=p["learning_rate"],
        max_depth=p["max_depth"], #How deep the trees can go
        subsample=p["subsample"], #Sample fraction of rows
        colsample_bytree=p["colsample_bytree"], #Sample fraction of features per tree
        objective=p["objective"], #Type of problem i.e regression
        n_jobs=p["n_jobs"], #Howmany CPU cores to use
        random_state=p["random_state"],
    )
    model.fit(X_train, y_train) #Trainind the model on past sales data

    #Making prediction on test data
    y_pred = model.predict(X_test)
    overall = compute_metrics(y_test, y_pred)

    #Finding test dates so results align with predictions
    test_len = len(y_test)
    test_dates = g["date"].iloc[-test_len:]  #Aligns correctly since shuffle=False
    #Saving a comparison plot of actual vs predicted values (only for one chosen product)
    plot_file = f"{product_id}_{_canonize(event_name)}_actual_vs_pred.png"
    if PRODUCT_ID is product_id:
        save_actual_vs_pred_plot(
            dates=test_dates, y_true=np.asarray(y_test), y_pred=np.asarray(y_pred),
            product_id=product_id, event_name=event_name, out_path=plot_file
        )
        print(f"[{product_id}] saved plot: {plot_file}")

    ev_lc = str(event_name).lower().strip() #Making event name lowercase and clean
    #Maping test rows to event names for event-specific metrics
    test_df = X_test.copy()
    test_df = test_df.assign(
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        #Converting numeric event codes back to readable names
        event_name_lc=test_df["event_name_enc"].round().astype(int).map(CODE_TO_EVENT_NAME)
    )
    #Filtering only rows that belong to this event
    sub = test_df[test_df["event_name_lc"] == ev_lc]
     #If we have data for this event, computing accuracy again just for that event
    if len(sub) >= 1:
        event_metrics = compute_metrics(sub["y_true"], sub["y_pred"])
    #Otherwise fill metrics with "Not Available"
    else:
        event_metrics = {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    
    #Printing overall test accuracy and event-specific accuracy
    print(f"[{product_id}] overall test metrics:", overall)
    print(f"[{product_id}] '{event_name}' test metrics:", event_metrics, f"(n={len(sub)} rows in test)")

    #Loading holiday/event calendar and estimating how long this event usually lasts
    cal = _load_calendar_flex(calendar_path)
    duration_days = _pick_event_duration_days(cal, ev_lc)
    print(f"[{product_id}] learned duration for '{event_name}': {duration_days} day(s)")

    #Converting event name into a code number (used for training/predictions)
    if ev_lc in EVENT_NAME_TO_CODE:
        ev_code = EVENT_NAME_TO_CODE[ev_lc]
        #Flagging as 1 if event is active, 0 if not
        event_flag = 1.0 if ev_lc not in ("none", "nan", "") else 0.0
    else:
        #If the event was not seen during training, treat as a "none" event
        ev_code = EVENT_NAME_TO_CODE.get("none", 0)
        event_flag = 1.0  # treat it as active event

    event_name_canonical = CODE_TO_EVENT_NAME.get(ev_code, ev_lc) #Getting the "standardized" name for this event
    print(f"[{product_id}] event requested='{event_name}' -> encoded={ev_code}, canonical='{event_name_canonical}'")

    #Building event windows 
    windows: List[Dict] = []
    if manual_start_date:
        #If a manual start date is given, creating one window with duration learned above
        start_dt = pd.to_datetime(manual_start_date, dayfirst=True, errors="coerce")
        if pd.isna(start_dt):
            raise ValueError(f"Could not parse manual_start_date='{manual_start_date}'. Use DD/MM/YYYY or YYYY-MM-DD.")
        end_dt = start_dt + pd.Timedelta(days=duration_days - 1)
        windows.append({"start_date": start_dt, "end_date": end_dt})
        print(f"[{product_id}] manual window: {start_dt.date()} -> {end_dt.date()} ({duration_days} day(s))")
    else:
        #Otherwise, using all event windows from the calendar file
        cal_ev = cal[cal["event_name"].str.lower().str.strip() == ev_lc]
        if cal_ev.empty:
            raise ValueError(f"Event '{event_name}' not found in calendar.")
        for _, r in cal_ev.iterrows():
            sdt = r["start_date"]
            edt = r["end_date"] if pd.notna(r["end_date"]) else r["start_date"] + pd.Timedelta(days=int(r["duration_days"]) - 1)
            windows.append({"start_date": sdt, "end_date": edt})
        print(f"[{product_id}] using {len(windows)} window(s) from calendar for '{event_name}'.")

    #Forecasting each window 
    #Base values are taken from the last sales record
    base_vals = {
        "shelf_life": float(g["shelf_life"].iloc[-1]),
        "category_enc": int(g["category_enc"].iloc[-1]),
        "discount": float(g["discount"].iloc[-1]) if "discount" in g.columns else 0.0,
        "holiday": float(g["holiday"].iloc[-1]) if "holiday" in g.columns else 0.0,
    }

    all_frames: List[pd.DataFrame] = []
    for idx, win in enumerate(windows, start=1):
        #Creating dates for each day in the event window
        dts = pd.date_range(win["start_date"], win["end_date"], freq="D")
        if len(dts) == 0:
            continue

        #Creating a dataframe of future dates with necessary features
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

        #Predicting sales for these future dates
        preds = model.predict(future_df[feature_cols])

        #Storing predictions with details
        out = pd.DataFrame({
            "product_id": product_id,
            "event_name": event_name,
            "window_index": idx,
            "window_start": win["start_date"].date(),
            "window_end": win["end_date"].date(),
            "date": dts,
            "forecast_quantity_sold": np.round(preds, 5),

            #Attaching accuracy columns 
            "MAE": overall["MAE"],
            "RMSE": overall["RMSE"],
            "R2": overall["R2"],
            "MAPE": overall["MAPE"],
        })
        all_frames.append(out)

    if not all_frames:
        raise ValueError(f"No valid dates generated for event '{event_name}'.")

    #Combining all event windows together
    result = pd.concat(all_frames, ignore_index=True)

    #Calculating total forecast for each window
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

    #Saving one row of metrics for easy reference
    metrics_row = {
        "product_id": product_id,
        "event_name": event_name,
        "MAE": overall["MAE"],
        "RMSE": overall["RMSE"],
        "R2": overall["R2"],
        "MAPE": overall["MAPE"],
    }

    #Savng forecasts and metrics to files if paths provided
    if out_csv:
        result.to_csv(out_csv, index=False)
        print(f"Saved forecasts: {out_csv}")

    if out_metrics_csv:
        pd.DataFrame([metrics_row]).to_csv(out_metrics_csv, index=False)
        print(f"Saved metrics: {out_metrics_csv}")

    #Returning both results and metrics 
    if return_metrics:
        return result, metrics_row
    return result

#All products summary
def summarize_metrics_all_products(
    event_name: str,
    calendar_path: str = "HolidaysEventsDataset.xlsx",
    sales_path: str = "Sales_with_HolidayEvents_true.csv",
    params_path: Optional[str] = "training_parameters_by_product.json",
    manual_start_date: Optional[str] = None,
    out_csv: Optional[str] = "ALL_PRODUCTS_metrics.csv",
) -> pd.DataFrame: 
    #Runs forecast_for_event for every product_id found in sales_path and collects metrics (MAE, RMSE, R2, MAPE)
    sales = pd.read_csv(sales_path)
    #Ensuring that the sales dataset has a "product_id" column
    if "product_id" not in sales.columns: 
        raise ValueError("sales_path must contain a 'product_id' column.")
    #Getting unique product IDs from the dataset
    products = (
        sales["product_id"].dropna().astype(str).drop_duplicates().sort_values().tolist()
    )

    rows = []
    #Looping through each product_id and run forecast
    for pid in products:
        try:
            #Running forecast for the product and collect metrics
            _, mrow = forecast_for_event(
                product_id=pid,
                event_name=event_name,
                calendar_path=calendar_path,
                sales_path=sales_path,
                params_path=params_path,
                out_csv=None,                 #Per-product forecasts not saved here
                out_metrics_csv=None,         #Collecting in one table
                manual_start_date=manual_start_date,
                return_metrics=True,          #Returning metrics for collection
            )
            rows.append(mrow)
        except Exception as e:
            #If forecast fails for a product, recording the error instead of stopping everything
            rows.append({
                "product_id": pid,
                "event_name_requested": event_name,
                "error": str(e),
            })

    summary = pd.DataFrame(rows) #Converting collected rows into a DataFrame

    #Putting the important metric columns side-by-side in a clean order
    desired_cols = [
        "product_id", "event_name",        
        "MAE", "RMSE", "R2", "MAPE",
    ]

    #Ensuring missing columns are added as empty
    for c in desired_cols:
        if c not in summary.columns:
            summary[c] = np.nan

    #Reordering columns (metrics first, then extra columns like errors)
    ordered = [c for c in desired_cols if c in summary.columns]
    if "error" in summary.columns:
        ordered = ordered + ["error"]
    extra = [c for c in summary.columns if c not in ordered]
    summary = summary[ordered + extra]

    #Saving summary table to CSV
    if out_csv:
        summary.to_csv(out_csv, index=False)
        print(f"[all-products] saved metrics summary -> {out_csv}")

    return summary

def save_product_forecasting_graph(df_out, save_image="product_forecast_plot"):
    #Saving a line graph of forecasted sales for a single product during an event.
    product_id = df_out['product_id'].iloc[0]
    event_name = df_out['event_name'].iloc[0]
    start_date = df_out['window_start'].min()
    end_date = df_out['window_end'].max()

    plt.figure(figsize=(10,5))
    plt.plot(df_out['date'], df_out['forecast_quantity_sold'], marker='o', linestyle='-')

    #Title with event, product_id and duration
    plt.title(f"Forecast Prediction for {product_id} - {event_name} \n (Dates from {start_date} to {end_date})")
    plt.xlabel("Forecast Dates")
    plt.ylabel("Forecast Quantity Sold")

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    #Saving the plot
    plt.savefig(f"{product_id}_{event_name}_{save_image}", dpi=300)  
    plt.close()  

if __name__ == "__main__":
    CAL_PATH = "HolidaysEventsDataset.xlsx"
    SALES_PATH = "Sales_with_HolidayEvents_true.csv"
    PARAMS_PATH = "training_parameters_by_product.json" 

    #Single product with manual window 
    PRODUCT_ID = "PN036"  #Change as needed
    EVENT_NAME = "May Day"  #Change as needed
    MANUAL_START = "05/05/2025"  #DD/MM/YYYY or YYYY-MM-DD; set to None to use calendar windows

    out_file = f"{PRODUCT_ID}_{_canonize(EVENT_NAME)}_manual_start_forecast.csv"
    out_metrics = f"{PRODUCT_ID}_{_canonize(EVENT_NAME)}_manual_start_metrics.csv"

    #Running forecast for a single product
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

    #Saving graph for single product forecast
    save_product_forecasting_graph(df_out)
    print(df_out.head(10))

    #Metrics summary for ALL products 
    all_metrics_file = f"ALL_PRODUCTS_{_canonize(EVENT_NAME)}_metrics.csv"
    summary_df = summarize_metrics_all_products(
        event_name=EVENT_NAME,
        calendar_path=CAL_PATH,
        sales_path=SALES_PATH,
        params_path=PARAMS_PATH,
        manual_start_date=MANUAL_START,     # or None to use calendar windows
        out_csv=all_metrics_file,
    )
    print(summary_df.head(10))
