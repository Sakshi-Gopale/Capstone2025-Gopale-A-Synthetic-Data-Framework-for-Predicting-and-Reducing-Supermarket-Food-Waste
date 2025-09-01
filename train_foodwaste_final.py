import argparse, json, re, warnings
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import (
    train_test_split, GroupShuffleSplit,
    KFold, GroupKFold, TimeSeriesSplit, StratifiedKFold
)
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)
from sklearn.ensemble import RandomForestRegressor
import joblib

#Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

#Optional libraries: XGBoost & Deep Learning
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

DL_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except Exception:
    DL_AVAILABLE = False


def infer_types(df: pd.DataFrame, target: Optional[str], date_col: Optional[str]) -> Tuple[List[str], List[str]]:
    #Identifying numeric and categorical columns, converting date_col to datetime 
    cols = df.columns.tolist() #Getting all column names from the dataframe
    drop = set([c for c in [target, date_col] if c]) #Collecting target and date_col into a set of columns to exclude from features
    #If a date_col is specified and exists but is not already datetime, attempting conversion
    if date_col and date_col in df.columns and not np.issubdtype(df[date_col].dtype, np.datetime64):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce") #Converting date_col to datetime
    numeric_cols = [c for c in cols if c not in drop and pd.api.types.is_numeric_dtype(df[c])]  #Selecting numeric columns, excluding target and date_col
    categorical_cols = [c for c in cols if c not in drop and c not in numeric_cols]  #Select remaining non-numeric columns as categorical, excluding target and date_col

    return numeric_cols, categorical_cols


def add_time_parts(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    #Expanding a datetime column into features like year, month, day, dow, week, quarter, etc
    df = df.copy()
    if date_col: #Proceeding only if a date column is specified
        if not np.issubdtype(df[date_col].dtype, np.datetime64):
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["year"] = df[date_col].dt.year
        df["month"] = df[date_col].dt.month
        df["day"] = df[date_col].dt.day
        df["dow"] = df[date_col].dt.dayofweek #(0=Monday, 6=Sunday)
        df["weekofyear"] = df[date_col].dt.isocalendar().week.astype("Int64")
        df["is_month_start"] = df[date_col].dt.is_month_start.astype(int)
        df["is_month_end"] = df[date_col].dt.is_month_end.astype(int)
        df["quarter"] = df[date_col].dt.quarter
    return df


def add_lag_rolling(df: pd.DataFrame, date_col: Optional[str], id_cols: List[str], target: str,
                    lags=(1, 7, 28), rolls=(7, 28)) -> pd.DataFrame:
    #Creating lag and rolling-mean features grouped by IDs to capture temporal patterns
    if not date_col or not id_cols: #If no date column or grouping IDs are provided, returning the dataframe unchanged
        return df
    
    df = df.copy()
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    
    df = df.sort_values(id_cols + [date_col]) #Sorting data by IDs and time to maintain chronological order for lag/rolling calculations
    grp = df.groupby(id_cols, sort=False)
    for L in lags:  #For each lag value (1, 7, 28 days), creating lag features
        df[f"lag_{target}_{L}"] = grp[target].shift(L) #Shifting target values by L steps within each group to create lag features
    
    for W in rolls: #For each window size (7, 28 days), creating rolling mean features
        #Computing rolling mean with minimum valid periods = half window size
        df[f"rmean_{target}_{W}"] = grp[target].rolling(W, min_periods=max(1, W//2))[target].mean().reset_index(level=id_cols, drop=True)
    return df

def time_based_split(df: pd.DataFrame, date_col: str, test_size: float = 0.2):
    #Splitting data chronologically into train/test sets using a cutoff quantile of date_col
    df = df.sort_values(date_col)
    cutoff = df[date_col].quantile(1 - test_size) #Determining the date value that separates train/test based on test_size fraction
    train_df = df[df[date_col] <= cutoff] 
    test_df = df[df[date_col] > cutoff] 
    return train_df, test_df

def evaluate_regression(y_true, y_pred):
    #Returning regression metrics: MAE, RMSE, R²
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str], scale_numeric: bool = True):
    #Building preprocessing pipeline: imputing/scaling numeric and imputing/one-hot encoding categorical features
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), #Filling missing numeric values using median
        ("scaler", StandardScaler() if scale_numeric else "passthrough"), #Standardizing numeric features to zero mean and unit variance
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop", verbose_feature_names_out=False) #Combining numeric and categorical pipelines, dropping any remaining columns
    return pre

def bin_target_for_stratify(y: pd.Series, q: int = 5) -> np.ndarray:
    #Bin continuous target values into quantiles for stratified cross-validation
    y = pd.Series(y).astype(float)
    try:
        bins = pd.qcut(y, q=q, labels=False, duplicates="drop")
        if len(pd.unique(bins.dropna())) < 2: #If too few bins created, fallback to fewer quantiles
            bins = pd.qcut(y, q=max(2, q // 2), labels=False, duplicates="drop")
    except Exception:
        ranks = y.rank(method="average", pct=True) #If qcut fails, ranking targets and convert to percentile bins
        bins = pd.cut(ranks, bins=max(2, q), labels=False, include_lowest=True)
    bins = bins.fillna(0).astype(int).values
    return bins


def get_cv_splits(X: pd.DataFrame, y: pd.Series, cv_type: str, folds: int,
                  groups: Optional[pd.Series], date_col: Optional[str]):
    #Generating train/validation indices for CV strategies: KFold, StratifiedKFold, GroupKFold, TimeSeries
    if cv_type == "timeseries": 
        if date_col is None or date_col not in X.columns:
            raise SystemExit("TimeSeries CV requires --date_col present in features.")
        order = np.argsort(pd.to_datetime(X[date_col], errors="coerce").values)  #Sorting indices by datetime
        X_sorted = X.iloc[order].reset_index(drop=True)
        splitter = TimeSeriesSplit(n_splits=folds) #Initializing time-series CV splitter
        for tr, va in splitter.split(X_sorted):
            yield (order[tr], order[va])
    
    elif cv_type == "groupkfold": #Ensures samples from the same group are not split across folds
        if groups is None:
            raise SystemExit("GroupKFold requires --cv_group_col (or first id col).")
        splitter = GroupKFold(n_splits=folds) 
        for tr, va in splitter.split(X, y, groups=groups):
            yield tr, va #Yielding train/validation indices for each fold without group leakage

    elif cv_type == "stratkfold": #Splits so that each fold has similar target distribution
        y_bins = bin_target_for_stratify(y, q=5)
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42) #Initializing stratified KFold with shuffling for randomness
        for tr, va in splitter.split(X, y_bins):
            yield tr, va
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42) #Simple random splits without stratification or groups
        for tr, va in splitter.split(X, y):
            yield tr, va

def train_one_model(model_name: str, Xtr, ytr, Xva, yva, input_dim=None):
    #Training a single model (RF, XGB, or DL) and return trained model + validation predictions
    if model_name == "rf":
        m = RandomForestRegressor(
            n_estimators=600, max_depth=None, random_state=42, n_jobs=-1, min_samples_leaf=2
        )
        m.fit(Xtr, ytr)
        preds = m.predict(Xva)
        return m, preds
    
    elif model_name == "xgb":
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed.")
        m = XGBRegressor(
            n_estimators=900, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            tree_method="hist" #Faster histogram-based tree building
        )
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        preds = m.predict(Xva)
        return m, preds
    
    elif model_name == "dl": #Feedforward Neural network for regression
        if not DL_AVAILABLE:
            raise RuntimeError("TensorFlow/Keras not installed.")
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)), #Inputing layer with dynamic input dimension
            layers.Dense(256, activation="relu"), #First hidden layer
            layers.BatchNormalization(),
            layers.Dropout(0.2),  #For regularization
            layers.Dense(128, activation="relu"), #Second hidden layer
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"), #Third hidden layer
            layers.Dense(1, activation="linear"), #Output layer
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3),  #Adam optimizer with learning rate 0.001
                      loss="mse", metrics=["mae"])
        cb = [keras.callbacks.EarlyStopping(monitor="val_mae", patience=15, restore_best_weights=True)] #Stopping training early if validation MAE does not improve
        model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=200, batch_size=256, verbose=0, callbacks=cb)
        preds = model.predict(Xva, verbose=0).ravel()
        return model, preds
    else:
        raise ValueError(model_name)


def cross_validate(X: pd.DataFrame, y: pd.Series, cv_type: str, folds: int, models: List[str],
                   numeric_cols: List[str], categorical_cols: List[str],
                   scale_numeric: bool, groups: Optional[pd.Series], date_col: Optional[str]) -> Dict[str, dict]:
    #Running CV for multiple models, collect per-fold metrics, and compute averaged performance
    results: Dict[str, dict] = {m: {"folds": [], "mean": {}} for m in models}
    
    for fold_idx, (tr_idx, va_idx) in enumerate(get_cv_splits(X, y, cv_type, folds, groups, date_col), start=1): #Looping through each fold
        Xtr_raw, Xva_raw = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy() #Extracting raw train/validation data
        ytr, yva = y.iloc[tr_idx].astype(float), y.iloc[va_idx].astype(float)  #Ensuring target is float
        pre = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)
        pre.fit(Xtr_raw) #Fitting preprocessing pipeline on training fold
        Xtr = pre.transform(Xtr_raw)
        Xva = pre.transform(Xva_raw)
        
        for mname in models: #Training and evaluating each model
            try:
                mdl, preds = train_one_model(mname, Xtr, ytr, Xva, yva, input_dim=Xtr.shape[1])
                met = evaluate_regression(yva, preds)
                results[mname]["folds"].append({"fold": fold_idx, **met})
            except Exception as e:
                results[mname]["folds"].append({"fold": fold_idx, "ERROR": str(e)})
    
    for mname in models: #Computing mean metrics across all successful folds for each model
        vals = [f for f in results[mname]["folds"] if "ERROR" not in f]
        if vals:
            results[mname]["mean"] = {
                "MAE": float(np.mean([v["MAE"] for v in vals])),
                "RMSE": float(np.mean([v["RMSE"] for v in vals])),
                "R2": float(np.mean([v["R2"] for v in vals])),
                "folds": len(vals)
            }
        else:
            results[mname]["mean"] = {"ERROR": "All folds failed."}
    return results

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, scheme: str = "zero_vs_nonzero", bins: int = 3):
    #Converting regression outputs to classes (zero_vs_nonzero or quantiles) and compute classification metrics
    if scheme == "zero_vs_nonzero":
        y_true_cls = (y_true > 0).astype(int)
        y_pred_cls = (y_pred > 0).astype(int)
        labels = [0, 1] #Class labels
        
    elif scheme == "quantile": #Quantile-based multi-class classification: splitting values into specified number of bins
        y_true_cls = pd.qcut(pd.Series(y_true), q=bins, labels=False, duplicates="drop")
        y_pred_cls = pd.qcut(pd.Series(y_pred), q=bins, labels=False, duplicates="drop")
        labels = sorted(pd.unique(y_true_cls.dropna()).tolist()) #Determining valid class labels
        y_true_cls = y_true_cls.fillna(0).astype(int).values
        y_pred_cls = y_pred_cls.fillna(0).astype(int).values
    else:
        raise ValueError("scheme must be 'zero_vs_nonzero' or 'quantile'")

    #Computing confusion matrix using the class labels
    cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y_true_cls, y_pred_cls, labels=labels, average=None, zero_division=0)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true_cls, y_pred_cls, average="macro", zero_division=0)
    accuracy = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
    #Returning a detailed dictionary of metrics including per-class and macro averages
    return {
        "scheme": scheme,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "accuracy": accuracy,
        "per_class": [
            {"label": int(lbl), "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for lbl, p, r, f, s in zip(labels, precision, recall, f1, support)
        ],
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)}
    }


def save_residual_plots(y_true, y_pred, out_png_scatter: Path, title: str):
    #Saving scatter plot of actual vs predicted values for residual analysis
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.title(f"Actual vs Predicted — {title}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(out_png_scatter)
    plt.close()


def plot_metrics_summary(reg: dict, cls: Optional[dict], out_png: Path, title: str):
    #Saving a summary plot with Accuracy, Precision, Recall, and F1 scores as text
    lines = [f"Accuracy: {cls.get('accuracy', 0.0):.4f}",
            f"Macro Precision: {cls.get('macro', {}).get('precision', 0.0):.4f}",
            f"Macro Recall: {cls.get('macro', {}).get('recall', 0.0):.4f}",
            f"Macro F1: {cls.get('macro', {}).get('f1', 0.0):.4f}"]
    txt = "\n".join(lines)

    plt.figure(figsize=(5, 3))
    plt.axis("off")
    plt.title(f"Metrics Summary — {title}")
    plt.text(0.02, 0.85, txt, va="top")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()  #Parsing command-line arguments
    p.add_argument("--data", required=False, default="wastage_data.csv")
    p.add_argument("--target", required=False, default="waste_product_quantity")
    p.add_argument("--date_col", required=False, default="date")
    p.add_argument("--id_cols", nargs="*", default=[])
    p.add_argument("--outdir", default="./models")
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--time_split", action="store_true", default=True)
    p.add_argument("--models", nargs="+", default=["xgb"], choices=["rf", "xgb", "dl"])
    p.add_argument("--no_scale_numeric", action="store_true")
    p.add_argument("--save_template", action="store_true")
    p.add_argument("--cv_type", choices=["kfold", "groupkfold", "timeseries", "stratkfold"], default="timeseries")
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--cv_group_col", default=None)
    p.add_argument("--cls_scheme", choices=["zero_vs_nonzero", "quantile", "none"], default="zero_vs_nonzero")
    p.add_argument("--cls_bins", type=int, default=5)
    args = p.parse_args()

    #Settign output directories
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    analysis_dir = outdir / "analysis"; analysis_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)

    #Determining target column
    target = args.target
    if not target:
        candidates = [c for c in df.columns if re.search(r"waste", c, re.IGNORECASE)]
        if candidates:
            target = candidates[0]
            print(f"[INFO] Inferred target column: {target}")
        else:
            raise SystemExit("Please provide --target; could not infer from columns.")

    #Verifying date column
    date_col = args.date_col
    if date_col and date_col not in df.columns:
        raise SystemExit(f"--date_col '{date_col}' not found in columns {df.columns.tolist()}")

    id_cols = [c for c in args.id_cols if c in df.columns] #Filtering valid ID columns

    #Feature engineering: time & lag features
    if date_col:
        df = add_time_parts(df, date_col)
        if id_cols:
            df = add_lag_rolling(df, date_col=date_col, id_cols=id_cols, target=target)

    #Cleaning target column
    df = df[pd.to_numeric(df[target], errors="coerce").notnull()].copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")

    #Splitting features and target
    y = df[target].astype(float)
    X = df.drop(columns=[target])

    numeric_cols, categorical_cols = infer_types(X, target=None, date_col=date_col)

    #Setting up group-based CV 
    groups = None
    if args.cv_type == "groupkfold":
        gcol = args.cv_group_col or (id_cols[0] if id_cols else None)
        if gcol is None or gcol not in X.columns:
            raise SystemExit("For groupkfold, provide --cv_group_col or ensure the first --id_cols exists in data.")
        groups = X[gcol]

    #Cross-validation for selected models
    cv_results = cross_validate(
        X, y, cv_type=args.cv_type, folds=args.folds, models=args.models,
        numeric_cols=numeric_cols, categorical_cols=categorical_cols,
        scale_numeric=not args.no_scale_numeric, groups=groups, date_col=date_col
    )
    (outdir / "cv_metrics.json").write_text(json.dumps(cv_results, indent=2))

    #Time-based train/test split or shuffle split
    if date_col and args.time_split:
        df2 = pd.concat([X, y], axis=1)
        train_df, test_df = time_based_split(df2, date_col=date_col, test_size=args.test_size)
        y_train = train_df[target].values.astype(float); X_train = train_df.drop(columns=[target])
        y_val = test_df[target].values.astype(float); X_val = test_df.drop(columns=[target])
    else:
        if id_cols:
            gss = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=42)
            grp = X[id_cols[0]]
            tr_idx, te_idx = next(gss.split(X, y, groups=grp))
            X_train, X_val = X.iloc[tr_idx], X.iloc[te_idx]
            y_train, y_val = y.iloc[tr_idx], y.iloc[te_idx]
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.test_size, random_state=42)

    #Preprocessing: fit and transform features
    numeric_cols, categorical_cols = infer_types(X_train, target=None, date_col=date_col)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=not args.no_scale_numeric)
    preprocessor.fit(X_train)
    Xt_train = preprocessor.transform(X_train)
    Xt_val = preprocessor.transform(X_val)

    #Saving preprocessing schema and preprocessor
    schema = {
        "target": target,
        "date_col": date_col,
        "id_cols": id_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        #Time-based features generated from date_col
        "generated_time_parts": ["year", "month", "day", "dow", "weekofyear", "is_month_start", "is_month_end", "quarter"] if date_col else [],
        #Lag and rolling features automatically created
        "generated_lag_parts": [c for c in X.columns if isinstance(c, str) and (c.startswith("lag_") or c.startswith("rmean_"))],
    }
    joblib.dump(preprocessor, outdir / "preprocessor.pkl")
    (outdir / "schema.json").write_text(json.dumps(schema, indent=2))

    #Training final models and evaluate on validation set
    final_metrics = {} #Storing holdout/validation metrics for each model
    
    for mname in args.models:
        try:
            #Training model based on type
            if mname == "dl":
                mdl, _ = train_one_model(mname, Xt_train, y_train, Xt_val, y_val, input_dim=Xt_train.shape[1])
                mdl.save(outdir / "dl.keras")
                y_pred = mdl.predict(Xt_val, verbose=0).ravel()
            elif mname == "rf":
                mdl, _ = train_one_model(mname, Xt_train, y_train, Xt_val, y_val)
                joblib.dump(mdl, outdir / "rf.joblib")
                y_pred = mdl.predict(Xt_val)
            elif mname == "xgb":
                mdl, _ = train_one_model(mname, Xt_train, y_train, Xt_val, y_val)
                joblib.dump(mdl, outdir / "xgb.joblib")
                y_pred = mdl.predict(Xt_val)

            #Evaluating regression metrics
            reg_metrics = evaluate_regression(y_val, y_pred)
            final_metrics[mname] = reg_metrics
            
            save_residual_plots(y_val, y_pred, analysis_dir / f"actual_vs_pred_{mname}.png", title=mname.upper())

            #Computing classification style metrics
            cls_result = None
            if args.cls_scheme != "none":
                bins = max(2, int(args.cls_bins))
                scheme = "quantile" if args.cls_scheme == "quantile" else "zero_vs_nonzero"
                cls_result = classification_metrics(y_val, y_pred, scheme=scheme, bins=bins)
                             
            #Metrics summary image
            plot_metrics_summary(reg_metrics, cls_result, analysis_dir / f"metrics_summary_{mname}.png", title=mname.upper())

        except Exception as e:
            final_metrics[mname] = {"ERROR": str(e)}

    #Saving final metrics to JSON
    (outdir / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

    #Saving manual input template for inference
    if args.save_template:
        template_cols = X.columns.tolist()
        pd.DataFrame(columns=template_cols).to_csv(outdir / "manual_input_template.csv", index=False)

    json.dumps({"cv_results": cv_results, "holdout_metrics": final_metrics,
                      "analysis_dir": str(analysis_dir)}, indent=2)


if __name__ == "__main__":
    main()
