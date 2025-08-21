#!/usr/bin/env python3
# merge_sales_holidays_strict.py
# Exact columns only; expiry_date from sales; separate row per event; inner join; DD-MM-YYYY

import re
import pandas as pd

SALES_PATH  = "Sales_Dataset.csv"
EVENTS_PATH = "HolidaysEventsEntries.xlsx"
OUT_CSV  = "Sales_with_HolidayEvents_true.csv"
OUT_XLSX = "Sales_with_HolidayEvents.xlsx"
DATE_FMT = "%d-%m-%Y"

FINAL_ORDER = [
    "date","product_id","product","category","quantity_sold","expiry_date",
    "unit_price","discount","discount_price","shelf_life","holiday","event","event_name"
]

norm = lambda s: re.sub(r"_+","_", re.sub(r"[^0-9a-z]+","_", str(s).strip().lower())).strip("_")
def ncols(df): df.columns = [norm(c) for c in df.columns]; return df

def detect(cols, cands):
    for c in cands:
        if c in cols: return c
    for c in cols:
        if any(k in c for k in cands): return c
    return None

def fmt_from_series(s, date_fmt):
    #Formatting with fallback to original text if unparseable
    orig = s.astype(str)
    dt = pd.to_datetime(s, errors="coerce", dayfirst=True)
    out = dt.dt.strftime(date_fmt)
    return out.where(dt.notna(), orig.str.strip())

#Loading & normalizing headers
sales  = pd.read_excel(SALES_PATH) if SALES_PATH.lower().endswith((".xlsx",".xls")) else pd.read_csv(SALES_PATH)
events = pd.read_excel(EVENTS_PATH) if EVENTS_PATH.lower().endswith((".xlsx",".xls")) else pd.read_csv(EVENTS_PATH)
sales, events = ncols(sales), ncols(events)

#Detecting key columns 
sales_date = detect(sales.columns.tolist(), ["date","sales_date","order_date","invoice_date","txn_date","transaction_date","day","ds"])
if not sales_date:
    for c in sales.columns:
        if pd.to_datetime(sales[c], errors="coerce", dayfirst=True).notna().mean() >= 0.8:
            sales_date = c; break
if not sales_date:
    raise ValueError("Sales date column not found.")

ev_name  = detect(events.columns.tolist(), ["event","events","event_name","holiday","holidays","holiday_name","holidays_event","name","title","festival","description"])
ev_start = detect(events.columns.tolist(), ["start_date","start","from","begin","date_start"])
ev_end   = detect(events.columns.tolist(), ["end_date","end","to","finish","date_end"])
if (not ev_start or not ev_end) and ("date" in events.columns): ev_start = ev_end = "date"
if not ev_start or not ev_end: raise ValueError("Event start/end columns not found.")
if not ev_name:
    ev_name = "event_name"; events[ev_name] = "Unnamed Event"

#Preparing sales (keep original expiry_date from sales only) 
sales[sales_date] = pd.to_datetime(sales[sales_date], errors="coerce", dayfirst=True)
sales = sales.dropna(subset=[sales_date]).copy()

#Keeping a formatted copy of the two date outputs from sales
sales["_date_out"] = sales[sales_date].dt.strftime(DATE_FMT)

exp_col = "expiry_date" if "expiry_date" in sales.columns else detect(sales.columns.tolist(), ["expiry_date","expiry","exp_date"])
if exp_col is None:
    sales["_expiry_out"] = ""  
else:
    sales["_expiry_out"] = fmt_from_series(sales[exp_col], DATE_FMT)

#Preparing events calendar (used only for matching; start/end NOT kept) 
events[ev_start] = pd.to_datetime(events[ev_start], errors="coerce", dayfirst=True)
events[ev_end]   = pd.to_datetime(events[ev_end],   errors="coerce", dayfirst=True)
events = events.dropna(subset=[ev_start, ev_end])
events = events[events[ev_start] <= events[ev_end]].copy()

rows = []
for _, r in events.iterrows():
    for d in pd.date_range(r[ev_start].normalize(), r[ev_end].normalize(), freq="D"):
        rows.append({"__match_date__": pd.Timestamp(d).normalize(), "event_name": r[ev_name]})
cal = pd.DataFrame(rows, columns=["__match_date__","event_name"])

#Merging (inner join: drop unmatched)
sales["__match_date__"] = sales[sales_date].dt.normalize()
merged = sales.merge(cal, on="__match_date__", how="inner")

#Building EXACT final columns only (no extras) 
out = pd.DataFrame()
# Date & expiry_date from sales cached outputs
out["date"] = merged["_date_out"]
out["product_id"] = merged.get("product_id", "")
out["product"] = merged.get("product", "")
out["category"] = merged.get("category", "")
out["quantity_sold"]= merged.get("quantity_sold", "")
out["expiry_date"] = merged["_expiry_out"]  #from sales data
#The rest from sales if present (else blank)
out["unit_price"] = merged.get("unit_price", "")
out["discount"] = merged.get("discount", "")
out["discount_price"] = merged.get("discount_price", "")
out["shelf_life"] = merged.get("shelf_life", "")
out["holiday"] = merged.get("holiday", "")
out["event"] = merged.get("event", "")
#Event_name from calendar
out["event_name"] = merged["event_name"]

#Enforcing exact column order and ONLY these columns
out = out[FINAL_ORDER]

#Saving
out.to_csv(OUT_CSV, index=False)
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as w:
    out.to_excel(w, index=False, sheet_name="merged")

print(f"Saved exact-columns output → {OUT_CSV}, {OUT_XLSX}")
