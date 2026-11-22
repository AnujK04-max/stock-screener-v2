"""
history.py - the daily snapshot log that the performance tracker reads.

- Stores RAW inputs plus outputs plus the model version, so any day can be re-scored later.
- Keys rows by (As_Of_Date, Model_Version, Ticker). As_Of_Date is the date of the last COMPLETE
  trading session behind the prices, not the date the script ran, so weekend and holiday runs
  update Friday's row instead of creating a fake extra cohort.
- Migrates the original ``screener_history.csv`` (Run_Date / Timestamp columns, no model version):
  old rows are kept and labelled ``legacy-v1``; the tracker never mixes them with the new model.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

LEGACY_VERSION = "legacy-v1"

SNAPSHOT_RAW = [
    "PE", "PB", "EPS", "Book_Value", "EV_EBITDA", "PS", "ROE_%", "ROA_%", "DE",
    "Rev_Growth_%", "EPS_Growth_%", "Rev_CAGR_3Y_%", "EPS_CAGR_3Y_%",
    "NIM_%", "GNPA_%", "CAR_%", "Mom_6m_1m_%", "Headline_Count", "FinBERT_Score",
]
SNAPSHOT_OUT = [
    "Val_Tag", "Val_Metric_Type", "Val_Metric", "Val_Rank", "Qual_Rank", "Growth_Rank",
    "Sent_Score", "Mom_Sent_Rank", "Base_Effect_Flag", "Rankable", "Composite_Score",
    "Screen_Flag",
]
HISTORY_COLUMNS = (
    ["As_Of_Date", "Run_Timestamp", "Model_Version", "Ticker", "Sector", "Industry",
     "Price", "Change_%"] + SNAPSHOT_RAW + SNAPSHOT_OUT
)
KEY = ["As_Of_Date", "Model_Version", "Ticker"]


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    """Bring any historical frame (including the original schema) to HISTORY_COLUMNS."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    out = df.copy()
    if "As_Of_Date" not in out.columns:
        out["As_Of_Date"] = out["Run_Date"] if "Run_Date" in out.columns else np.nan
    elif "Run_Date" in out.columns:
        out["As_Of_Date"] = out["As_Of_Date"].fillna(out["Run_Date"])
    if "Run_Timestamp" not in out.columns:
        out["Run_Timestamp"] = out["Timestamp"] if "Timestamp" in out.columns else np.nan
    if "Model_Version" not in out.columns:
        out["Model_Version"] = LEGACY_VERSION
    out["Model_Version"] = out["Model_Version"].fillna(LEGACY_VERSION)
    out = out.reindex(columns=HISTORY_COLUMNS)
    out = out.dropna(subset=["As_Of_Date", "Ticker"])
    out["As_Of_Date"] = pd.to_datetime(out["As_Of_Date"]).dt.strftime("%Y-%m-%d")
    return out


def snapshot_rows(scored: pd.DataFrame, as_of, run_ts, model_version: str) -> pd.DataFrame:
    """Turn a scored universe into history rows for one run."""
    out = scored.copy()
    out["As_Of_Date"] = pd.Timestamp(as_of).strftime("%Y-%m-%d")
    out["Run_Timestamp"] = run_ts
    out["Model_Version"] = model_version
    return normalize_history(out)


def merge_history(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Append ``new`` to ``existing``; a later run for the same key replaces the earlier one."""
    parts = [normalize_history(existing), normalize_history(new)]
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.drop_duplicates(subset=KEY, keep="last")
    return combined.sort_values(KEY, kind="stable").reset_index(drop=True)[HISTORY_COLUMNS]


def read_history(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return normalize_history(pd.read_csv(path))


def write_history(df: pd.DataFrame, path: str) -> None:
    out = df.copy()
    float_cols = out.select_dtypes(include="float").columns
    out[float_cols] = out[float_cols].round(4)      # keeps the file small and diffs stable
    out.to_csv(path, index=False)
