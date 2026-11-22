"""growth_data.py - 3-year revenue and EPS/net-income CAGR helper."""
from __future__ import annotations

import math
import time
from typing import Dict

import numpy as np
import pandas as pd

KEYS = ("Rev_CAGR_3Y_%", "EPS_CAGR_3Y_%")
REV_LABELS = ("Total Revenue", "Operating Revenue", "Total Revenues")
EPS_LABELS = ("Diluted EPS", "Basic EPS", "Net Income", "Net Income Common Stockholders")


def _nan_result() -> Dict[str, float]:
    return {k: float("nan") for k in KEYS}


def _find_row(df: pd.DataFrame, candidates) -> pd.Series | None:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    idx_map = {str(i).strip().lower(): i for i in df.index}
    for label in candidates:
        key = label.strip().lower()
        if key in idx_map:
            return df.loc[idx_map[key]]
    return None


def _cagr_3y_from_series(series: pd.Series) -> float:
    if series is None or series.empty:
        return float("nan")
    s = series.copy()
    try:
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
    except Exception:
        pass
    vals = [float(x) for x in pd.to_numeric(s, errors="coerce").dropna().tolist()]
    if len(vals) < 4:
        return float("nan")
    start_val, end_val = vals[-4], vals[-1]
    if not (math.isfinite(start_val) and math.isfinite(end_val)):
        return float("nan")
    if start_val <= 0 or end_val <= 0:
        return float("nan")
    cagr = ((end_val / start_val) ** (1.0 / 3.0) - 1.0) * 100.0
    return float(cagr) if math.isfinite(cagr) else float("nan")


def fetch_cagr_3y(symbol: str, retries: int = 2, pause: float = 0.25) -> Dict[str, float]:
    """Return 3Y CAGR percentages for revenue and EPS/net-income; never raises."""
    out = _nan_result()
    try:
        import yfinance as yf
    except Exception:
        return out

    attempts = max(1, int(retries))
    financials = None
    for attempt in range(1, attempts + 1):
        try:
            ticker_obj = yf.Ticker(symbol)
            financials = getattr(ticker_obj, "financials", None)
            break
        except Exception:
            if attempt < attempts and pause > 0:
                time.sleep(pause)
            else:
                return out

    if financials is None or not isinstance(financials, pd.DataFrame) or financials.empty:
        return out

    rev_row = _find_row(financials, REV_LABELS)
    eps_row = _find_row(financials, EPS_LABELS)

    if rev_row is not None:
        out["Rev_CAGR_3Y_%"] = _cagr_3y_from_series(rev_row)
    if eps_row is not None:
        out["EPS_CAGR_3Y_%"] = _cagr_3y_from_series(eps_row)

    return out
