"""tests/test_growth_data.py - unit tests for growth_data.py."""
from __future__ import annotations

import math
import pandas as pd

from growth_data import KEYS, _cagr_3y_from_series, fetch_cagr_3y


def test_cagr_3y_from_series_exact_and_edge_cases():
    dates = pd.to_datetime(["2022-03-31", "2023-03-31", "2024-03-31", "2025-03-31"])
    s = pd.Series([100.0, 110.0, 121.0, 133.1], index=dates)
    cagr = _cagr_3y_from_series(s)
    assert math.isfinite(cagr)
    assert abs(cagr - 10.0) < 1e-6

    # Too few periods (<4) returns NaN
    short_s = pd.Series([100.0, 120.0, 140.0], index=dates[:3])
    assert math.isnan(_cagr_3y_from_series(short_s))

    # Non-positive start/end returns NaN
    neg_s = pd.Series([-50.0, 10.0, 20.0, 30.0], index=dates)
    assert math.isnan(_cagr_3y_from_series(neg_s))


def test_fetch_cagr_3y_handles_missing_and_errors(monkeypatch):
    class DummyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        @property
        def financials(self):
            raise RuntimeError("Simulated upstream failure")

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    res = fetch_cagr_3y("INFY.NS", retries=1, pause=0.0)
    assert set(KEYS).issubset(res.keys())
    assert all(math.isnan(res[k]) for k in KEYS)
