"""tests/test_fetchers.py - unit tests for fetchers.py."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

from fetchers import (
    compute_momentum_6m_1m,
    extract_as_of_date,
    fetch_fundamentals,
    fetch_price_panels,
    score_headlines_finbert,
)


def test_extract_as_of_date_and_momentum():
    idx = pd.bdate_range("2025-01-02", periods=145)
    s = pd.Series(np.linspace(100.0, 145.0, len(idx)), index=idx)
    assert extract_as_of_date(s) == idx[-1].strftime("%Y-%m-%d")

    mom = compute_momentum_6m_1m(s)
    assert math.isfinite(mom)
    assert mom > 0

    assert math.isnan(compute_momentum_6m_1m(pd.Series([100.0, 105.0])))


def test_fetch_price_panels_parses_close(monkeypatch):
    dates = pd.bdate_range("2025-03-03", periods=10)
    cols = pd.MultiIndex.from_product([["Close"], ["INFY.NS", "^NSEI"]])
    mock_df = pd.DataFrame(
        {
            ("Close", "INFY.NS"): np.linspace(1500.0, 1550.0, len(dates)),
            ("Close", "^NSEI"): np.linspace(23000.0, 23500.0, len(dates)),
        },
        index=dates,
    )
    mock_df.columns = cols

    import yfinance as yf

    monkeypatch.setattr(yf, "download", lambda *a, **kw: mock_df)
    close_df, nifty = fetch_price_panels(["INFY"], benchmark="^NSEI", period="1mo", retries=1, pause=0.0)
    assert "INFY" in close_df.columns
    assert len(close_df) == len(dates)
    assert len(nifty) == len(dates)


def test_fetch_fundamentals_skips_missing_price(monkeypatch):
    class DummyTicker:
        def __init__(self, sym):
            self.sym = sym

        @property
        def info(self):
            if self.sym == "BAD.NS":
                return {"shortName": "Bad Co"}
            return {
                "shortName": "Good Co",
                "sector": "Technology",
                "currentPrice": 500.0,
                "trailingPE": 20.0,
                "priceToBook": 4.0,
                "returnOnEquity": 0.18,
            }

    import yfinance as yf

    monkeypatch.setattr(yf, "Ticker", DummyTicker)
    df, skipped = fetch_fundamentals(["GOOD.NS", "BAD.NS"], cagr_fn=None, retries=1, pause=0.0)
    assert list(df["Ticker"]) == ["GOOD"]
    assert len(skipped) == 1
    assert skipped[0]["ticker"] == "BAD.NS"


def test_score_headlines_finbert_with_stub_classifier():
    def stub_clf(texts):
        out = []
        for t in texts:
            low = t.lower()
            if "surge" in low or "profit" in low:
                out.append({"label": "positive", "score": 0.9})
            elif "drop" in low or "loss" in low:
                out.append({"label": "negative", "score": 0.8})
            else:
                out.append({"label": "neutral", "score": 0.7})
        return out

    res = score_headlines_finbert(
        {
            "INFY": ["Profit rises in Q2", "Shares surge on deal", "Company holds AGM"],
            "TCS": ["Only one headline"],
        },
        classifier=stub_clf,
    )
    cnt, avg, items = res["INFY"]
    assert cnt == 3
    assert avg is not None and avg > 0
    assert len(items) == 3
    assert res["TCS"][0] == 1
