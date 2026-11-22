"""tests/synthetic.py - deterministic synthetic fixtures for testing the screener."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_raw_universe() -> pd.DataFrame:
    """Create a representative multi-sector NSE universe covering edge cases."""
    return pd.DataFrame(
        [
            {
                "Ticker": "HDFCBANK",
                "Name": "HDFC Bank Ltd",
                "Sector": "Financial Services",
                "Industry": "Banks - Regional",
                "Price": 1650.0,
                "Change_%": 1.2,
                "PE": 18.5,
                "PB": 2.6,
                "EPS": 89.2,
                "Book_Value": 634.6,
                "ROE_%": 16.8,
                "ROA_%": 2.1,
                "NIM_%": 4.1,
                "GNPA_%": 1.2,
                "CAR_%": 18.5,
                "DE": 720.0,
                "Rev_Growth_%": 16.0,
                "EPS_Growth_%": 18.0,
                "Rev_CAGR_3Y_%": 15.5,
                "EPS_CAGR_3Y_%": 17.0,
                "Mom_6m_1m_%": 12.0,
                "Headline_Count": 5,
                "FinBERT_Score": 0.45,
            },
            {
                "Ticker": "INFY",
                "Name": "Infosys Ltd",
                "Sector": "Technology",
                "Industry": "Information Technology Services",
                "Price": 1520.0,
                "Change_%": 0.8,
                "PE": 24.0,
                "PB": 7.5,
                "EPS": 63.3,
                "Book_Value": 202.7,
                "ROE_%": 31.0,
                "ROA_%": 19.0,
                "DE": 10.0,
                "Rev_Growth_%": 7.0,
                "EPS_Growth_%": 9.0,
                "Rev_CAGR_3Y_%": 8.5,
                "EPS_CAGR_3Y_%": 10.0,
                "Mom_6m_1m_%": 8.5,
                "Headline_Count": 4,
                "FinBERT_Score": 0.20,
            },
            {
                "Ticker": "HIGHPE",
                "Name": "High Multiple Corp",
                "Sector": "Industrials",
                "Industry": "Specialty Industrial Machinery",
                "Price": 2400.0,
                "Change_%": -0.5,
                "PE": 145.0,
                "PB": 18.0,
                "EPS": 16.5,
                "Book_Value": 133.3,
                "EV_EBITDA": 22.0,
                "PS": 6.5,
                "ROE_%": 22.0,
                "ROA_%": 11.0,
                "DE": 25.0,
                "Rev_Growth_%": 180.0,
                "EPS_Growth_%": 22.0,
                "Rev_CAGR_3Y_%": 24.0,
                "EPS_CAGR_3Y_%": 20.0,
                "Mom_6m_1m_%": 19.0,
                "Headline_Count": 2,
                "FinBERT_Score": 0.80,
            },
            {
                "Ticker": "LOSSCO",
                "Name": "Loss Making Ltd",
                "Sector": "Consumer Cyclical",
                "Industry": "Apparel Retail",
                "Price": 310.0,
                "Change_%": -2.1,
                "PE": np.nan,
                "PB": 4.2,
                "EPS": -8.5,
                "Book_Value": 73.8,
                "ROE_%": -11.5,
                "ROA_%": -6.0,
                "DE": 65.0,
                "Rev_Growth_%": 14.0,
                "EPS_Growth_%": -40.0,
                "Rev_CAGR_3Y_%": 12.0,
                "EPS_CAGR_3Y_%": np.nan,
                "Mom_6m_1m_%": -9.0,
                "Headline_Count": 3,
                "FinBERT_Score": -0.35,
            },
            {
                "Ticker": "NEGEQ",
                "Name": "Negative Equity Corp",
                "Sector": "Communication Services",
                "Industry": "Telecom Services",
                "Price": 14.5,
                "Change_%": 0.0,
                "PE": 4.0,
                "PB": -1.2,
                "EPS": 3.6,
                "Book_Value": -12.0,
                "ROE_%": np.nan,
                "ROA_%": -4.0,
                "DE": np.nan,
                "Rev_Growth_%": 3.0,
                "EPS_Growth_%": 5.0,
                "Rev_CAGR_3Y_%": 2.0,
                "EPS_CAGR_3Y_%": 1.0,
                "Mom_6m_1m_%": -15.0,
                "Headline_Count": 1,
                "FinBERT_Score": -0.60,
            },
        ]
    )


def make_price_panel(n_days: int = 140, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Create deterministic daily prices across trading days for tracker tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-02", periods=n_days)
    syms = ["HDFCBANK", "INFY", "HIGHPE", "LOSSCO", "NEGEQ"]

    data = {}
    for idx, sym in enumerate(syms):
        drift = 0.0008 + idx * 0.0002
        shocks = rng.normal(drift, 0.012, size=n_days)
        base = 100.0 * (idx + 1)
        data[sym] = base * np.cumprod(1.0 + shocks)

    close_df = pd.DataFrame(data, index=dates)
    nifty_shocks = rng.normal(0.0006, 0.009, size=n_days)
    nifty = pd.Series(24000.0 * np.cumprod(1.0 + nifty_shocks), index=dates, name="^NSEI")
    return close_df, nifty


def make_history_frame(dates: list[str], model_version: str = "2026.11.4") -> pd.DataFrame:
    """Create multi-date scored snapshot rows for testing tracker and history merges."""
    base_df = make_raw_universe()
    rows = []
    for d_idx, dt in enumerate(dates):
        for s_idx, rec in enumerate(base_df.to_dict("records")):
            rankable = rec["Ticker"] not in {"LOSSCO", "NEGEQ"}
            comp = np.nan if not rankable else round(82.0 - s_idx * 11.0 + (d_idx % 3) * 1.5, 1)
            if not rankable:
                flag = "NM (Loss-Making)" if rec["Ticker"] == "LOSSCO" else "NM (Neg. Equity)"
            elif s_idx == 0:
                flag = "Passes Multi-Factor Screen"
            elif s_idx == 1:
                flag = "Neutral / Watchlist"
            else:
                flag = "Fails Screen / Lagging"

            rows.append(
                {
                    "As_Of_Date": dt,
                    "Run_Timestamp": f"{dt} 16:00:00 UTC",
                    "Model_Version": model_version,
                    "Ticker": rec["Ticker"],
                    "Sector": rec["Sector"],
                    "Industry": rec["Industry"],
                    "Price": rec["Price"] * (1.0 + 0.005 * d_idx),
                    "Change_%": rec["Change_%"],
                    "PE": rec["PE"],
                    "PB": rec["PB"],
                    "EPS": rec["EPS"],
                    "Book_Value": rec["Book_Value"],
                    "ROE_%": rec["ROE_%"],
                    "DE": rec["DE"],
                    "Rev_Growth_%": rec["Rev_Growth_%"],
                    "EPS_Growth_%": rec["EPS_Growth_%"],
                    "Rev_CAGR_3Y_%": rec["Rev_CAGR_3Y_%"],
                    "EPS_CAGR_3Y_%": rec["EPS_CAGR_3Y_%"],
                    "Mom_6m_1m_%": rec["Mom_6m_1m_%"],
                    "Headline_Count": rec["Headline_Count"],
                    "FinBERT_Score": rec["FinBERT_Score"],
                    "Val_Tag": flag if not rankable else ("High-Multiple (>100x P/E)" if rec["Ticker"] == "HIGHPE" else "Valid"),
                    "Val_Metric_Type": "P/B" if rec["Sector"] == "Financial Services" else ("EV/EBITDA" if rec["Ticker"] == "HIGHPE" else "P/E"),
                    "Val_Metric": rec["PB"] if rec["Sector"] == "Financial Services" else (rec["EV_EBITDA"] if rec["Ticker"] == "HIGHPE" else rec["PE"]),
                    "Val_Rank": 75.0 - s_idx * 8.0,
                    "Qual_Rank": 78.0 - s_idx * 7.0,
                    "Growth_Rank": 72.0 - s_idx * 6.0,
                    "Sent_Score": 50.0 + (rec["FinBERT_Score"] or 0.0) * 25.0,
                    "Mom_Sent_Rank": 70.0 - s_idx * 5.0,
                    "Base_Effect_Flag": "Base Effect" if rec["Ticker"] == "HIGHPE" else "Normal",
                    "Rankable": rankable,
                    "Composite_Score": comp,
                    "Screen_Flag": flag,
                }
            )
    return pd.DataFrame(rows)
