"""fetchers.py - market data, fundamentals, news headlines, and FinBERT scoring."""
from __future__ import annotations

import math
import re
import time
from typing import Callable, Iterable

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

from growth_data import fetch_cagr_3y
from universe import to_symbol

NEWS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]


def _f(val) -> float:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def extract_as_of_date(nifty_series: pd.Series) -> str | None:
    """Return ISO date (YYYY-MM-DD) of the latest valid Nifty 50 close."""
    if nifty_series is None or len(nifty_series) == 0:
        return None
    s = pd.Series(nifty_series).dropna()
    if s.empty:
        return None
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize().max().strftime("%Y-%m-%d")


def compute_momentum_6m_1m(close_series: pd.Series) -> float:
    """Compute price return from ~6 months ago (126 trading days) to ~1 month ago (21 trading days)."""
    if close_series is None or len(close_series) == 0:
        return float("nan")
    s = pd.to_numeric(pd.Series(close_series).dropna(), errors="coerce").dropna()
    if len(s) < 25:
        return float("nan")
    end_idx = -22 if len(s) >= 23 else -1
    start_idx = -127 if len(s) >= 127 else 0
    start_p = _f(s.iloc[start_idx])
    end_p = _f(s.iloc[end_idx])
    if not (math.isfinite(start_p) and math.isfinite(end_p)) or start_p <= 0:
        return float("nan")
    return float((end_p / start_p - 1.0) * 100.0)


def fetch_price_panels(
    tickers: Iterable[str],
    benchmark: str = "^NSEI",
    period: str = "1y",
    retries: int = 2,
    pause: float = 0.25,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download adjusted/close prices for tickers and benchmark; return (close_df, nifty_series)."""
    t_list = [str(t).strip().upper() for t in tickers if str(t).strip()]
    syms = [to_symbol(t) for t in t_list]
    download_syms = list(dict.fromkeys(t_list + ([benchmark] if benchmark else [])))

    raw = None
    attempts = max(1, int(retries))
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                tickers=download_syms,
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and len(raw) > 0:
                break
        except Exception:
            if attempt >= attempts:
                raise
        if attempt < attempts and pause > 0:
            time.sleep(pause)

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=syms), pd.Series(dtype=float)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            prices = raw.xs(raw.columns.levels[0][0], axis=1, level=0).copy()
    else:
        prices = raw.to_frame(name=download_syms[0]) if isinstance(raw, pd.Series) else raw.copy()

    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    prices.index = idx.normalize()
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()

    nifty = pd.Series(dtype=float)
    if benchmark and benchmark in prices.columns:
        nifty = pd.to_numeric(prices[benchmark], errors="coerce").dropna()

    close_df = pd.DataFrame(index=prices.index)
    for t, sym in zip(t_list, syms):
        if t in prices.columns:
            close_df[sym] = pd.to_numeric(prices[t], errors="coerce")
        elif sym in prices.columns:
            close_df[sym] = pd.to_numeric(prices[sym], errors="coerce")
        else:
            close_df[sym] = np.nan

    return close_df, nifty


def _extract_fundamentals_from_info(ticker: str, info: dict, close_df: pd.DataFrame, cagr_fn: Callable) -> dict:
    sym = to_symbol(ticker)
    price = _f(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    prev_close = _f(info.get("previousClose") or info.get("regularMarketPreviousClose"))
    chg_pct = _f(info.get("regularMarketChangePercent"))
    if not math.isfinite(chg_pct):
        if math.isfinite(price) and math.isfinite(prev_close) and prev_close > 0:
            chg_pct = (price / prev_close - 1.0) * 100.0
        else:
            chg_pct = float("nan")

    mom = float("nan")
    if not close_df.empty and sym in close_df.columns:
        if not math.isfinite(price):
            s = pd.to_numeric(close_df[sym].dropna(), errors="coerce").dropna()
            if not s.empty:
                price = _f(s.iloc[-1])
                if len(s) >= 2 and _f(s.iloc[-2]) > 0 and not math.isfinite(chg_pct):
                    chg_pct = (_f(s.iloc[-1]) / _f(s.iloc[-2]) - 1.0) * 100.0
        mom = compute_momentum_6m_1m(close_df[sym])

    roe = _f(info.get("returnOnEquity"))
    if math.isfinite(roe) and abs(roe) <= 1.5:
        roe = roe * 100.0

    roa = _f(info.get("returnOnAssets"))
    if math.isfinite(roa) and abs(roa) <= 1.5:
        roa = roa * 100.0

    rev_g = _f(info.get("revenueGrowth"))
    if math.isfinite(rev_g) and abs(rev_g) <= 2.5:
        rev_g = rev_g * 100.0

    eps_g = _f(info.get("earningsGrowth"))
    if not math.isfinite(eps_g):
        eps_g = _f(info.get("earningsQuarterlyGrowth"))
    if math.isfinite(eps_g) and abs(eps_g) <= 2.5:
        eps_g = eps_g * 100.0

    cagr_3y = {"Rev_CAGR_3Y_%": float("nan"), "EPS_CAGR_3Y_%": float("nan")}
    if cagr_fn is not None:
        try:
            res = cagr_fn(ticker)
            if isinstance(res, dict):
                cagr_3y.update(res)
        except Exception:
            pass

    return {
        "Ticker": sym,
        "Name": info.get("shortName") or info.get("longName") or sym,
        "Sector": info.get("sector") or "Unknown",
        "Industry": info.get("industry") or "Unknown",
        "Price": price,
        "Change_%": chg_pct,
        "PE": _f(info.get("trailingPE")),
        "PB": _f(info.get("priceToBook")),
        "EPS": _f(info.get("trailingEps")),
        "Book_Value": _f(info.get("bookValue")),
        "EV_EBITDA": _f(info.get("enterpriseToEbitda")),
        "PS": _f(info.get("priceToSalesTrailing12Months")),
        "ROE_%": roe,
        "ROA_%": roa,
        "DE": _f(info.get("debtToEquity")),
        "Rev_Growth_%": rev_g,
        "EPS_Growth_%": eps_g,
        "Rev_CAGR_3Y_%": _f(cagr_3y.get("Rev_CAGR_3Y_%")),
        "EPS_CAGR_3Y_%": _f(cagr_3y.get("EPS_CAGR_3Y_%")),
        "Mom_6m_1m_%": mom,
        "Avg_Volume_20D": _f(info.get("averageDailyVolume3Month") or info.get("averageVolume")),
    }


def fetch_fundamentals(
    tickers: Iterable[str],
    close_df: pd.DataFrame | None = None,
    cagr_fn: Callable | None = fetch_cagr_3y,
    retries: int = 2,
    pause: float = 0.25,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch fundamentals for each ticker; return (df, skipped_list)."""
    c_df = close_df if close_df is not None else pd.DataFrame()
    rows = []
    skipped = []
    attempts = max(1, int(retries))

    for t in tickers:
        ticker = str(t).strip().upper()
        if not ticker:
            continue
        info = None
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                tk = yf.Ticker(ticker)
                info = getattr(tk, "info", None) or {}
                if isinstance(info, dict) and info:
                    break
            except Exception as exc:
                last_err = exc
            if attempt < attempts and pause > 0:
                time.sleep(pause)

        if not isinstance(info, dict) or not info:
            skipped.append({"ticker": ticker, "reason": f"Missing fundamentals ({last_err or 'empty info'})"})
            continue

        row = _extract_fundamentals_from_info(ticker, info, c_df, cagr_fn)
        if not math.isfinite(row["Price"]) or row["Price"] <= 0:
            skipped.append({"ticker": ticker, "reason": "Missing or non-positive price"})
            continue
        rows.append(row)

    return pd.DataFrame(rows), skipped


def _matches_ticker(text: str, sym: str, name: str) -> bool:
    low = text.lower()
    s_low = sym.lower()
    if re.search(r"(?<![a-z0-9])" + re.escape(s_low) + r"(?![a-z0-9])", low):
        return True
    first_word = re.split(r"[\s\-&]+", (name or "").strip().lower())[0]
    if len(first_word) >= 4 and re.search(r"(?<![a-z0-9])" + re.escape(first_word) + r"(?![a-z0-9])", low):
        return True
    return False


def fetch_headlines(
    records: list[dict],
    feeds: Iterable[str] = NEWS_FEEDS,
    max_per_stock: int = 8,
) -> dict[str, list[str]]:
    """Collect recent headline titles matching each stock from RSS feeds."""
    titles = []
    for url in feeds:
        try:
            parsed = feedparser.parse(url)
            for entry in getattr(parsed, "entries", []) or []:
                title = str(getattr(entry, "title", "") or "").strip()
                if title and title not in titles:
                    titles.append(title)
        except Exception:
            continue

    out: dict[str, list[str]] = {}
    for rec in records:
        sym = str(rec.get("Ticker") or "").strip().upper()
        name = str(rec.get("Name") or "")
        if not sym:
            continue
        matched = [t for t in titles if _matches_ticker(t, sym, name)]
        out[sym] = matched[:max_per_stock]
    return out


def score_headlines_finbert(
    headlines_by_ticker: dict[str, list[str]],
    classifier: Callable | None = None,
) -> dict[str, tuple[int, float | None, list[dict]]]:
    """Score headlines with FinBERT (or a test classifier); return {ticker: (count, avg_score, items)}."""
    clf = classifier
    if clf is None:
        try:
            from transformers import pipeline

            clf = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception:
            clf = None

    results: dict[str, tuple[int, float | None, list[dict]]] = {}
    for ticker, heads in headlines_by_ticker.items():
        clean_heads = [str(h).strip() for h in (heads or []) if str(h).strip()]
        if not clean_heads:
            results[ticker] = (0, None, [])
            continue

        if clf is None:
            results[ticker] = (len(clean_heads), None, [{"t": h, "s": None} for h in clean_heads])
            continue

        preds = clf(clean_heads)
        items = []
        numeric_scores = []
        for h, p in zip(clean_heads, preds):
            label = str((p or {}).get("label", "")).lower()
            conf = _f((p or {}).get("score"))
            signed = None
            if math.isfinite(conf):
                if "pos" in label:
                    signed = conf
                elif "neg" in label:
                    signed = -conf
                else:
                    signed = 0.0
                numeric_scores.append(signed)
            items.append({"t": h, "s": signed})

        avg_score = float(np.mean(numeric_scores)) if numeric_scores else None
        results[ticker] = (len(items), avg_score, items)

    return results
