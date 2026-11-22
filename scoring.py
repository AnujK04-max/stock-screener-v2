"""
scoring.py - multi-factor scoring for the Indian stock screener
================================================================
Pure functions (pandas + numpy only, no network) so every rule can be unit-tested.

Usage
-----
    from scoring import score_universe
    scored = score_universe(raw_df)

``raw_df`` has one row per stock with RAW inputs. Do not pre-compute valuation tags and
do not turn negative multiples into NaN - the classification happens here.

    required: Ticker, Sector, Industry, PE, PB, EPS, Book_Value, ROE_%, DE,
              Rev_Growth_%, EPS_Growth_%, Mom_6m_1m_%, Headline_Count, FinBERT_Score
    optional: ROA_%, EV_EBITDA, PS, Rev_CAGR_3Y_%, EPS_CAGR_3Y_%, NIM_%, GNPA_%, CAR_%

Growth and quality columns are in percent. ``DE`` is yfinance's debtToEquity (percent).

What changed versus the first multi-factor version
--------------------------------------------------
1. VALUE   - each metric type (P/E, P/B, EV/EBITDA, P/S) is ranked only against the same
             metric type. The rank blends a within-industry rank (shrunk toward the
             pool-wide rank when the industry is small) with the pool-wide rank, so a stock
             is never "100 vs 50" on near-identical multiples. No winsorizing: percentile
             ranks already ignore magnitude, so clipping only creates ties.
2. GROWTH  - base-effect detection, per metric (revenue and EPS are judged separately).
             A flagged metric uses its 3Y CAGR when available and is neutralised (50)
             otherwise, instead of letting a one-off jump drive the rank.
3. QUALITY - lenders are never ranked on D/E. Financials are scored on ROE, ROA (plus NIM,
             GNPA, CAR when supplied) against financials only, by sub-type. Non-financials
             use ROE and D/E against non-financials only.
4. NM      - loss-makers, negative equity and missing data are "not rankable": no composite,
             no multiplier, excluded from the quintile cut. P/E > 100 names are scored on
             EV/EBITDA (or P/S) instead of being penalised.
5. SENTIMENT - absolute FinBERT score shrunk by headline count (not a rank), and a lower
             weight than momentum, because coverage is thin.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

MODEL_VERSION = "2026.11.4"   # bumped: per-metric base-effect flags, word-boundary industry types

WEIGHTS = {"value": 0.25, "quality": 0.30, "growth": 0.20, "mom_sent": 0.25}

FIN_SECTOR = "Financial Services"
PRIOR_K = 5.0                 # shrinkage strength: weight on a group = n / (n + k)
RELATIVE_WEIGHT = 0.5         # blend of within-industry rank vs pool-wide rank
HIGH_PE = 100.0               # non-financials above this are scored on another multiple
EV_EBITDA_CAP = 150.0         # EV/EBITDA above this is treated as unusable

BASE_EFFECT_REV = 60.0        # |TTM revenue growth| above this (%) is flagged
BASE_EFFECT_EPS = 75.0        # |TTM EPS growth| above this (%) is flagged
CAGR_GAP = 50.0               # TTM vs 3Y CAGR gap (percentage points) that is flagged
# Known one-off distortions, e.g. {"TICKER": "reason"}. Flagged regardless of thresholds.
BASE_EFFECT_OVERRIDES: dict = {}

MIN_HEADLINES = 3             # fewer scored headlines than this -> neutral sentiment
SENT_PRIOR = 3.0              # sentiment shrink = n / (n + SENT_PRIOR)
MOM_SHARE = 0.75              # share of Mom_Sent_Rank from price momentum

QUINTILE = 0.80

REQUIRED_COLUMNS = [
    "Ticker", "Sector", "Industry", "PE", "PB", "EPS", "Book_Value", "ROE_%", "DE",
    "Rev_Growth_%", "EPS_Growth_%", "Mom_6m_1m_%", "Headline_Count", "FinBERT_Score",
]
OPTIONAL_COLUMNS = [
    "ROA_%", "EV_EBITDA", "PS", "Rev_CAGR_3Y_%", "EPS_CAGR_3Y_%",
    "NIM_%", "GNPA_%", "CAR_%",
]


# --------------------------------------------------------------------------- helpers
def _f(x) -> float:
    """Coerce to a finite float, else NaN."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return np.nan
    return v if np.isfinite(v) else np.nan


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def pct_rank(series, higher_is_better: bool = True) -> pd.Series:
    """Percentile rank in (0, 100]; ties share the average rank; NaN stays NaN."""
    return _num(series).rank(pct=True, ascending=higher_is_better, method="average") * 100.0


def blended_rank(values, groups, higher_is_better: bool = True,
                 prior_k: float = PRIOR_K,
                 relative_weight: float = RELATIVE_WEIGHT) -> pd.Series:
    """
    Rank ``values`` (all one metric type) with industry context but without small-group noise.

- pool rank      - percentile among every stock in ``values``
- group rank     - percentile within the stock's group, pulled toward the pool rank by
                       n / (n + k), so a 2-stock group barely moves the result
- result         - relative_weight * group + (1 - relative_weight) * pool, then pulled
                       toward 50 if the whole pool is tiny
    """
    v = _num(values)
    out = pd.Series(np.nan, index=v.index, dtype=float)
    valid = v.dropna()
    if valid.empty:
        return out
    pool = pct_rank(valid, higher_is_better)
    rel = pd.Series(np.nan, index=valid.index, dtype=float)
    for _, idx in valid.groupby(groups.loc[valid.index]).groups.items():
        sub = valid.loc[idx]
        grp = pct_rank(sub, higher_is_better)
        n = float(len(sub))
        w = n / (n + prior_k)
        rel.loc[idx] = w * grp + (1.0 - w) * pool.loc[idx]
    blended = relative_weight * rel + (1.0 - relative_weight) * pool
    pool_w = len(valid) / (len(valid) + prior_k)
    out.loc[valid.index] = pool_w * blended + (1.0 - pool_w) * 50.0
    return out


def _industry_key(df: pd.DataFrame) -> pd.Series:
    ind = df["Industry"].astype("object")
    blank = ind.isna() | (ind.astype(str).str.strip() == "")
    key = ind.where(~blank, df["Sector"].astype("object"))
    return key.fillna("General").astype(str)


def _financial_type(industry) -> str:
    """Bank / Insurance / Lender-Other from Yahoo's industry text (whole words, not substrings)."""
    s = str(industry or "").lower()
    if re.search(r"\bbanks?\b", s):            # "Banks - Regional" yes; "Investment Banking" no
        return "Bank"
    if re.search(r"\binsurance\b", s) and "broker" not in s:   # brokers do not underwrite risk
        return "Insurance"
    return "Lender / Other"


# --------------------------------------------------------------------------- valuation
def classify_valuation(row: dict):
    """Return (tag, metric_type, metric_value, reason) for one stock."""
    is_fin = row.get("Sector") == FIN_SECTOR
    pe, pb = _f(row.get("PE")), _f(row.get("PB"))
    eps, bv = _f(row.get("EPS")), _f(row.get("Book_Value"))
    ev, ps, roe = _f(row.get("EV_EBITDA")), _f(row.get("PS")), _f(row.get("ROE_%"))

    if (not np.isnan(bv) and bv <= 0) or (not np.isnan(pb) and pb <= 0):
        return "NM (Neg. Equity)", "", np.nan, "Book value is zero or negative"
    if (not np.isnan(eps) and eps <= 0) or (
        np.isnan(eps) and np.isnan(pe) and not np.isnan(roe) and roe < 0
    ):
        return "NM (Loss-Making)", "", np.nan, "Trailing earnings are not positive"

    if is_fin:
        if not np.isnan(pb) and pb > 0:
            return "Valid", "P/B", pb, ""
        return "Insufficient Data", "", np.nan, "No usable P/B for a financial"

    if not np.isnan(pe) and pe > HIGH_PE:
        tag = "High-Multiple (>100x P/E)"
        if 0 < ev <= EV_EBITDA_CAP:
            return tag, "EV/EBITDA", ev, ""
        if ps > 0:
            return tag, "P/S", ps, ""
        return tag, "", np.nan, "No alternative multiple available"
    if not np.isnan(pe) and pe > 0:
        return "Valid", "P/E", pe, ""
    return "Insufficient Data", "", np.nan, "No usable P/E"


# --------------------------------------------------------------------------- growth
def _growth_values(df: pd.DataFrame):
    """
    Return (rev_value, eps_value, flagged, note) after base-effect handling.

    Flags are PER METRIC: a low-base EPS spike must not throw away a normal revenue figure.
    For a flagged metric: use its 3Y CAGR if available, otherwise neutralise it (NaN -> rank 50).
    For an unflagged metric: average TTM and CAGR when both exist, else whichever exists.
    """
    rev_t, eps_t = _num(df["Rev_Growth_%"]), _num(df["EPS_Growth_%"])
    rev_c, eps_c = _num(df["Rev_CAGR_3Y_%"]), _num(df["EPS_CAGR_3Y_%"])
    override = df["Ticker"].isin(list(BASE_EFFECT_OVERRIDES))

    flag_rev = (rev_t.abs() > BASE_EFFECT_REV) | ((rev_t - rev_c).abs() > CAGR_GAP) | override
    flag_eps = (eps_t.abs() > BASE_EFFECT_EPS) | ((eps_t - eps_c).abs() > CAGR_GAP) | override

    def combine(ttm: pd.Series, cagr: pd.Series, flag: pd.Series) -> pd.Series:
        out = ttm.copy()                              # default: TTM only
        out[cagr.notna()] = cagr[cagr.notna()]        # CAGR where we have it
        both = ttm.notna() & cagr.notna() & ~flag     # unflagged and both: 50/50 blend
        out[both] = 0.5 * ttm[both] + 0.5 * cagr[both]
        out[flag & cagr.isna()] = np.nan              # flagged with no CAGR: neutralise
        return out

    note = pd.Series("", index=df.index, dtype=object)
    for i in df.index:
        parts = []
        for label, f, c in (("revenue", flag_rev, rev_c), ("EPS", flag_eps, eps_c)):
            if f[i]:
                parts.append(f"{label} growth " + ("ranked on 3Y CAGR" if pd.notna(c[i]) else "neutralised"))
        if parts:
            note[i] = "Flagged: " + "; ".join(parts)
        if df.at[i, "Ticker"] in BASE_EFFECT_OVERRIDES:
            note[i] = f"Override: {BASE_EFFECT_OVERRIDES[df.at[i, 'Ticker']]}"
    return combine(rev_t, rev_c, flag_rev), combine(eps_t, eps_c, flag_eps), flag_rev | flag_eps, note


# --------------------------------------------------------------------------- main entry
def score_universe(raw: pd.DataFrame, weights: Optional[dict] = None) -> pd.DataFrame:
    weights = dict(weights or WEIGHTS)
    if set(weights) != set(WEIGHTS) or abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError(f"weights must have keys {sorted(WEIGHTS)} and sum to 1")
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = raw.copy().reset_index(drop=True)
    for c in OPTIONAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    if df.empty:
        return df

    industry = _industry_key(df)
    is_fin = df["Sector"].eq(FIN_SECTOR)

    # ---- 1. value ---------------------------------------------------------------
    cls = [classify_valuation(r) for r in df.to_dict("records")]
    df["Val_Tag"] = [c[0] for c in cls]
    df["Val_Metric_Type"] = [c[1] for c in cls]
    df["Val_Metric"] = [c[2] for c in cls]
    df["Not_Rankable_Reason"] = [
        c[3] if (c[0].startswith("NM") or c[0] == "Insufficient Data") else "" for c in cls
    ]
    rankable = df["Not_Rankable_Reason"].eq("")

    df["Val_Rank"] = np.nan
    has_metric = df["Val_Metric"].notna()
    for mtype in sorted(df.loc[has_metric, "Val_Metric_Type"].unique()):
        idx = df.index[has_metric & df["Val_Metric_Type"].eq(mtype)]
        df.loc[idx, "Val_Rank"] = blended_rank(
            df.loc[idx, "Val_Metric"], industry.loc[idx], higher_is_better=False
        )
    df.loc[rankable, "Val_Rank"] = df.loc[rankable, "Val_Rank"].fillna(50.0)

    # ---- 2. quality -------------------------------------------------------------
    qual = pd.Series(np.nan, index=df.index, dtype=float)

    nf = ~is_fin
    roe_r = pd.Series(np.nan, index=df.index, dtype=float)
    de_r = pd.Series(np.nan, index=df.index, dtype=float)
    roe_r[nf] = pct_rank(df.loc[nf, "ROE_%"], True)
    de_clean = _num(df["DE"]).where(_num(df["DE"]) >= 0)
    de_r[nf] = pct_rank(de_clean[nf], False)
    qual[nf] = 0.6 * roe_r[nf].fillna(50.0) + 0.4 * de_r[nf].fillna(50.0)

    if is_fin.any():
        fin_type = df["Industry"].map(_financial_type)
        parts = {}
        for col, hib in [("ROE_%", True), ("ROA_%", True), ("NIM_%", True),
                         ("GNPA_%", False), ("CAR_%", True)]:
            vals = df.loc[is_fin, col]
            if _num(vals).notna().any():
                parts[col] = blended_rank(vals, fin_type[is_fin], higher_is_better=hib)
        if parts:
            qual[is_fin] = pd.DataFrame(parts).mean(axis=1, skipna=True)
    df["Qual_Rank"] = qual.fillna(50.0)

    # ---- 3. growth --------------------------------------------------------------
    rev_v, eps_v, flagged, note = _growth_values(df)
    growth = pd.concat([pct_rank(rev_v, True), pct_rank(eps_v, True)], axis=1).mean(
        axis=1, skipna=True
    )
    df["Growth_Rank"] = growth.fillna(50.0)
    df["Base_Effect_Flag"] = np.where(flagged, "Base Effect", "Normal")
    df["Base_Effect_Note"] = note

    # ---- 4. momentum + sentiment ------------------------------------------------
    mom_r = pct_rank(df["Mom_6m_1m_%"], True).fillna(50.0)
    n_head = _num(df["Headline_Count"]).fillna(0.0)
    score = _num(df["FinBERT_Score"])
    usable = score.notna() & (n_head >= MIN_HEADLINES)
    shrunk = score * n_head / (n_head + SENT_PRIOR)
    df["Sent_Score"] = (50.0 + 50.0 * shrunk).clip(0.0, 100.0).where(usable, 50.0)
    df["Mom_Sent_Rank"] = MOM_SHARE * mom_r + (1.0 - MOM_SHARE) * df["Sent_Score"]

    # ---- 5. composite and flags -------------------------------------------------
    comp = (
        weights["value"] * df["Val_Rank"]
        + weights["quality"] * df["Qual_Rank"]
        + weights["growth"] * df["Growth_Rank"]
        + weights["mom_sent"] * df["Mom_Sent_Rank"]
    )
    df["Composite_Score"] = comp.where(rankable).round(1)

    cutoff = (
        df.loc[rankable, "Composite_Score"].quantile(QUINTILE) if rankable.any() else np.nan
    )
    cs = df["Composite_Score"]
    df["Screen_Flag"] = np.select(
        [~rankable, cs >= cutoff, cs >= 50.0],
        [df["Val_Tag"].to_numpy(), "Passes Multi-Factor Screen", "Neutral / Watchlist"],
        default="Fails Screen / Lagging",
    )

    df["Rankable"] = rankable.to_numpy()
    df["Model_Version"] = MODEL_VERSION
    df = df.assign(_s=df["Composite_Score"].fillna(-1.0)).sort_values(
        "_s", ascending=False, kind="stable"
    ).drop(columns="_s").reset_index(drop=True)
    df["Rank"] = df["Composite_Score"].rank(ascending=False, method="first")
    return df
