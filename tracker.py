"""
tracker.py - does the screen actually work? Forward performance of each day's picks.

Method (all of it is shown on the page):
- A *cohort* is one day's snapshot from the history log, evaluated for ONE model version only.
- Entry and exit prices both come from the same adjusted price download (not the logged price),
  so splits and dividends cannot distort a return. Windows are counted in trading days using the
  Nifty calendar, starting from the last complete session on or before the cohort date.
- Benchmarks: (1) equal-weight average of every rankable stock in that cohort - the fair
  comparison because it uses the same stocks and the same price basis; (2) the Nifty 50 price
  index, which excludes dividends and so flatters the stock baskets slightly.
- Cohorts closer together than the horizon overlap, so statistics use only non-overlapping ones.
- An assumed round-trip cost is subtracted from the top basket for every cohort (conservative:
  it charges a full round trip even when most of the basket is unchanged).
- With fewer than MIN_INDEPENDENT independent cohorts the result is labelled preliminary. A
  handful of cohorts cannot show skill; the numbers are there for transparency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = [(5, "1 week"), (21, "1 month"), (63, "3 months")]   # trading days
MIN_NAMES = 30            # rankable stocks with a valid return needed in a cohort
MIN_TOP = 5               # names in the top basket needed in a cohort
MIN_INDEPENDENT = 8       # independent cohorts before a result is called evaluable
ROUND_TRIP_COST_PCT = 0.25
TOP_FLAG = "Passes Multi-Factor Screen"
BOTTOM_FLAG = "Fails Screen / Lagging"


def _calendar(nifty: pd.Series) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(nifty.dropna().index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return pd.DatetimeIndex(sorted(set(idx.normalize())))


def _align(obj, calendar):
    out = obj.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx.normalize()
    out = out[~out.index.duplicated(keep="last")]
    return out.reindex(calendar)


def _position(calendar: pd.DatetimeIndex, as_of) -> int | None:
    pos = int(calendar.searchsorted(pd.Timestamp(as_of), side="right")) - 1
    if pos < 0 or (pd.Timestamp(as_of) - calendar[pos]).days > 5:
        return None
    return pos


def spearman(x, y) -> float:
    """Spearman rank correlation = Pearson correlation of average ranks (no scipy needed;
    pandas 2.x raises ModuleNotFoundError for corr(method="spearman") without it)."""
    a = pd.Series(np.asarray(x, dtype=float))
    b = pd.Series(np.asarray(y, dtype=float))
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 3 or a.nunique() < 2 or b.nunique() < 2:
        return float("nan")
    return float(a.rank(method="average").corr(b.rank(method="average")))


def _tstat(values) -> float | None:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 3:
        return None
    sd = v.std(ddof=1)
    if sd == 0:
        return None
    return float(v.mean() / (sd / np.sqrt(len(v))))


def select_non_overlapping(cohorts: list, horizon: int) -> list:
    """Greedy: keep cohorts whose entry is at least ``horizon`` sessions after the last kept one."""
    kept, last = [], None
    for c in sorted(cohorts, key=lambda c: c["i0"]):
        if last is None or c["i0"] >= last + horizon:
            kept.append(c)
            last = c["i0"]
    return kept


def _cohort_result(cohort, close, nifty_vals, i0, i1):
    c = cohort.dropna(subset=["Composite_Score"])
    c = c[c["Ticker"].isin(close.columns)]
    if c.empty:
        return None
    p0 = close.iloc[i0][c["Ticker"]].to_numpy(dtype=float)
    p1 = close.iloc[i1][c["Ticker"]].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = (p1 / p0 - 1.0) * 100.0
    ok = np.isfinite(ret) & (p0 > 0)
    c = c.assign(ret=ret)[ok]
    if len(c) < MIN_NAMES:
        return None
    top = c[c["Screen_Flag"] == TOP_FLAG]
    if len(top) < MIN_TOP:
        return None
    bottom = c[c["Screen_Flag"] == BOTTOM_FLAG]
    n0, n1 = nifty_vals[i0], nifty_vals[i1]
    if not (np.isfinite(n0) and np.isfinite(n1) and n0 > 0):
        return None
    ic = spearman(c["Composite_Score"], c["ret"])
    return {
        "i0": i0,
        "top_gross": float(top["ret"].mean()),
        "bottom_gross": float(bottom["ret"].mean()) if len(bottom) else np.nan,
        "ew": float(c["ret"].mean()),
        "nifty": float((n1 / n0 - 1.0) * 100.0),
        "ic": float(ic) if np.isfinite(ic) else np.nan,
        "n_names": int(len(c)),
        "n_top": int(len(top)),
        "top_names": frozenset(top["Ticker"]),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def _round(x, nd=2):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def evaluate_horizon(hist, close, nifty_vals, calendar, horizon, label, cost_pct):
    dates = sorted(hist["As_Of_Date"].unique())
    cohorts = []
    for d in dates:
        i0 = _position(calendar, d)
        if i0 is None or i0 + horizon >= len(calendar):
            continue
        res = _cohort_result(hist[hist["As_Of_Date"] == d], close, nifty_vals, i0, i0 + horizon)
        if res is not None:
            res["as_of"] = d
            cohorts.append(res)

    indep = select_non_overlapping(cohorts, horizon)
    n = len(indep)
    for c in indep:
        c["top_net"] = c["top_gross"] - cost_pct
        c["excess_ew"] = c["top_net"] - c["ew"]
        c["excess_nifty"] = c["top_net"] - c["nifty"]
        c["spread"] = (c["top_gross"] - c["bottom_gross"]) if np.isfinite(c["bottom_gross"]) else np.nan

    turnovers = []
    for prev, cur in zip(indep, indep[1:]):
        turnovers.append(1.0 - len(prev["top_names"] & cur["top_names"]) / len(cur["top_names"]))

    status = "collecting" if not cohorts else ("preliminary" if n < MIN_INDEPENDENT else "evaluable")
    t = _tstat([c["excess_ew"] for c in indep])
    verdict = None
    if status == "evaluable":
        verdict = "inconclusive" if t is None else ("ahead" if t >= 2 else "behind" if t <= -2 else "inconclusive")
    ic_t = _tstat([c["ic"] for c in indep])

    eta = None
    if not cohorts and dates:
        eta = (pd.Timestamp(dates[0]) + pd.offsets.BDay(horizon)).strftime("%d %b %Y")

    recent = [
        {"as_of": c["as_of"], "top_net": _round(c["top_net"]), "ew": _round(c["ew"]),
         "nifty": _round(c["nifty"]), "excess_ew": _round(c["excess_ew"]), "ic": _round(c["ic"], 3),
         "n_top": c["n_top"]}
        for c in sorted(indep, key=lambda c: c["i0"], reverse=True)[:8]
    ]
    return {
        "days": horizon, "label": label, "status": status, "verdict": verdict,
        "n_matured": len(cohorts), "n_independent": n,
        "top_gross": _round(_mean([c["top_gross"] for c in indep])),
        "top_net": _round(_mean([c["top_net"] for c in indep])),
        "ew": _round(_mean([c["ew"] for c in indep])),
        "nifty": _round(_mean([c["nifty"] for c in indep])),
        "excess_ew": _round(_mean([c["excess_ew"] for c in indep])),
        "excess_nifty": _round(_mean([c["excess_nifty"] for c in indep])),
        "spread": _round(_mean([c["spread"] for c in indep])),
        "hit_rate_cohorts": _round(100.0 * float(np.mean([c["excess_ew"] > 0 for c in indep])), 1) if indep else None,
        "mean_ic": _round(_mean([c["ic"] for c in indep]), 3),
        "ic_t": _round(ic_t, 2), "t_stat": _round(t, 2),
        "turnover": _round(100.0 * float(np.mean(turnovers)), 0) if turnovers else None,
        "avg_top_names": _round(_mean([c["n_top"] for c in indep]), 1),
        "first_cohort": indep[0]["as_of"] if indep else None,
        "last_cohort": indep[-1]["as_of"] if indep else None,
        "eta": eta, "recent": recent,
    }


def build_tracker_summary(history, close, nifty, model_version,
                          cost_pct=ROUND_TRIP_COST_PCT, horizons=HORIZONS) -> dict:
    """
    history: frame with As_Of_Date, Model_Version, Ticker, Composite_Score, Screen_Flag.
    close:   adjusted closes, index = dates, columns = ticker symbols (no .NS).
    nifty:   Nifty 50 closes (also defines the trading calendar).
    """
    h = history.copy()
    h["Composite_Score"] = pd.to_numeric(h["Composite_Score"], errors="coerce")
    cur = h[h["Model_Version"] == model_version]
    dates = sorted(cur["As_Of_Date"].unique())
    summary = {
        "model_version": model_version, "cost_pct": cost_pct,
        "cohorts_logged": len(dates),
        "first_cohort": dates[0] if dates else None,
        "last_cohort": dates[-1] if dates else None,
        "legacy_cohorts": int(h.loc[h["Model_Version"] != model_version, "As_Of_Date"].nunique()),
        "horizons": [],
    }
    calendar = _calendar(nifty)
    if len(calendar) == 0 or cur.empty:
        summary["horizons"] = [_empty(d, lab, dates) for d, lab in horizons]
        return summary
    close_a = _align(close, calendar).ffill(limit=3)
    nifty_vals = _align(nifty, calendar).to_numpy(dtype=float)
    summary["horizons"] = [
        evaluate_horizon(cur, close_a, nifty_vals, calendar, d, lab, cost_pct) for d, lab in horizons
    ]
    return summary


def _empty(days, label, dates):
    eta = (pd.Timestamp(dates[0]) + pd.offsets.BDay(days)).strftime("%d %b %Y") if dates else None
    keys = ["top_gross", "top_net", "ew", "nifty", "excess_ew", "excess_nifty", "spread",
            "hit_rate_cohorts", "mean_ic", "ic_t", "t_stat", "turnover", "avg_top_names",
            "first_cohort", "last_cohort"]
    out = {k: None for k in keys}
    out.update({"days": days, "label": label, "status": "collecting", "verdict": None,
                "n_matured": 0, "n_independent": 0, "eta": eta, "recent": []})
    return out
