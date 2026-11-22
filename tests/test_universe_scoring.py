"""tests/test_universe_scoring.py - unit tests for universe.py and scoring.py."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scoring import MODEL_VERSION, WEIGHTS, score_universe
from tests.synthetic import make_raw_universe
from universe import build_universe, to_symbol


def test_symbol_normalization_and_universe():
    assert to_symbol("INFY.NS") == "INFY"
    assert to_symbol("  tcs.ns ") == "TCS"
    assert to_symbol("INFY") == "INFY"


def test_build_universe_deduplicates_and_formats():
    u = build_universe(extra=["infy", "RELIANCE.NS", "NEWSTOCK"])
    assert len(u) == len(set(u))
    assert all(isinstance(t, str) and len(t) > 0 for t in u)
    assert any("NEWSTOCK" in t for t in u)


def test_score_universe_core_rules():
    raw = make_raw_universe()
    scored = score_universe(raw, weights=WEIGHTS)

    assert len(scored) == len(raw)
    assert (scored["Model_Version"] == MODEL_VERSION).all()

    by_ticker = scored.set_index("Ticker")

    # Bank uses P/B valuation
    assert by_ticker.loc["HDFCBANK", "Val_Metric_Type"] == "P/B"
    assert bool(by_ticker.loc["HDFCBANK", "Rankable"]) is True

    # High P/E (>100x) falls back to EV/EBITDA or P/S and flags base-effect growth
    assert by_ticker.loc["HIGHPE", "Val_Tag"] == "High-Multiple (>100x P/E)"
    assert by_ticker.loc["HIGHPE", "Val_Metric_Type"] in {"EV/EBITDA", "P/S"}
    assert by_ticker.loc["HIGHPE", "Base_Effect_Flag"] == "Base Effect"

    # Loss-making and negative-equity stocks are excluded (NM)
    assert bool(by_ticker.loc["LOSSCO", "Rankable"]) is False
    assert by_ticker.loc["LOSSCO", "Screen_Flag"] == "NM (Loss-Making)"
    assert np.isnan(by_ticker.loc["LOSSCO", "Composite_Score"])

    assert bool(by_ticker.loc["NEGEQ", "Rankable"]) is False
    assert by_ticker.loc["NEGEQ", "Screen_Flag"] == "NM (Neg. Equity)"
    assert np.isnan(by_ticker.loc["NEGEQ", "Composite_Score"])


def test_custom_weights_normalize_cleanly():
    raw = make_raw_universe()
    s1 = score_universe(raw, weights={"value": 0.25, "quality": 0.30, "growth": 0.20, "momentum_sentiment": 0.25})
    s2 = score_universe(raw, weights={"value": 25, "quality": 30, "growth": 20, "momentum_sentiment": 25})
    pd.testing.assert_series_equal(s1["Composite_Score"], s2["Composite_Score"])
