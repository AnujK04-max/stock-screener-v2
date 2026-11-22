"""tests/test_report.py - unit tests for report.py."""
from __future__ import annotations

from scoring import MODEL_VERSION, score_universe
from report import build_report_html
from tests.synthetic import make_raw_universe


def test_build_report_html_renders_sections_and_escapes():
    raw = make_raw_universe()
    raw.loc[0, "Name"] = 'HDFC Bank <script>alert("x")</script>'
    scored = score_universe(raw)

    tracker = {
        "model_version": MODEL_VERSION,
        "cohorts_logged": 12,
        "legacy_cohorts": 0,
        "horizons": [
            {"days": 5, "label": "1 week", "status": "evaluable", "n_independent": 10, "top_net": 2.4, "ew": 1.1, "nifty": 0.9},
            {"days": 21, "label": "1 month", "status": "preliminary", "n_independent": 4, "top_net": None, "ew": None, "nifty": None},
            {"days": 63, "label": "3 months", "status": "collecting", "n_independent": 1, "top_net": None, "ew": None, "nifty": None},
        ],
    }

    html_out = build_report_html(
        scored=scored,
        tracker=tracker,
        as_of_date="2026-11-20",
        updated_at="2026-11-20 16:00:00 UTC",
        model_version=MODEL_VERSION,
        skipped=[{"ticker": "BAD.NS", "reason": "Missing price"}],
    )

    assert "<!DOCTYPE html>" in html_out
    assert "NSE Multi-Factor &amp; FinBERT Screener" in html_out
    assert MODEL_VERSION in html_out
    assert "Forward Performance Tracker" in html_out
    assert "Interactive Factor Weight Simulator" in html_out
    assert "BAD.NS" in html_out
    assert "<script>alert(\"x\")</script>" not in html_out
