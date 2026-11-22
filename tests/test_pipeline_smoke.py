"""tests/test_pipeline_smoke.py - end-to-end smoke test for run_screener.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import run_screener
from tests.synthetic import make_price_panel, make_raw_universe


def test_run_pipeline_no_publish_and_history_merge(tmp_path, monkeypatch):
    close_df, nifty = make_price_panel(n_days=90, seed=11)
    raw_df = make_raw_universe().drop(columns=["Headline_Count", "FinBERT_Score"])

    monkeypatch.setattr(run_screener, "fetch_price_panels", lambda *a, **kw: (close_df, nifty))
    monkeypatch.setattr(run_screener, "fetch_fundamentals", lambda *a, **kw: (raw_df.copy(), []))
    monkeypatch.setattr(
        run_screener,
        "fetch_headlines",
        lambda records, *a, **kw: {r["Ticker"]: ["Strong earnings update"] for r in records},
    )

    def fake_clf(texts):
        return [{"label": "positive", "score": 0.85} for _ in texts]

    out_dir = tmp_path / "build_out"
    res1 = run_screener.run_pipeline(
        out_dir=out_dir,
        tickers=["HDFCBANK.NS", "INFY.NS", "HIGHPE.NS", "LOSSCO.NS", "NEGEQ.NS"],
        no_publish=True,
        classifier=fake_clf,
    )

    assert (out_dir / "index.html").exists()
    assert (out_dir / "screener_history.csv").exists()
    assert len(res1["history"]) == len(raw_df)

    # Second run with shifted benchmark date appends new snapshot rows
    nifty_shifted = nifty.copy()
    nifty_shifted.index = nifty_shifted.index + pd.Timedelta(days=1)
    monkeypatch.setattr(run_screener, "fetch_price_panels", lambda *a, **kw: (close_df, nifty_shifted))

    res2 = run_screener.run_pipeline(
        out_dir=out_dir,
        tickers=["HDFCBANK.NS", "INFY.NS", "HIGHPE.NS", "LOSSCO.NS", "NEGEQ.NS"],
        no_publish=True,
        classifier=fake_clf,
    )
    assert len(res2["history"]) == len(raw_df) * 2


def test_publish_to_hf_missing_token_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    html_file = tmp_path / "index.html"
    html_file.write_text("<html></html>", encoding="utf-8")

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run_screener.publish_to_hf(tmp_path, token=None)
