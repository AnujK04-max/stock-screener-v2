"""tests/test_history_tracker.py - unit tests for history.py and tracker.py."""
from __future__ import annotations

import pandas as pd

from history import (
    LEGACY_VERSION,
    merge_history,
    normalize_history,
    read_history,
    snapshot_rows,
    write_history,
)
from scoring import MODEL_VERSION, score_universe
from tests.synthetic import make_history_frame, make_price_panel, make_raw_universe
from tracker import HORIZONS, ROUND_TRIP_COST_PCT, build_tracker_summary, select_non_overlapping


def test_legacy_history_normalization_and_merge(tmp_path):
    legacy_df = pd.DataFrame(
        [
            {
                "Run_Date": "2026-10-01",
                "Timestamp": "2026-10-01 10:00:00",
                "Ticker": "INFY",
                "Composite_Score": 71.2,
                "Screen_Flag": "Passes Multi-Factor Screen",
            }
        ]
    )
    norm_legacy = normalize_history(legacy_df)
    assert (norm_legacy["Model_Version"] == LEGACY_VERSION).all()
    assert norm_legacy.loc[0, "As_Of_Date"] == "2026-10-01"

    scored = score_universe(make_raw_universe())
    snap = snapshot_rows(scored, as_of="2026-10-02", run_ts="2026-10-02 16:00:00", model_version=MODEL_VERSION)
    merged = merge_history(norm_legacy, snap)
    assert len(merged) == len(norm_legacy) + len(snap)

    out_csv = tmp_path / "screener_history.csv"
    write_history(merged, str(out_csv))
    reloaded = read_history(str(out_csv))
    assert len(reloaded) == len(merged)


def test_select_non_overlapping_cohorts():
    cohorts = [{"i0": d} for d in range(0, 30, 3)]
    kept = select_non_overlapping(cohorts, horizon=10)
    assert [c["i0"] for c in kept] == [0, 12, 24]


def test_tracker_summary_preliminary_and_evaluable():
    close_df, nifty = make_price_panel(n_days=140, seed=7)
    trading_dates = [d.strftime("%Y-%m-%d") for d in close_df.index[:95]]
    hist = make_history_frame(trading_dates, model_version=MODEL_VERSION)

    summary = build_tracker_summary(
        history=hist,
        close=close_df,
        nifty=nifty,
        model_version=MODEL_VERSION,
        cost_pct=ROUND_TRIP_COST_PCT,
        horizons=HORIZONS,
    )
    assert summary["model_version"] == MODEL_VERSION
    assert summary["cohorts_logged"] == len(trading_dates)
    assert len(summary["horizons"]) == len(HORIZONS)

    h5 = summary["horizons"][0]
    assert h5["status"] == "evaluable"
    assert h5["n_independent"] >= 8
    assert h5["top_net"] is not None
    assert h5["ew"] is not None
    assert h5["nifty"] is not None
