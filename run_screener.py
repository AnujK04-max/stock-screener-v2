"""run_screener.py - end-to-end runner for the NSE multi-factor & FinBERT screener."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from fetchers import (
    extract_as_of_date,
    fetch_fundamentals,
    fetch_headlines,
    fetch_price_panels,
    score_headlines_finbert,
)
from history import merge_history, read_history, snapshot_rows, write_history
from report import build_report_html
from scoring import MODEL_VERSION, WEIGHTS, score_universe
from tracker import HORIZONS, ROUND_TRIP_COST_PCT, build_tracker_summary
from universe import BENCHMARK, build_universe

DEFAULT_HF_REPO_ID = "Anujk04/indian-stock-screener"


def publish_to_hf(
    out_dir: Path,
    repo_id: str = DEFAULT_HF_REPO_ID,
    token: str | None = None,
) -> None:
    """Upload index.html and screener_history.csv to Hugging Face Space/Dataset repo."""
    tok = (token or os.environ.get("HF_TOKEN") or "").strip()
    if not tok:
        raise RuntimeError("HF_TOKEN is not set; pass --hf-token or set HF_TOKEN env var.")

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    html_path = out_dir / "index.html"
    hist_path = out_dir / "screener_history.csv"
    if not html_path.exists():
        raise FileNotFoundError(f"Missing generated report: {html_path}")

    api.upload_file(
        path_or_fileobj=str(html_path),
        path_in_repo="index.html",
        repo_id=repo_id,
        repo_type="space",
        commit_message=f"Update screener report ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
    )
    if hist_path.exists():
        api.upload_file(
            path_or_fileobj=str(hist_path),
            path_in_repo="screener_history.csv",
            repo_id=repo_id,
            repo_type="space",
            commit_message=f"Update screener history ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
        )


def run_pipeline(
    out_dir: str | Path = "build_out",
    history_path: str | Path | None = None,
    tickers: Iterable[str] | None = None,
    no_publish: bool = False,
    hf_repo_id: str = DEFAULT_HF_REPO_ID,
    hf_token: str | None = None,
    classifier=None,
) -> dict:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    hist_file = Path(history_path) if history_path is not None else (out_path / "screener_history.csv")

    universe_tickers = build_universe() if tickers is None else list(tickers)
    close_df, nifty = fetch_price_panels(universe_tickers, benchmark=BENCHMARK, period="1y")
    as_of_date = extract_as_of_date(nifty) or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    raw_df, skipped = fetch_fundamentals(universe_tickers, close_df=close_df)
    if raw_df is None or raw_df.empty:
        raise RuntimeError("No stock fundamentals returned; aborting run.")

    records = raw_df.to_dict("records")
    headlines_map = fetch_headlines(records)
    sent_map = score_headlines_finbert(headlines_map, classifier=classifier)

    counts = []
    scores = []
    details_json = []
    for rec in records:
        sym = str(rec.get("Ticker") or "").strip().upper()
        cnt, avg_sc, items = sent_map.get(sym, (0, None, []))
        counts.append(int(cnt))
        scores.append(np.nan if avg_sc is None else float(avg_sc))
        details_json.append(items)

    raw_df["Headline_Count"] = counts
    raw_df["FinBERT_Score"] = scores
    raw_df["News_Items"] = details_json

    scored = score_universe(raw_df, weights=WEIGHTS)

    existing_hist = read_history(str(hist_file))
    new_snap = snapshot_rows(scored, as_of=as_of_date, run_ts=run_ts, model_version=MODEL_VERSION)
    merged_hist = merge_history(existing_hist, new_snap)
    write_history(merged_hist, str(hist_file))

    tracker_summary = build_tracker_summary(
        history=merged_hist,
        close=close_df,
        nifty=nifty,
        model_version=MODEL_VERSION,
        cost_pct=ROUND_TRIP_COST_PCT,
        horizons=HORIZONS,
    )

    html = build_report_html(
        scored=scored,
        tracker=tracker_summary,
        as_of_date=as_of_date,
        updated_at=run_ts,
        model_version=MODEL_VERSION,
        skipped=skipped,
    )
    html_file = out_path / "index.html"
    html_file.write_text(html, encoding="utf-8")

    if not no_publish:
        publish_to_hf(out_path, repo_id=hf_repo_id, token=hf_token)

    return {
        "as_of_date": as_of_date,
        "scored": scored,
        "history": merged_hist,
        "tracker": tracker_summary,
        "html_path": str(html_file),
        "history_path": str(hist_file),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NSE Multi-Factor & FinBERT Screener")
    parser.add_argument("--out-dir", default="build_out", help="Output directory for index.html and CSV")
    parser.add_argument("--history-path", default=None, help="Optional path to existing screener_history.csv")
    parser.add_argument("--no-publish", action="store_true", help="Do not upload artifacts to Hugging Face")
    parser.add_argument("--hf-repo-id", default=DEFAULT_HF_REPO_ID, help="Target Hugging Face repo id")
    args = parser.parse_args(argv)

    run_pipeline(
        out_dir=args.out_dir,
        history_path=args.history_path,
        no_publish=args.no_publish,
        hf_repo_id=args.hf_repo_id,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
