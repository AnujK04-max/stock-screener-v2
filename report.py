"""report.py - static HTML report generator for the NSE multi-factor & FinBERT screener."""
from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd

from scoring import MODEL_VERSION, WEIGHTS


def _safe_float(val: Any) -> float | None:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _fmt_num(val: Any, decimals: int = 2, prefix: str = "", suffix: str = "") -> str:
    v = _safe_float(val)
    if v is None:
        return "N/A"
    return f"{prefix}{v:,.{decimals}f}{suffix}"


def _quantile_sorted(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(float(x) for x in values if x is not None and np.isfinite(x))
    if not s:
        return None
    pos = max(0.0, min(1.0, float(q))) * (len(s) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def build_report_html(
    scored: pd.DataFrame,
    tracker: dict[str, Any] | None = None,
    as_of_date: str = "",
    updated_at: str = "",
    model_version: str = MODEL_VERSION,
    skipped: list[dict[str, Any]] | None = None,
) -> str:
    """Render a self-contained HTML dashboard from scored rows and tracker summary."""
    df = scored.copy() if isinstance(scored, pd.DataFrame) else pd.DataFrame()
    skipped_rows = list(skipped or [])
    tracker_obj = tracker or {"horizons": [], "cohorts_logged": 0, "legacy_cohorts": 0}

    records: list[dict[str, Any]] = []
    if not df.empty:
        for _, row in df.iterrows():
            news_raw = row.get("News_Items")
            if isinstance(news_raw, str):
                try:
                    news_items = json.loads(news_raw)
                except Exception:
                    news_items = []
            elif isinstance(news_raw, list):
                news_items = news_raw
            else:
                news_items = []

            records.append(
                {
                    "rank": int(_safe_float(row.get("Rank")) or 0),
                    "ticker": str(row.get("Ticker") or ""),
                    "name": str(row.get("Name") or row.get("Ticker") or ""),
                    "sector": str(row.get("Sector") or "Unknown"),
                    "industry": str(row.get("Industry") or "Unknown"),
                    "price": _safe_float(row.get("Price")),
                    "change_pct": _safe_float(row.get("Change_%")),
                    "pe": _safe_float(row.get("PE")),
                    "pb": _safe_float(row.get("PB")),
                    "roe": _safe_float(row.get("ROE_%")),
                    "rev_cagr_3y": _safe_float(row.get("Rev_CAGR_3Y_%")),
                    "eps_cagr_3y": _safe_float(row.get("EPS_CAGR_3Y_%")),
                    "val_rank": _safe_float(row.get("Val_Rank")) or 50.0,
                    "qual_rank": _safe_float(row.get("Qual_Rank")) or 50.0,
                    "growth_rank": _safe_float(row.get("Growth_Rank")) or 50.0,
                    "mom_sent_rank": _safe_float(row.get("Mom_Sent_Rank")) or 50.0,
                    "sent_score": _safe_float(row.get("Sent_Score")) or 50.0,
                    "finbert_score": _safe_float(row.get("FinBERT_Score")),
                    "headline_count": int(_safe_float(row.get("Headline_Count")) or 0),
                    "composite_score": _safe_float(row.get("Composite_Score")),
                    "screen_flag": str(row.get("Screen_Flag") or ""),
                    "val_tag": str(row.get("Val_Tag") or ""),
                    "base_effect_flag": str(row.get("Base_Effect_Flag") or "Normal"),
                    "base_effect_note": str(row.get("Base_Effect_Note") or ""),
                    "not_rankable_reason": str(row.get("Not_Rankable_Reason") or ""),
                    "rankable": bool(row.get("Rankable", True)),
                    "news": news_items,
                }
            )

    total_count = len(records)
    top_count = sum(1 for r in records if r["screen_flag"] == "Passes Multi-Factor Screen")
    nlp_count = sum(1 for r in records if r["headline_count"] >= 3 and r["finbert_score"] is not None)
    nm_count = sum(1 for r in records if not r["rankable"])

    sample_vals = [r["val_rank"] for r in records if r["rankable"]]
    q80_sample = _quantile_sorted(sample_vals, 0.80)
    q80_label = f"{q80_sample:.1f}" if q80_sample is not None else "N/A"

    table_rows_html: list[str] = []
    for idx, r in enumerate(records):
        comp_txt = f"{r['composite_score']:.1f}" if r["composite_score"] is not None else "NM"
        pe_txt = _fmt_num(r["pe"], 2)
        pb_txt = _fmt_num(r["pb"], 2)
        roe_txt = _fmt_num(r["roe"], 1, suffix="%")
        if r["headline_count"] >= 3 and r["finbert_score"] is not None:
            fb_txt = f"{r['finbert_score']:+.2f} ({r['headline_count']} news)"
        else:
            fb_txt = f"Insufficient (&lt;{3})"

        news_li = "".join(
            f"<li>{html.escape(str(item.get('t') or ''))}"
            + (f" <span>({float(item['s']):+.2f})</span>" if item.get("s") is not None else "")
            + "</li>"
            for item in (r["news"] or [])
        ) or "<li>No recent headlines matched.</li>"

        detail_note = r["not_rankable_reason"] or r["base_effect_note"] or "Standard multi-factor evaluation."
        row_html = (
            f"<tr class='stock-row' data-idx='{idx}' data-flag='{html.escape(r['screen_flag'])}' "
            f"data-search='{html.escape((r['ticker'] + ' ' + r['name'] + ' ' + r['sector']).lower())}'>"
            f"<td>#{r['rank'] or (idx + 1)} <strong>{html.escape(r['ticker'])}</strong> "
            f"<span class='sub'>{html.escape(r['sector'])}</span></td>"
            f"<td>{_fmt_num(r['price'], 2, prefix='₹')} "
            f"<span class='chg'>{_fmt_num(r['change_pct'], 2, suffix='%')}</span></td>"
            f"<td><strong>{html.escape(comp_txt)}</strong></td>"
            f"<td>PE: {html.escape(pe_txt)}<br>PB: {html.escape(pb_txt)}</td>"
            f"<td>{html.escape(roe_txt)}</td>"
            f"<td>{html.escape(fb_txt)}</td>"
            f"<td>{html.escape(r['screen_flag'])}</td>"
            f"</tr>"
            f"<tr class='detail-row' id='detail-{idx}' style='display:none;'>"
            f"<td colspan='7'>"
            f"<div class='detail-box'>"
            f"<div><strong>Factor Ranks:</strong> Val {r['val_rank']:.1f} | Qual {r['qual_rank']:.1f} | "
            f"Growth {r['growth_rank']:.1f} | Mom+Sent {r['mom_sent_rank']:.1f}</div>"
            f"<div><strong>3Y CAGR:</strong> Rev {_fmt_num(r['rev_cagr_3y'], 1, suffix='%')} | "
            f"EPS {_fmt_num(r['eps_cagr_3y'], 1, suffix='%')} | <strong>Base Effect:</strong> {html.escape(r['base_effect_flag'])}</div>"
            f"<div><strong>Notes:</strong> {html.escape(detail_note)}</div>"
            f"<ul class='news-list'>{news_li}</ul>"
            f"</div></td></tr>"
        )
        table_rows_html.append(row_html)

    skipped_html = (
        "<ul>"
        + "".join(
            f"<li>{html.escape(str(s.get('ticker') or ''))}: {html.escape(str(s.get('reason') or ''))}</li>"
            for s in skipped_rows
        )
        + "</ul>"
        if skipped_rows
        else "<p>All requested tickers passed initial price validation.</p>"
    )

    payload_json = json.dumps(
        {
            "records": records,
            "tracker": tracker_obj,
            "weights": WEIGHTS,
            "as_of_date": as_of_date,
            "updated_at": updated_at,
            "model_version": model_version,
        },
        ensure_ascii=False,
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>NSE Multi-Factor &amp; FinBERT Quantitative Screener</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #0f172a; color: #e2e8f0; }}
    .meta {{ color: #94a3b8; font-size: 0.9rem; margin-bottom: 12px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin: 16px 0; }}
    .card {{ background: #1e293b; padding: 14px; border-radius: 8px; border: 1px solid #334155; }}
    .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }}
    input, select, button {{ padding: 8px 10px; border-radius: 6px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; background: #1e293b; }}
    th, td {{ padding: 10px; border-bottom: 1px solid #334155; text-align: left; vertical-align: top; }}
    .stock-row {{ cursor: pointer; }}
    .stock-row:hover {{ background: #273549; }}
    .sub, .chg {{ color: #94a3b8; font-size: 0.85rem; }}
    .tabs {{ display: flex; gap: 8px; margin-top: 10px; }}
    .tab-btn.active {{ background: #2563eb; border-color: #3b82f6; }}
  </style>
</head>
<body>
  <h1>NSE Multi-Factor &amp; FinBERT Screener</h1>
  <div class="meta">
    Prices: {html.escape(as_of_date or "N/A")} (as of last close) &bull;
    Updated: {html.escape(updated_at or "N/A")} &bull;
    model {html.escape(model_version)} &bull;
    Sample Q80: {html.escape(q80_label)}
  </div>

  <div class="kpis">
    <div class="card"><div>Screened</div><strong>{total_count}</strong></div>
    <div class="card"><div>Top Quintile</div><strong>{top_count}</strong></div>
    <div class="card"><div>FinBERT Coverage (&ge;3)</div><strong>{nlp_count}</strong></div>
    <div class="card"><div>Non-Rankable / NM</div><strong>{nm_count}</strong></div>
  </div>

  <section class="card" id="tracker-panel">
    <h2>Forward Performance Tracker</h2>
    <div class="tabs" id="horizon-tabs">
      <button type="button" class="tab-btn active" data-days="5">1 week</button>
      <button type="button" class="tab-btn" data-days="21">1 month</button>
      <button type="button" class="tab-btn" data-days="63">3 months</button>
    </div>
    <div id="tracker-body" class="meta" style="margin-top:10px;">Loading tracker summary...</div>
  </section>

  <section class="card" style="margin-top:16px;">
    <h2>Interactive Factor Weight Simulator</h2>
    <div class="controls">
      <label>Value <input id="w-value" type="range" min="0" max="100" value="25" /></label>
      <label>Quality <input id="w-quality" type="range" min="0" max="100" value="30" /></label>
      <label>Growth <input id="w-growth" type="range" min="0" max="100" value="20" /></label>
      <label>Mom+Sent <input id="w-mom" type="range" min="0" max="100" value="25" /></label>
      <button type="button" id="reset-weights">Reset Weights</button>
    </div>
  </section>

  <div class="controls">
    <input id="search-input" type="search" placeholder="Search ticker, name, or sector..." />
    <select id="flag-filter">
      <option value="">All Screen Flags</option>
      <option value="Passes Multi-Factor Screen">Passes Multi-Factor Screen</option>
      <option value="Neutral / Watchlist">Neutral / Watchlist</option>
      <option value="Fails Screen / Lagging">Fails Screen / Lagging</option>
    </select>
  </div>

  <table id="screener-table">
    <thead>
      <tr>
        <th>Rank &amp; Ticker</th>
        <th>Price</th>
        <th>Composite</th>
        <th>Multiples</th>
        <th>ROE %</th>
        <th>FinBERT (7D)</th>
        <th>Screen Flag</th>
      </tr>
    </thead>
    <tbody>
      {"".join(table_rows_html)}
    </tbody>
  </table>

  <section class="card" style="margin-top:20px;">
    <h2>Data Quality &amp; Skipped Tickers</h2>
    {skipped_html}
  </section>

  <script id="screener-data" type="application/json">{payload_json}</script>
  <script>
    (function () {{
      const payload = JSON.parse(document.getElementById("screener-data").textContent || "{{}}");
      const horizons = (payload.tracker && payload.tracker.horizons) || [];
      const trackerBody = document.getElementById("tracker-body");

      function renderHorizon(days) {{
        const h = horizons.find((item) => Number(item.days) === Number(days)) || horizons[0];
        if (!h) {{
          trackerBody.textContent = "Collecting forward cohorts...";
          return;
        }}
        trackerBody.textContent =
          h.label + " (" + h.days + "d): status=" + (h.status || "collecting") +
          ", independent cohorts=" + (h.n_independent ?? 0) +
          ", net top=" + (h.top_net ?? "N/A") +
          ", EW=" + (h.ew ?? "N/A") +
          ", Nifty=" + (h.nifty ?? "N/A");
      }}

      document.querySelectorAll(".tab-btn").forEach((btn) => {{
        btn.addEventListener("click", () => {{
          document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
          btn.classList.add("active");
          renderHorizon(btn.getAttribute("data-days"));
        }});
      }});
      renderHorizon(5);

      document.querySelectorAll(".stock-row").forEach((row) => {{
        row.addEventListener("click", () => {{
          const idx = row.getAttribute("data-idx");
          const detail = document.getElementById("detail-" + idx);
          if (detail) {{
            detail.style.display = detail.style.display === "none" ? "table-row" : "none";
          }}
        }});
      }});

      const searchEl = document.getElementById("search-input");
      const flagEl = document.getElementById("flag-filter");
      function applyFilters() {{
        const q = (searchEl.value || "").trim().toLowerCase();
        const fl = flagEl.value || "";
        document.querySelectorAll(".stock-row").forEach((row) => {{
          const hay = row.getAttribute("data-search") || "";
          const rf = row.getAttribute("data-flag") || "";
          const match = (!q || hay.includes(q)) && (!fl || rf === fl);
          row.style.display = match ? "" : "none";
          const idx = row.getAttribute("data-idx");
          const detail = document.getElementById("detail-" + idx);
          if (!match && detail) detail.style.display = "none";
        }});
      }}
      if (searchEl) searchEl.addEventListener("input", applyFilters);
      if (flagEl) flagEl.addEventListener("change", applyFilters);

      const resetBtn = document.getElementById("reset-weights");
      if (resetBtn) {{
        resetBtn.addEventListener("click", () => {{
          document.getElementById("w-value").value = 25;
          document.getElementById("w-quality").value = 30;
          document.getElementById("w-growth").value = 20;
          document.getElementById("w-mom").value = 25;
        }});
      }}
    }})();
  </script>
</body>
</html>
"""
