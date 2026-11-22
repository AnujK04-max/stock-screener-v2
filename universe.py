"""universe.py - the list of stocks the screener covers.

The original script built the list with ``list(set([...]))``, which gives a different order
on every run and hid two dead symbols (they never appeared in your published output).
This keeps your order, removes duplicates, and records why anything was dropped.
"""
from __future__ import annotations

BENCHMARK = "^NSEI"

_UNIVERSE = [
    "COCHINSHIP.NS", "INFY.NS", "ITC.NS", "LGEINDIA.NS", "RELIANCE.NS", "SHRIRAMFIN.NS",
    "IDEA.NS", "LT.NS", "BEL.NS", "HDFCBANK.NS", "ETERNAL.NS", "HAL.NS", "DLF.NS",
    "DRREDDY.NS", "TMPV.NS", "TCS.NS", "ICICIBANK.NS", "BHARTIARTL.NS", "SBIN.NS",
    "LICI.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "M&M.NS", "KOTAKBANK.NS",
    "AXISBANK.NS", "MARUTI.NS", "ULTRACEMCO.NS", "NTPC.NS", "TITAN.NS", "ADANIENT.NS",
    "ONGC.NS", "POWERGRID.NS", "ADANIPORTS.NS", "COALINDIA.NS", "BAJAJFINSV.NS",
    "ASIANPAINT.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "TRENT.NS", "SIEMENS.NS", "ABB.NS",
    "PIDILITIND.NS", "GRASIM.NS", "INDIGO.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TECHM.NS",
    "WIPRO.NS", "HINDALCO.NS", "CIPLA.NS", "EICHERMOT.NS", "DIVISLAB.NS", "BPCL.NS",
    "BRITANNIA.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS",
    "CHOLAFIN.NS", "BANKBARODA.NS", "PNB.NS", "IOB.NS", "CANBK.NS", "UNIONBANK.NS",
    "RECLTD.NS", "PFC.NS", "IRFC.NS", "VBL.NS", "GODREJCP.NS", "DABUR.NS", "MARICO.NS",
    "COLPAL.NS", "HAVELLS.NS", "AMBUJACEM.NS", "SHREECEM.NS", "GAIL.NS", "IOC.NS",
    "VEDL.NS", "JINDALSTEL.NS", "TVSMOTOR.NS", "MOTHERSON.NS", "BOSCHLTD.NS", "ICICIGI.NS",
    "ICICIPRULI.NS", "SBICARD.NS", "NAUKRI.NS", "UNITDSPR.NS", "TORNTPHARM.NS",
    "ZYDUSLIFE.NS", "LUPIN.NS", "AUROPHARMA.NS", "MANKIND.NS", "CGPOWER.NS", "POLYCAB.NS",
    "CUMMINSIND.NS", "ASTRAL.NS", "INDHOTEL.NS", "LODHA.NS", "GODREJPROP.NS",
    "OBEROIRLTY.NS",
]

# Symbols that were in the original list but are removed on purpose. Neither appeared in the
# published output, so they were being skipped silently. Delete an entry here to restore it.
DROPPED_SYMBOLS = {
    "ZOMATO.NS": "Renamed Eternal; ETERNAL.NS is already in the list.",
    "TATAMOTORS.NS": "Legacy symbol after the Tata Motors demerger; TMPV.NS is in the list.",
}


def build_universe(extra=()) -> list:
    """Ordered, de-duplicated tickers (original order preserved, extras appended)."""
    seen = {}
    for t in list(_UNIVERSE) + list(extra):
        t = t.strip().upper()
        if t and t not in DROPPED_SYMBOLS and t not in seen:
            seen[t] = None
    return list(seen)


def to_symbol(ticker: str) -> str:
    return ticker[:-3] if ticker.endswith(".NS") else ticker
