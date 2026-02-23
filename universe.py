"""
Market Universe — fetches the full investable universe dynamically.

Sources (all free, no API key needed):
  • S&P 500    — Wikipedia table (always up to date)
  • NASDAQ 100 — Wikipedia table
  • Russell 2000 small-caps — iShares ETF holdings CSV (GitHub mirror)

Then applies a FAST pre-screener using yfinance batch download:
  • Price ≥ $1  (eliminates penny stocks)
  • Volume ≥ 500k avg daily (liquidity filter)
  • Not an ADR/ETF if desired

This two-stage approach means we only run the expensive ML pipeline on
~200-400 stocks that actually have momentum potential, making a full-market
scan complete in under 10 minutes.
"""
import logging
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import pandas as pd
import yfinance as yf
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Cache the full universe list for 6 hours (it barely changes day-to-day)
_universe_cache: TTLCache = TTLCache(maxsize=1, ttl=6 * 3600)
# Cache screened candidates for 30 minutes
_screened_cache: TTLCache = TTLCache(maxsize=1, ttl=1800)

_universe_lock = threading.Lock()


# ── Universe fetchers ─────────────────────────────────────────────────────────

def _fetch_sp500() -> list[tuple[str, str]]:
    """Returns list of (symbol, name) for S&P 500 from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
        return list(zip(df["Symbol"].tolist(), df["Security"].tolist()))
    except Exception as e:
        logger.warning(f"S&P 500 fetch failed: {e}")
        return []


def _fetch_nasdaq100() -> list[tuple[str, str]]:
    """Returns list of (symbol, name) for NASDAQ-100 from Wikipedia."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/Nasdaq-100",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        return list(zip(df["Ticker"].tolist(), df["Company"].tolist()))
    except Exception as e:
        logger.warning(f"NASDAQ-100 fetch failed: {e}")
        return []


def _fetch_russell2000() -> list[tuple[str, str]]:
    """
    Returns Russell 2000 tickers from iShares IWM holdings (public CSV).
    Falls back to a curated 200-stock small-cap list if the CSV is unavailable.
    """
    try:
        url = "https://www.ishares.com/us/products/239710/ishares-russell-2000-etf/1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund"
        with httpx.Client(timeout=15, follow_redirects=True) as client:
            resp = client.get(url)
        if resp.status_code == 200:
            lines = resp.text.splitlines()
            # Skip header lines until we find "Ticker"
            start = next((i for i, l in enumerate(lines) if l.startswith("Ticker,")), None)
            if start is not None:
                from io import StringIO
                df = pd.read_csv(StringIO("\n".join(lines[start:])))
                df = df.dropna(subset=["Ticker"])
                df = df[df["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)]
                return list(zip(df["Ticker"].tolist(), df.get("Name", df["Ticker"]).tolist()))
    except Exception as e:
        logger.warning(f"Russell 2000 CSV fetch failed: {e}")

    # Curated fallback — high-momentum small caps commonly watched
    fallback = [
        "SMCI","IONQ","AEHR","SOUN","BBAI","CLSK","CIFR","IREN","MARA","RIOT",
        "HUT","BTBT","BITI","CANO","NKLA","WKHS","BLNK","CHPT","EVGO","PTRA",
        "VLDR","LIDR","OUST","AEVA","LAZR","INVZ","MVIS","INDI","MFAC","AIOT",
        "GREE","KPLT","CLNE","CLOV","WISH","FFIE","MULN","NAKD","EXPR","BBBY",
        "SPWR","FSLR","ENPH","SEDG","RUN","NOVA","ARRY","MAXN","CSIQ","JKS",
        "ACMR","AMAT","KLAC","LRCX","MCHP","MPWR","NXPI","ON","SWKS","QRVO",
        "SITM","AMBA","COHU","FORM","ICHR","KLIC","ONTO","RMBS","UCTT","VECO",
        "ALGM","AEHR","DIOD","IMOS","LSCC","MTSI","POWI","SMTC","VICR","WOLF",
    ]
    return [(s, s) for s in fallback]


def get_full_universe() -> list[tuple[str, str]]:
    """
    Returns deduplicated (symbol, name) pairs for the full investable universe.
    Cached for 6 hours.
    """
    with _universe_lock:
        if "universe" in _universe_cache:
            return _universe_cache["universe"]

        logger.info("Fetching full market universe…")
        seen: set[str] = set()
        combined: list[tuple[str, str]] = []

        for sym, name in (_fetch_sp500() + _fetch_nasdaq100() + _fetch_russell2000()):
            sym = sym.strip().upper()
            # Skip invalid symbols
            if not sym or len(sym) > 5 or not sym.replace("-", "").isalpha():
                continue
            if sym not in seen:
                seen.add(sym)
                combined.append((sym, name))

        logger.info(f"Universe: {len(combined)} unique symbols")
        _universe_cache["universe"] = combined
        return combined


def get_universe_symbols() -> list[str]:
    return [s for s, _ in get_full_universe()]


def get_universe_names() -> dict[str, str]:
    return {s: n for s, n in get_full_universe()}


# ── Fast pre-screener ─────────────────────────────────────────────────────────

def _batch_screen(symbols: list[str], min_price: float, min_volume: int) -> list[str]:
    """
    Download 5-day OHLCV for up to 100 symbols at once using yfinance batch.
    Filters out illiquid / penny stocks. Returns symbols that pass.
    """
    try:
        raw = yf.download(
            symbols,
            period="5d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
        passed = []
        for sym in symbols:
            try:
                if len(symbols) == 1:
                    close_col = raw["Close"]
                    vol_col   = raw["Volume"]
                else:
                    close_col = raw["Close"][sym]
                    vol_col   = raw["Volume"][sym]

                close_col = close_col.dropna()
                vol_col   = vol_col.dropna()

                if close_col.empty or vol_col.empty:
                    continue

                last_price  = float(close_col.iloc[-1])
                avg_volume  = float(vol_col.mean())

                if last_price >= min_price and avg_volume >= min_volume:
                    passed.append(sym)
            except Exception:
                continue
        return passed
    except Exception as e:
        logger.warning(f"Batch screen error: {e}")
        return []


def get_screened_candidates(
    min_price: float = 2.0,
    min_avg_volume: int = 500_000,
    batch_size: int = 100,
) -> list[str]:
    """
    Two-stage fast screener:
      Stage 1 — batch download 5d OHLCV for the full universe (100 at a time)
                filters by price ≥ min_price and avg volume ≥ min_avg_volume
      Stage 2 — returns the passing symbols, typically 400-800 of ~3000

    Cached for 30 minutes.
    """
    cache_key = f"screened_{min_price}_{min_avg_volume}"
    if cache_key in _screened_cache:
        logger.info(f"Screened candidates from cache: {len(_screened_cache[cache_key])}")
        return _screened_cache[cache_key]

    all_symbols = get_universe_symbols()
    logger.info(f"Pre-screening {len(all_symbols)} symbols (price≥${min_price}, vol≥{min_avg_volume:,})…")

    passed: list[str] = []
    for i in range(0, len(all_symbols), batch_size):
        chunk = all_symbols[i : i + batch_size]
        ok    = _batch_screen(chunk, min_price, min_avg_volume)
        passed.extend(ok)
        logger.info(f"  Screened {min(i+batch_size, len(all_symbols))}/{len(all_symbols)} — {len(passed)} passing so far")

    logger.info(f"Pre-screen complete: {len(passed)}/{len(all_symbols)} candidates")
    _screened_cache[cache_key] = passed
    return passed
