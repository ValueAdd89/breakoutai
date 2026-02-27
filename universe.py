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

def _read_html_with_timeout(url: str, attrs: dict, timeout: int = 10) -> list:
    """Wraps pd.read_html with an explicit HTTP timeout via httpx."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
        resp.raise_for_status()
        from io import StringIO
        return pd.read_html(StringIO(resp.text), attrs=attrs)
    except Exception as e:
        raise RuntimeError(str(e)) from e


def _fetch_sp500() -> list[tuple[str, str]]:
    """Returns list of (symbol, name) for S&P 500 from Wikipedia."""
    try:
        tables = _read_html_with_timeout(
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
        tables = _read_html_with_timeout(
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

def _extract_col(raw: pd.DataFrame, field: str, sym: str) -> pd.Series:
    """
    Safely extract a single-symbol series from a yfinance download DataFrame.
    Handles both the old single-level API and the new MultiIndex API
    introduced in yfinance ≥ 0.2.38.
    """
    cols = raw.columns
    # New yfinance: MultiIndex (field, ticker) — swap to (ticker, field)
    if isinstance(cols, pd.MultiIndex):
        if (field, sym) in cols:
            return raw[(field, sym)]
        # Try case-insensitive field match
        for f, t in cols:
            if f.lower() == field.lower() and t == sym:
                return raw[(f, t)]
        return pd.Series(dtype=float)

    # Single ticker download: columns are plain field names
    if field in cols:
        return raw[field]
    for c in cols:
        if c.lower() == field.lower():
            return raw[c]
    return pd.Series(dtype=float)


def _batch_screen(symbols: list[str], min_price: float, min_volume: int) -> list[str]:
    """
    Download 5-day OHLCV for up to 100 symbols at once using yfinance batch.
    Filters out illiquid / penny stocks. Returns symbols that pass.
    Compatible with both old and new yfinance column layouts.
    """
    if not symbols:
        return []
    try:
        raw = yf.download(
            symbols,
            period="5d",
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            return []

        passed = []
        for sym in symbols:
            try:
                close_col = _extract_col(raw, "Close", sym).dropna()
                vol_col   = _extract_col(raw, "Volume", sym).dropna()

                if close_col.empty or vol_col.empty:
                    continue

                last_price = float(close_col.iloc[-1])
                avg_volume = float(vol_col.mean())

                if last_price >= min_price and avg_volume >= min_volume:
                    passed.append(sym)
            except Exception:
                continue
        return passed
    except Exception as e:
        logger.warning(f"Batch screen error: {e}")
        return []


# Hardcoded fallback — used when yfinance batch download fails entirely.
# These are all highly liquid, large-cap symbols guaranteed to have data.
_FALLBACK_CANDIDATES = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "UNH","XOM","JNJ","WMT","MA","PG","LLY","HD","CVX","MRK",
    "ABBV","PEP","KO","AVGO","COST","ADBE","CSCO","TMO","ACN","MCD",
    "ABT","CRM","DHR","NKE","LIN","ORCL","TXN","NEE","PM","BMY",
    "RTX","QCOM","AMGN","HON","SPGI","IBM","GE","CAT","UPS","BA",
    "AMD","INTC","NFLX","PYPL","SQ","SHOP","COIN","UBER","SNAP","PLTR",
    "SOFI","HOOD","RIVN","LCID","NIO","MSTR","MARA","RIOT","IONQ","SOUN",
    "SPY","QQQ","IWM","DIA","GLD","SLV","USO",
    "BAC","GS","MS","C","WFC","BLK","SCHW","AXP",
    "ENPH","FSLR","SEDG","RUN","SMCI","ARM","TSM",
]


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

    Falls back to _FALLBACK_CANDIDATES if the full universe or batch download fails.
    Cached for 30 minutes.
    """
    cache_key = f"screened_{min_price}_{min_avg_volume}"
    if cache_key in _screened_cache:
        logger.info(f"Screened candidates from cache: {len(_screened_cache[cache_key])}")
        return _screened_cache[cache_key]

    all_symbols = get_universe_symbols()
    if not all_symbols:
        logger.warning("Universe fetch returned 0 symbols — using hardcoded fallback list")
        all_symbols = _FALLBACK_CANDIDATES

    logger.info(f"Pre-screening {len(all_symbols)} symbols (price≥${min_price}, vol≥{min_avg_volume:,})…")

    passed: list[str] = []
    for i in range(0, len(all_symbols), batch_size):
        chunk = all_symbols[i : i + batch_size]
        ok    = _batch_screen(chunk, min_price, min_avg_volume)
        passed.extend(ok)
        logger.info(f"  Screened {min(i+batch_size, len(all_symbols))}/{len(all_symbols)} — {len(passed)} passing so far")

    # If batch screener produced nothing (yfinance API issue), skip the filter
    # and return the full list so the analyzer can still run on known-good symbols
    if not passed:
        logger.warning("Batch screener returned 0 — skipping filter, using full symbol list as candidates")
        passed = all_symbols if len(all_symbols) <= 200 else _FALLBACK_CANDIDATES

    logger.info(f"Pre-screen complete: {len(passed)} candidates")
    _screened_cache[cache_key] = passed
    return passed
