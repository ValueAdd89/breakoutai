"""
options_engine.py — Real-time options analytics (Skylit-style terminal).

Provides:
  - Full options chain fetching with Black-Scholes greeks (gamma, vanna)
  - GEX (Gamma Exposure) and VEX (Vanna Exposure) strike profiles
  - Key structural levels: flip zone, King Nodes, support/resistance
  - Unusual flow scanner across configurable symbol lists
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from cachetools import TTLCache

log = logging.getLogger(__name__)

_chain_cache: TTLCache = TTLCache(maxsize=64, ttl=600)   # 10 min
_lock = threading.Lock()

RISK_FREE_RATE = 0.053   # approximate 3-month T-bill yield

FLOW_SYMBOLS = [
    "SPY", "QQQ", "IWM", "AAPL", "MSFT", "TSLA", "NVDA", "AMZN",
    "META", "AMD", "GOOGL", "NFLX", "COIN", "PLTR", "MSTR",
    "SOFI", "HOOD", "RIVN", "GME", "MARA",
]


# ── Black-Scholes helpers ─────────────────────────────────────────────────────

def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _bs_gamma(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float = RISK_FREE_RATE,
) -> np.ndarray:
    """Black-Scholes gamma (same for calls and puts)."""
    T     = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    d1    = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return _norm_pdf(d1) / (S * sigma * np.sqrt(T))


def _bs_vanna(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    r: float = RISK_FREE_RATE,
) -> np.ndarray:
    """Vanna = ∂Delta/∂σ."""
    T     = np.maximum(T, 1e-8)
    sigma = np.maximum(sigma, 0.001)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return -_norm_pdf(d1) * d2 / sigma


# ── Options chain fetching ────────────────────────────────────────────────────

def fetch_options_chain(symbol: str) -> tuple[Optional[pd.DataFrame], float]:
    """
    Fetch the full options chain and compute Black-Scholes greeks.
    Returns (chain_df, spot_price). Cached for 10 minutes.
    chain_df columns include: optionType, expiry, T, dte, spot, strike,
      volume, openInterest, impliedVolatility, gamma, vanna, mid, vol_oi,
      dollar_premium, inTheMoney.
    """
    key = symbol.upper()
    with _lock:
        if key in _chain_cache:
            return _chain_cache[key]

    try:
        ticker = yf.Ticker(symbol)

        # Spot price — try fast_info first, fall back to history
        try:
            spot = float(ticker.fast_info.last_price or 0)
        except Exception:
            spot = 0.0
        if spot <= 0:
            hist = ticker.history(period="2d")
            spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
        if spot <= 0:
            return None, 0.0

        exps = ticker.options
        if not exps:
            return None, spot

        frames: list[pd.DataFrame] = []
        now = datetime.now()

        for expiry in exps[:10]:
            try:
                ch = ticker.option_chain(expiry)
                try:
                    exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
                except ValueError:
                    continue
                dte = max(0, (exp_dt - now).days)
                T   = max(dte / 365.0, 1.0 / 365)

                for opt_type, df in (("call", ch.calls), ("put", ch.puts)):
                    df = df.copy()
                    df["optionType"] = opt_type
                    df["expiry"]     = expiry
                    df["T"]          = T
                    df["dte"]        = dte
                    df["spot"]       = spot
                    frames.append(df)
            except Exception:
                continue

        if not frames:
            return None, spot

        full = pd.concat(frames, ignore_index=True)

        # Coerce numeric cols
        for col in ("volume", "openInterest", "impliedVolatility", "bid", "ask", "lastPrice"):
            full[col] = pd.to_numeric(full.get(col, 0), errors="coerce").fillna(0)

        full = full[
            (full["impliedVolatility"] > 0.005)
            & (full["strike"] > 0)
            & (full["T"] > 0)
        ].copy()

        if full.empty:
            return None, spot

        S     = full["spot"].to_numpy(float)
        K     = full["strike"].to_numpy(float)
        T_arr = full["T"].to_numpy(float)
        iv    = full["impliedVolatility"].to_numpy(float)

        full["gamma"] = _bs_gamma(S, K, T_arr, iv)
        full["vanna"] = _bs_vanna(S, K, T_arr, iv)

        full["mid"] = ((full["bid"] + full["ask"]) / 2).clip(lower=0)
        full.loc[full["mid"] <= 0, "mid"] = full["lastPrice"].clip(lower=0)

        full["vol_oi"] = np.where(
            full["openInterest"] > 0,
            full["volume"] / full["openInterest"],
            0.0,
        )
        full["dollar_premium"] = full["mid"] * full["volume"] * 100

        result = (full, spot)
        with _lock:
            _chain_cache[key] = result
        return result

    except Exception as exc:
        log.warning("fetch_options_chain(%s): %s", symbol, exc)
        return None, 0.0


# ── GEX / VEX profiles ────────────────────────────────────────────────────────

def compute_gex_profile(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Aggregate gamma exposure by strike across all expiries.
    GEX unit: $ millions of notional gamma per 1% spot move.
    Convention: call_gex positive (dealers long gamma), put_gex subtracted.
    Returns DataFrame[strike, call_gex, put_gex, net_gex].
    """
    if chain is None or chain.empty:
        return pd.DataFrame()

    factor = spot * spot * 100.0 / 1_000_000.0

    calls = chain[chain["optionType"] == "call"]
    puts  = chain[chain["optionType"] == "put"]

    def _agg(df: pd.DataFrame, name: str) -> pd.Series:
        return (
            df.groupby("strike")
            .apply(lambda x: (x["gamma"] * x["openInterest"] * factor).sum())
            .rename(name)
        )

    call_gex = _agg(calls, "call_gex")
    put_gex  = _agg(puts,  "put_gex")

    gex = (
        pd.DataFrame({"call_gex": call_gex, "put_gex": put_gex})
        .fillna(0)
        .reset_index()
        .sort_values("strike")
    )
    gex["net_gex"] = gex["call_gex"] - gex["put_gex"]
    return gex


def compute_vex_profile(chain: pd.DataFrame, spot: float) -> pd.DataFrame:
    """
    Vanna Exposure by strike.  Same sign convention as GEX.
    Returns DataFrame[strike, call_vex, put_vex, net_vex].
    """
    if chain is None or chain.empty:
        return pd.DataFrame()

    factor = spot * 100.0 / 1_000_000.0

    calls = chain[chain["optionType"] == "call"]
    puts  = chain[chain["optionType"] == "put"]

    def _agg(df: pd.DataFrame, name: str) -> pd.Series:
        return (
            df.groupby("strike")
            .apply(lambda x: (x["vanna"] * x["openInterest"] * factor).sum())
            .rename(name)
        )

    call_vex = _agg(calls, "call_vex")
    put_vex  = _agg(puts,  "put_vex")

    vex = (
        pd.DataFrame({"call_vex": call_vex, "put_vex": put_vex})
        .fillna(0)
        .reset_index()
        .sort_values("strike")
    )
    vex["net_vex"] = vex["call_vex"] - vex["put_vex"]
    return vex


def get_key_levels(gex: pd.DataFrame, spot: float) -> dict:
    """
    Identify structural levels from the GEX profile.
    Returns dict with:
      flip_level   – strike where net GEX crosses zero (nearest to spot)
      king_nodes   – top-3 strikes by |net_gex| within ±15% of spot
      resistance   – strongest positive-GEX strike above spot
      support      – strongest negative-GEX strike below spot
      gex_regime   – 'Long Gamma' | 'Short Gamma'
      total_gex    – sum of net_gex ($ millions)
    """
    if gex is None or gex.empty:
        return {}

    total_gex = float(gex["net_gex"].sum())

    # GEX flip — zero crossing nearest to spot
    net = gex["net_gex"].values
    crossings = np.where(np.diff(np.sign(net)))[0]
    flip_level: Optional[float] = None
    if len(crossings):
        candidates = gex.iloc[crossings]["strike"].values.astype(float)
        flip_level = float(candidates[np.argmin(np.abs(candidates - spot))])

    # King Nodes — within ±15% of spot by |net_gex|
    near = gex[(gex["strike"] >= spot * 0.85) & (gex["strike"] <= spot * 1.15)]
    king_nodes: list[float] = (
        near.reindex(near["net_gex"].abs().nlargest(3).index)["strike"]
        .astype(float)
        .tolist()
    )

    # Resistance = max positive GEX strike above spot
    above = gex[gex["strike"] > spot * 1.003]
    resistance: Optional[float] = None
    if not above.empty and above["net_gex"].max() > 0:
        resistance = float(above.loc[above["net_gex"].idxmax(), "strike"])

    # Support = most negative GEX strike below spot
    below = gex[gex["strike"] < spot * 0.997]
    support: Optional[float] = None
    if not below.empty and below["net_gex"].min() < 0:
        support = float(below.loc[below["net_gex"].idxmin(), "strike"])

    return {
        "flip_level": flip_level,
        "king_nodes": king_nodes,
        "resistance": resistance,
        "support":    support,
        "gex_regime": "Long Gamma" if total_gex > 0 else "Short Gamma",
        "total_gex":  total_gex,
    }


# ── Unusual flow scanner ──────────────────────────────────────────────────────

def _flow_for_symbol(
    symbol: str, min_premium: float, min_vol: int
) -> list[dict]:
    """Return unusual-flow rows for a single symbol."""
    chain, spot = fetch_options_chain(symbol)
    if chain is None or spot <= 0:
        return []

    unusual = chain[
        (chain["volume"] >= min_vol)
        & (chain["dollar_premium"] >= min_premium)
    ]

    rows: list[dict] = []
    for _, row in unusual.iterrows():
        is_call  = row["optionType"] == "call"
        strike   = float(row["strike"])
        otm_pct  = (
            (strike - spot) / spot * 100
            if is_call
            else (spot - strike) / spot * 100
        )
        vol   = int(row["volume"])
        oi    = int(row["openInterest"])
        iv    = round(float(row["impliedVolatility"]) * 100, 1)
        prem  = round(float(row["mid"]), 2)
        dp    = int(row["dollar_premium"])
        vo    = round(float(row["vol_oi"]), 2)
        dte   = int(row["dte"])

        rows.append({
            "symbol":        symbol,
            "expiry":        str(row["expiry"]),
            "dte":           dte,
            "strike":        strike,
            "type":          "CALL" if is_call else "PUT",
            "volume":        vol,
            "oi":            oi,
            "iv":            iv,
            "premium":       prem,
            "dollar_premium": dp,
            "vol_oi":        vo,
            "otm_pct":       round(otm_pct, 1),
            "itm":           bool(row.get("inTheMoney", False)),
            "sentiment":     "bullish" if is_call else "bearish",
            "unusual":       (vo >= 0.5) or (vol >= 1_000) or (dp >= 100_000),
            "spot":          round(spot, 2),
        })
    return rows


def scan_unusual_flow(
    symbols: Optional[list[str]] = None,
    min_premium: float = 25_000,
    min_vol: int = 100,
    max_workers: int = 8,
) -> pd.DataFrame:
    """
    Parallel scan for unusual options activity.
    Returns DataFrame sorted by dollar_premium descending.
    """
    syms = symbols or FLOW_SYMBOLS
    all_rows: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_flow_for_symbol, s, min_premium, min_vol): s
            for s in syms
        }
        for fut in concurrent.futures.as_completed(futures):
            try:
                all_rows.extend(fut.result())
            except Exception as exc:
                log.warning("flow scan %s: %s", futures[fut], exc)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.sort_values("dollar_premium", ascending=False).reset_index(drop=True)
    return df


def get_flow_summary(symbol: str) -> dict:
    """Quick call/put bias summary for a single symbol."""
    chain, spot = fetch_options_chain(symbol)
    if chain is None or spot <= 0:
        return {}

    traded = chain[chain["volume"] > 0]
    calls  = traded[traded["optionType"] == "call"]
    puts   = traded[traded["optionType"] == "put"]

    call_prem = float((calls["dollar_premium"]).sum())
    put_prem  = float((puts["dollar_premium"]).sum())
    total     = call_prem + put_prem

    return {
        "spot":        spot,
        "call_premium": call_prem,
        "put_premium":  put_prem,
        "total_premium": total,
        "call_pct":    call_prem / total * 100 if total > 0 else 50,
        "put_pct":     put_prem  / total * 100 if total > 0 else 50,
        "bias":        (
            "bullish" if call_prem > put_prem * 1.2
            else "bearish" if put_prem > call_prem * 1.2
            else "neutral"
        ),
    }
