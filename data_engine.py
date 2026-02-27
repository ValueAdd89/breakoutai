"""
Data ingestion layer — synchronous, cache-backed.
Works on Streamlit Cloud (no asyncio required).
"""
import logging
import math
import calendar
from datetime import datetime, timezone, date, timedelta
from typing import Optional

import httpx
import feedparser
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from cachetools import TTLCache

import config

logger = logging.getLogger(__name__)


def _market_open() -> bool:
    """True during NYSE core hours Mon–Fri 09:30–16:00 ET (14:30–21:00 UTC)."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:          # Saturday / Sunday
        return False
    return 14 <= now.hour < 21      # rough UTC window


def _price_ttl() -> int:
    """5 min during market hours; 4 hours when closed — no point refetching stale data."""
    return 300 if _market_open() else 14_400


def _news_ttl() -> int:
    """10 min during market hours; 1 hour when closed."""
    return 600 if _market_open() else 3_600


_price_cache: TTLCache = TTLCache(maxsize=500, ttl=_price_ttl())
_news_cache:  TTLCache = TTLCache(maxsize=200, ttl=_news_ttl())

COMPANY_NAMES: dict[str, str] = {
    "AAPL": "Apple Inc.",        "MSFT": "Microsoft Corp.",    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",   "NVDA": "NVIDIA Corp.",       "META": "Meta Platforms",
    "TSLA": "Tesla Inc.",        "NFLX": "Netflix Inc.",       "AMD":  "Advanced Micro Devices",
    "PLTR": "Palantir Technologies","SOFI": "SoFi Technologies","RIVN": "Rivian Automotive",
    "COIN": "Coinbase Global",   "HOOD": "Robinhood Markets",  "MSTR": "MicroStrategy",
    "GME":  "GameStop Corp.",    "AMC":  "AMC Entertainment",  "SPY":  "SPDR S&P 500 ETF",
    "QQQ":  "Invesco QQQ Trust", "BABA": "Alibaba Group",      "NIO":  "NIO Inc.",
    "LCID": "Lucid Group",       "SNAP": "Snap Inc.",          "UBER": "Uber Technologies",
    "LYFT": "Lyft Inc.",         "SHOP": "Shopify Inc.",       "SQ":   "Block Inc.",
    "PYPL": "PayPal Holdings",   "ROKU": "Roku Inc.",          "TWLO": "Twilio Inc.",
    "JPM":  "JPMorgan Chase",    "BAC":  "Bank of America",    "GS":   "Goldman Sachs",
    "XOM":  "ExxonMobil",        "CVX":  "Chevron",
}


def get_price_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    key = f"price_{symbol}_{period}"
    if key in _price_cache:
        return _price_cache[key]
    df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for {symbol}")
    # Flatten MultiIndex columns if present (yfinance >= 0.2.38 with single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [c.lower() for c in df.columns]
    needed = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    if len(needed) < 4:
        raise ValueError(f"Unexpected columns for {symbol}: {list(df.columns)}")
    df = df[needed].dropna()
    _price_cache[key] = df
    return df


def get_quote(symbol: str) -> dict:
    key = f"quote_{symbol}"
    if key in _price_cache:
        return _price_cache[key]
    hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No quote data for {symbol}")
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = [c[0] for c in hist.columns]
    # Normalize to lowercase
    hist.columns = [c.lower() for c in hist.columns]
    current = float(hist["close"].iloc[-1])
    prev    = float(hist["close"].iloc[-2]) if len(hist) > 1 else current
    change_pct = (current - prev) / (prev + 1e-9) * 100
    result = {
        "symbol":     symbol,
        "name":       COMPANY_NAMES.get(symbol, symbol),
        "price":      round(current, 2),
        "change":     round(current - prev, 2),
        "change_pct": round(change_pct, 2),
        "volume":     int(hist["volume"].iloc[-1]),
        "avg_volume": int(hist["volume"].mean()),
    }
    _price_cache[key] = result
    return result


def _sentiment(text: str) -> tuple[float, str]:
    try:
        s = float(TextBlob(text).sentiment.polarity)
        label = "positive" if s > 0.1 else "negative" if s < -0.1 else "neutral"
        return round(s, 3), label
    except Exception:
        return 0.0, "neutral"


def fetch_news(symbol: str) -> list[dict]:
    key = f"news_{symbol}"
    if key in _news_cache:
        return _news_cache[key]

    items: list[dict] = []

    # Yahoo Finance RSS — free, no key needed
    rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(rss)
            if resp.status_code == 200:
                for entry in feedparser.parse(resp.text).entries[:10]:
                    title = entry.get("title", "")
                    score, label = _sentiment(title)
                    items.append({
                        "title": title, "source": "Yahoo Finance",
                        "url": entry.get("link", ""),
                        "published_at": entry.get("published", ""),
                        "sentiment": score, "sentiment_label": label,
                    })
    except Exception as e:
        logger.debug(f"RSS error {symbol}: {e}")

    # NewsAPI (optional)
    if config.NEWS_API_KEY:
        try:
            with httpx.Client(timeout=8) as client:
                resp = client.get(
                    "https://newsapi.org/v2/everything",
                    params={"q": symbol, "apiKey": config.NEWS_API_KEY,
                            "language": "en", "sortBy": "publishedAt", "pageSize": 10},
                )
                if resp.status_code == 200:
                    for a in resp.json().get("articles", []):
                        title = a.get("title") or ""
                        score, label = _sentiment(title + " " + (a.get("description") or ""))
                        items.append({
                            "title": title,
                            "source": a.get("source", {}).get("name", ""),
                            "url": a.get("url", ""),
                            "published_at": a.get("publishedAt", ""),
                            "sentiment": score, "sentiment_label": label,
                        })
        except Exception as e:
            logger.debug(f"NewsAPI error {symbol}: {e}")

    seen: set[str] = set()
    unique = []
    for item in items:
        k = item["title"][:60].lower()
        if k not in seen:
            seen.add(k)
            unique.append(item)

    _news_cache[key] = unique[:12]
    return unique[:12]


def compute_entry_exit(
    df: pd.DataFrame,
    direction: str,
    current_price: float,
    atr: float,
) -> dict:
    """
    Compute precise entry, stop-loss, and take-profit levels.

    Methodology (priority order):
      Entry  — nearest key level the price should confirm above/below:
               BB upper/lower band, recent swing pivot, or key SMA.
               For bullish: entry = max(current, BB upper, recent breakout level).
               A small ATR buffer is added so we buy on confirmed breakout.
      Stop   — placed below the most recent swing low (bullish) or above the
               most recent swing high (bearish), then widened by 0.25×ATR as
               breathing room. Never closer than 0.5×ATR from entry.
      TP1    — 1.5× risk from entry  (conservative, ~50% of position)
      TP2    — 2.5× risk from entry  (full target, remaining position)
      TP3    — 4.0× risk from entry  (runner, if momentum persists)

    All prices rounded to 2 decimal places.
    Returns dict with keys: entry, stop_loss, tp1, tp2, tp3,
            risk_per_share, risk_pct, rr1, rr2, rr3,
            entry_reason, stop_reason, support, resistance,
            position_pct, breakeven.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]

    # ── Swing pivots (last 20 bars) ───────────────────────────────────────────
    window = 5   # pivot window
    n      = len(df)

    swing_highs = []
    swing_lows  = []
    for i in range(window, min(n - window, n)):
        if high.iloc[i] == high.iloc[i-window:i+window+1].max():
            swing_highs.append(float(high.iloc[i]))
        if low.iloc[i]  == low.iloc[i-window:i+window+1].min():
            swing_lows.append(float(low.iloc[i]))

    # Most recent swing levels (last 20 bars)
    recent_highs = sorted(swing_highs[-10:], reverse=True) if swing_highs else []
    recent_lows  = sorted(swing_lows[-10:])                if swing_lows  else []

    sma20  = float(close.rolling(20).mean().iloc[-1])
    sma50  = float(close.rolling(50).mean().iloc[-1])
    std20  = float(close.rolling(20).std().iloc[-1])
    bb_up  = sma20 + 2 * std20
    bb_lo  = sma20 - 2 * std20

    # ── VWAP (rolling 20-day intraday approximation using daily OHLC) ─────────
    typical = ((df["high"] + df["low"] + df["close"]) / 3).iloc[-20:]
    vol20   = df["volume"].iloc[-20:]
    vwap    = float((typical * vol20).sum() / (vol20.sum() + 1e-9))

    # ── Entry logic ───────────────────────────────────────────────────────────
    entry_reason = ""
    stop_reason  = ""

    if direction == "bullish":
        # Candidate levels the price needs to clear
        candidates = [current_price]
        if bb_up > current_price * 0.998:   # BB upper is near/above → breakout above it
            candidates.append(bb_up + 0.01)
        above_sma20 = current_price > sma20
        if not above_sma20:                 # Not yet above SMA20 → wait for reclaim
            candidates.append(sma20 + 0.01)
        if recent_highs:
            # Nearest swing high above current price = breakout level
            levels_above = [h for h in recent_highs if h > current_price * 0.995]
            if levels_above:
                candidates.append(min(levels_above) + 0.01)

        entry = round(min(candidates) + atr * 0.10, 2)   # tiny confirmation buffer
        if entry <= current_price:
            entry = round(current_price + atr * 0.05, 2)

        # Entry reason
        if abs(entry - bb_up) < atr * 0.3:
            entry_reason = f"Breakout above BB upper band (${bb_up:.2f})"
        elif abs(entry - sma20) < atr * 0.3:
            entry_reason = f"Reclaim above SMA20 (${sma20:.2f})"
        else:
            entry_reason = f"Breakout above swing pivot (${entry:.2f})"

        # Stop: below nearest swing low, or BB lower, whichever is higher
        # (tighter stop = better R:R)
        stop_candidates = [bb_lo - atr * 0.25]
        levels_below = [l for l in recent_lows if l < entry * 0.999]
        if levels_below:
            stop_candidates.append(max(levels_below) - atr * 0.25)
        if vwap < entry:
            stop_candidates.append(vwap - atr * 0.15)

        stop_loss = round(max(stop_candidates), 2)
        # Enforce minimum distance
        if entry - stop_loss < atr * 0.5:
            stop_loss = round(entry - atr * 0.5, 2)

        if abs(stop_loss - bb_lo) < atr * 0.3:
            stop_reason = f"Below BB lower band (${bb_lo:.2f})"
        elif levels_below and abs(stop_loss - max(levels_below)) < atr * 0.4:
            stop_reason = f"Below swing low (${max(levels_below):.2f})"
        else:
            stop_reason = f"Below VWAP (${vwap:.2f})"

    else:  # bearish / short
        candidates = [current_price]
        if bb_lo < current_price * 1.002:
            candidates.append(bb_lo - 0.01)
        below_sma20 = current_price < sma20
        if not below_sma20:
            candidates.append(sma20 - 0.01)
        if recent_lows:
            levels_below = [l for l in recent_lows if l < current_price * 1.005]
            if levels_below:
                candidates.append(max(levels_below) - 0.01)

        entry = round(max(candidates) - atr * 0.10, 2)
        if entry >= current_price:
            entry = round(current_price - atr * 0.05, 2)

        if abs(entry - bb_lo) < atr * 0.3:
            entry_reason = f"Breakdown below BB lower band (${bb_lo:.2f})"
        elif abs(entry - sma20) < atr * 0.3:
            entry_reason = f"Break below SMA20 (${sma20:.2f})"
        else:
            entry_reason = f"Breakdown below swing pivot (${entry:.2f})"

        stop_candidates = [bb_up + atr * 0.25]
        levels_above = [h for h in recent_highs if h > entry * 1.001]
        if levels_above:
            stop_candidates.append(min(levels_above) + atr * 0.25)
        if vwap > entry:
            stop_candidates.append(vwap + atr * 0.15)

        stop_loss = round(min(stop_candidates), 2)
        if stop_loss - entry < atr * 0.5:
            stop_loss = round(entry + atr * 0.5, 2)

        if abs(stop_loss - bb_up) < atr * 0.3:
            stop_reason = f"Above BB upper band (${bb_up:.2f})"
        elif levels_above and abs(stop_loss - min(levels_above)) < atr * 0.4:
            stop_reason = f"Above swing high (${min(levels_above):.2f})"
        else:
            stop_reason = f"Above VWAP (${vwap:.2f})"

    # ── Risk / Reward ─────────────────────────────────────────────────────────
    risk = abs(entry - stop_loss)
    if risk < 1e-6:
        risk = atr * 0.5

    if direction == "bullish":
        tp1 = round(entry + 1.5 * risk, 2)
        tp2 = round(entry + 2.5 * risk, 2)
        tp3 = round(entry + 4.0 * risk, 2)
    else:
        tp1 = round(entry - 1.5 * risk, 2)
        tp2 = round(entry - 2.5 * risk, 2)
        tp3 = round(entry - 4.0 * risk, 2)

    rr1 = round(1.5, 1)
    rr2 = round(2.5, 1)
    rr3 = round(4.0, 1)

    risk_pct = round((risk / (entry + 1e-9)) * 100, 2)

    # Position size: risk 1% of portfolio per trade, capped at 5%
    # position_pct = (1% portfolio risk) / risk_pct_per_share
    position_pct = round(min(5.0, max(0.5, 1.0 / max(0.1, risk_pct) * 100)), 1)

    breakeven = round(entry + risk * 0.15, 2) if direction == "bullish" \
                else round(entry - risk * 0.15, 2)

    support    = round(min(bb_lo, min(recent_lows[-3:]) if recent_lows else bb_lo), 2)
    resistance = round(max(bb_up, max(recent_highs[:3]) if recent_highs else bb_up), 2)

    return {
        "entry":         entry,
        "stop_loss":     stop_loss,
        "tp1":           tp1,
        "tp2":           tp2,
        "tp3":           tp3,
        "rr1":           rr1,
        "rr2":           rr2,
        "rr3":           rr3,
        "risk_per_share": round(risk, 2),
        "risk_pct":      risk_pct,
        "position_pct":  position_pct,
        "breakeven":     breakeven,
        "entry_reason":  entry_reason,
        "stop_reason":   stop_reason,
        "support":       support,
        "resistance":    resistance,
        "vwap":          round(vwap, 2),
        "bb_upper":      round(bb_up, 2),
        "bb_lower":      round(bb_lo, 2),
        "sma20":         round(sma20, 2),
        "sma50":         round(sma50, 2),
    }


def compute_option_play(
    direction: str,
    expiry_bucket: str,
    price: float,
    entry: float,
    tp1: float,
    tp2: float,
    stop_loss: float,
    atr: float,
    confidence: float,
    bb_squeeze: bool,
    vol_ratio: float,
    rsi: float,
    iv_estimate: Optional[float] = None,
) -> dict:
    """
    Recommend a concrete options play based on the signal setup.

    Returns a dict with:
      strategy      — human label  e.g. "Long Call", "Call Debit Spread"
      contract      — short code   e.g. "03/06 85c"
      strike        — float
      strike2       — float | None  (short leg for spreads)
      expiry_str    — "MM/DD" string
      expiry_date   — date ISO string
      rationale     — one-line reason
      max_profit    — "unlimited" | "$X.XX"
      max_loss      — "premium paid" | "$X.XX"
    """
    today = date.today()

    # ── 1. Pick expiry date from bucket ───────────────────────────────────────
    def _next_friday(n_weeks: int = 0) -> date:
        days_ahead = 4 - today.weekday()   # 4 = Friday
        if days_ahead <= 0:
            days_ahead += 7
        return today + timedelta(days=days_ahead + n_weeks * 7)

    def _monthly_opex(month_offset: int = 0) -> date:
        year  = today.year + (today.month + month_offset - 1) // 12
        month = (today.month + month_offset - 1) % 12 + 1
        cal   = calendar.monthcalendar(year, month)
        fridays = [week[4] for week in cal if week[4] != 0]
        return date(year, month, fridays[2])   # 3rd Friday = monthly OPEX

    bucket_map: dict = {
        "0dte":      today if today.weekday() < 5 else _next_friday(0),
        "2dte":      _next_friday(0),
        "weeklies":  _next_friday(1),
        "monthlies": _monthly_opex(1),
        "yearly":    _monthly_opex(3),
    }
    expiry_date = bucket_map.get(expiry_bucket, _next_friday(1))
    expiry_str  = f"{expiry_date.month}/{expiry_date.day}"

    dte = (expiry_date - today).days

    # ── 2. Round strike to nearest standard increment ──────────────────────────
    def _round_strike(px: float) -> float:
        if px < 5:    inc = 0.50
        elif px < 20:  inc = 1.0
        elif px < 50:  inc = 1.0
        elif px < 200: inc = 5.0
        else:          inc = 10.0
        return round(round(px / inc) * inc, 2)

    # ── 3. Determine IV context (simple heuristic if not provided) ─────────────
    if iv_estimate is None:
        # Approximate IV from ATR: IV ≈ (ATR / price) * sqrt(252)
        iv_estimate = round((atr / (price + 1e-9)) * (252 ** 0.5) * 100, 1)

    high_iv = iv_estimate > 60    # high IV → prefer spreads to cap premium cost
    low_dte = dte <= 2

    # ── 4. Choose strategy ────────────────────────────────────────────────────
    is_bull = direction == "bullish"
    high_conf = confidence >= 75
    explosive = bb_squeeze and vol_ratio >= 2.0

    if explosive and not high_iv and not low_dte:
        # Straddle when a big move is expected but direction uncertain
        # But if direction is strong, stay directional
        if confidence < 60:
            strategy = "Straddle"
        else:
            strategy = "Long Call" if is_bull else "Long Put"
    elif high_iv and high_conf:
        strategy = "Call Debit Spread" if is_bull else "Put Debit Spread"
    elif low_dte:
        # 0/2 DTE — directional only, ATM
        strategy = "Long Call" if is_bull else "Long Put"
    elif is_bull:
        strategy = "Long Call" if not high_iv else "Call Debit Spread"
    else:
        strategy = "Long Put" if not high_iv else "Put Debit Spread"

    # ── 5. Set strikes ────────────────────────────────────────────────────────
    if strategy in ("Long Call", "Long Put"):
        # Slightly OTM (better leverage, defined risk)
        otm_buffer = atr * 0.25
        if strategy == "Long Call":
            strike  = _round_strike(entry + otm_buffer)
        else:
            strike  = _round_strike(entry - otm_buffer)
        strike2 = None

        suffix = "c" if strategy == "Long Call" else "p"
        contract = f"{expiry_str} ${strike:.0f}{suffix}"
        max_profit = "Unlimited" if strategy == "Long Call" else f"~${abs(strike - 0):.0f} (stock → 0)"
        max_loss   = "Premium paid"

        # Rationale
        if strategy == "Long Call":
            target_move = round(tp1 - entry, 2)
            rationale = (
                f"OTM call on bullish breakout. Entry ${entry:.2f} → TP1 ${tp1:.2f} "
                f"(+${target_move:.2f}). Expires {expiry_str}."
            )
        else:
            target_move = round(entry - tp1, 2)
            rationale = (
                f"OTM put on bearish breakdown. Entry ${entry:.2f} → TP1 ${tp1:.2f} "
                f"(−${target_move:.2f}). Expires {expiry_str}."
            )

    elif strategy in ("Call Debit Spread", "Put Debit Spread"):
        otm_buffer = atr * 0.20
        spread_width = _round_strike(abs(tp1 - entry) * 0.6)  # short leg near TP1
        spread_width = max(spread_width, 2.5 if price < 50 else 5.0)

        if strategy == "Call Debit Spread":
            strike  = _round_strike(entry + otm_buffer)
            strike2 = _round_strike(strike + spread_width)
            suffix  = "c"
        else:
            strike  = _round_strike(entry - otm_buffer)
            strike2 = _round_strike(strike - spread_width)
            suffix  = "p"

        contract   = f"{expiry_str} ${strike:.0f}/{strike2:.0f}{suffix}s"
        max_profit = f"${abs(strike2 - strike):.2f} − premium"
        max_loss   = "Net debit paid"
        rationale  = (
            f"{'Bull' if is_bull else 'Bear'} debit spread — caps cost in high-IV environment. "
            f"Long ${strike:.0f} / Short ${strike2:.0f}. Expires {expiry_str}."
        )

    elif strategy == "Straddle":
        strike  = _round_strike(price)   # ATM
        strike2 = strike                  # same strike, both legs
        contract   = f"{expiry_str} ${strike:.0f} straddle"
        max_profit = "Unlimited"
        max_loss   = "Net debit (both premiums)"
        rationale  = (
            f"BB squeeze + volume surge — big move expected but direction uncertain. "
            f"ATM straddle at ${strike:.0f}. Expires {expiry_str}."
        )
    else:
        # Fallback
        strike  = _round_strike(entry)
        strike2 = None
        suffix  = "c" if is_bull else "p"
        contract   = f"{expiry_str} ${strike:.0f}{suffix}"
        max_profit = "Unlimited"
        max_loss   = "Premium paid"
        rationale  = f"Directional play on {direction} signal."

    return {
        "strategy":    strategy,
        "contract":    contract,
        "strike":      strike,
        "strike2":     strike2,
        "expiry_str":  expiry_str,
        "expiry_date": expiry_date.isoformat(),
        "rationale":   rationale,
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "iv_estimate": iv_estimate,
        "dte":         dte,
    }


def compute_sentiment_score(news: list[dict]) -> float:
    if not news:
        return 50.0
    avg = sum(n["sentiment"] for n in news) / len(news)
    return round((avg + 1) / 2 * 100, 1)


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute technical indicators from an OHLCV DataFrame.
    Returns a plain dict — no Pydantic, no external ta library needed.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    # RSI 14
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9)))

    # MACD
    ema12      = close.ewm(span=12, adjust=False).mean()
    ema26      = close.ewm(span=26, adjust=False).mean()
    macd_line  = ema12 - ema26
    macd_sig   = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist  = float(macd_line.iloc[-1] - macd_sig.iloc[-1])
    macd_hist2 = float(macd_line.iloc[-2] - macd_sig.iloc[-2]) if len(df) > 2 else 0.0

    # Bollinger Bands 20,2
    sma20  = close.rolling(20).mean()
    std20  = close.rolling(20).std()
    bb_up  = (sma20 + 2 * std20).iloc[-1]
    bb_lo  = (sma20 - 2 * std20).iloc[-1]
    bb_mid = sma20.iloc[-1]

    bb_width     = float((bb_up - bb_lo) / (bb_mid + 1e-9))
    bb_width_avg = float(
        ((sma20 + 2*std20 - (sma20 - 2*std20)) / (sma20 + 1e-9))
        .rolling(20).mean().iloc[-1]
    )
    is_squeeze = bb_width < bb_width_avg * 0.85

    # Volume
    vol_avg   = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / (vol_avg + 1e-9)

    # Trend
    sma50      = float(close.rolling(50).mean().iloc[-1])
    sma200_s   = close.rolling(200).mean()
    sma200     = float(sma200_s.iloc[-1]) if not pd.isna(sma200_s.iloc[-1]) else sma50
    current    = float(close.iloc[-1])
    sma20_val  = float(sma20.iloc[-1])

    # ATR 14
    tr  = pd.concat([high - low,
                     (high - close.shift(1)).abs(),
                     (low  - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])

    # Stochastic 14,3
    lowest14  = low.rolling(14).min()
    highest14 = high.rolling(14).max()
    stoch_k   = float((close - lowest14).iloc[-1] / ((highest14 - lowest14).iloc[-1] + 1e-9) * 100)

    # 52-week high
    high52  = float(high.rolling(min(252, len(high))).max().iloc[-1])
    pct52   = (current - high52) / (high52 + 1e-9) * 100

    # OBV slope
    obv_raw   = (volume * close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
    obv_slope = float(obv_raw.diff(5).iloc[-1])

    # Build signals list
    signals: list[dict] = []

    if rsi < 30:
        signals.append({"name": "RSI Oversold",     "type": "bullish",
                         "desc": f"RSI {rsi:.1f} — deeply oversold, potential bounce"})
    elif rsi > 70:
        signals.append({"name": "RSI Overbought",   "type": "bearish",
                         "desc": f"RSI {rsi:.1f} — overbought"})

    if macd_hist > 0 and macd_hist2 <= 0:
        signals.append({"name": "MACD Bullish Cross","type": "bullish",
                         "desc": "MACD crossed above signal line"})
    elif macd_hist < 0 and macd_hist2 >= 0:
        signals.append({"name": "MACD Bearish Cross","type": "bearish",
                         "desc": "MACD crossed below signal line"})

    if is_squeeze:
        signals.append({"name": "BB Squeeze",       "type": "bullish" if macd_hist > 0 else "neutral",
                         "desc": "Bollinger Band compression — big move imminent"})

    if vol_ratio > 2.0:
        signals.append({"name": f"Volume Surge {vol_ratio:.1f}x", "type": "bullish" if current > sma20_val else "bearish",
                         "desc": f"Volume {vol_ratio:.1f}x 20-day average — institutional activity"})
    elif vol_ratio > 1.5:
        signals.append({"name": f"Above-Avg Volume {vol_ratio:.1f}x", "type": "bullish" if current > sma20_val else "neutral",
                         "desc": f"Volume {vol_ratio:.1f}x 20-day average"})

    if pct52 > -2:
        signals.append({"name": "Near 52-Week High", "type": "bullish",
                         "desc": "Within 2% of 52-week high — potential breakout"})

    if sma50 > sma200 and current > sma50:
        signals.append({"name": "Above Golden Cross", "type": "bullish",
                         "desc": "Price > SMA50 > SMA200 — strong uptrend"})
    elif sma50 < sma200:
        signals.append({"name": "Below Death Cross",  "type": "bearish",
                         "desc": "SMA50 < SMA200 — bearish trend"})

    if stoch_k < 20:
        signals.append({"name": "Stochastic Oversold", "type": "bullish",
                         "desc": f"Stochastic %K {stoch_k:.1f} — oversold"})

    if obv_slope > 0 and current > sma20_val:
        signals.append({"name": "OBV Uptrend", "type": "bullish",
                         "desc": "On-Balance Volume rising — buying pressure"})

    # Technical score
    tech_score = 50.0
    for s in signals:
        if s["type"] == "bullish": tech_score += 8
        elif s["type"] == "bearish": tech_score -= 8
    if rsi < 30:    tech_score += 10
    elif rsi > 70:  tech_score -= 8
    if macd_hist > 0: tech_score += 7
    else:             tech_score -= 7
    if vol_ratio > 1.5: tech_score += 8
    if current > sma20_val > sma50: tech_score += 8
    tech_score = round(max(0, min(100, tech_score)), 1)

    return {
        "rsi":        round(rsi, 1),
        "macd_hist":  round(macd_hist, 4),
        "macd_hist2": round(macd_hist2, 4),
        "bb_upper":   round(float(bb_up), 2),
        "bb_lower":   round(float(bb_lo), 2),
        "bb_middle":  round(float(bb_mid), 2),
        "bb_squeeze": is_squeeze,
        "vol_ratio":  round(vol_ratio, 2),
        "sma20":      round(sma20_val, 2),
        "sma50":      round(sma50, 2),
        "sma200":     round(sma200, 2),
        "atr":        round(atr, 2),
        "stoch_k":    round(stoch_k, 1),
        "pct_from_52w": round(pct52, 1),
        "obv_slope":  round(obv_slope, 0),
        "tech_score": tech_score,
        "signals":    signals,
    }
