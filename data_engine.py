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
_flow_cache:  TTLCache = TTLCache(maxsize=300, ttl=600)    # 10 min for options flow
_social_cache: TTLCache = TTLCache(maxsize=300, ttl=900)   # 15 min for social sentiment

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


# ── Geopolitical / macro / rumor keywords for news categorization ──────────────
_GEO_KEYWORDS = frozenset({
    "war", "military", "sanctions", "tariff", "trade war", "china", "russia",
    "iran", "ukraine", "taiwan", "nato", "oil", "opec", "inflation", "recession",
    "fed", "rate cut", "rate hike", "election", "congress", "white house",
    "geopolitical", "regulatory", "sec", "doj", "antitrust", "lawsuit",
    "ban", "embargo", "nuclear", "military strike", "border", "immigration",
})
_RUMOR_KEYWORDS = frozenset({
    "rumor", "rumour", "reportedly", "sources say", "insiders", "leak",
    "speculation", "hearing", "chatter", "buzz", "could", "might", "may",
    "exploring", "considering", "in talks", "merger talks", "buyout",
})


def _categorize_news(title: str, desc: str = "") -> list[str]:
    """Return tags like ['geopolitical'], ['rumor'], or [] for a news item."""
    text = (title + " " + (desc or "")).lower()
    tags = []
    if any(kw in text for kw in _GEO_KEYWORDS):
        tags.append("geopolitical")
    if any(kw in text for kw in _RUMOR_KEYWORDS):
        tags.append("rumor")
    return tags


def fetch_options_flow(symbol: str) -> Optional[dict]:
    """
    Fetch options call vs put flow from yfinance (free, no API key).
    Uses volume × lastPrice as a premium proxy from the live options chain.
    Returns {call_premium, put_premium, net_call_put, bias} or None.
    Bias: 'bullish' if net calls > puts, 'bearish' if net puts > calls, else 'neutral'.
    """
    key = f"flow_{symbol}"
    if key in _flow_cache:
        return _flow_cache[key]
    try:
        ticker = yf.Ticker(symbol)
        raw_opts = ticker.options or ()
        if not raw_opts:
            return None
        tot_call, tot_put = 0.0, 0.0
        for exp_str in list(raw_opts)[:2]:
            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue
            for df, is_call in [(chain.calls, True), (chain.puts, False)]:
                if df is None or df.empty:
                    continue
                cols_lower = {str(c).lower(): c for c in df.columns if c is not None}
                vol_col = cols_lower.get("volume")
                price_col = cols_lower.get("lastprice") or cols_lower.get("last")
                if vol_col is None:
                    continue
                vol = df[vol_col].fillna(0)
                if price_col is not None:
                    price = df[price_col].fillna(0)
                    premium = (vol * price * 100).sum()  # 100 shares per contract
                else:
                    premium = float(vol.sum()) * 100  # rough proxy
                if is_call:
                    tot_call += premium
                else:
                    tot_put += premium
        if tot_call == 0 and tot_put == 0:
            return None
        net = tot_call - tot_put
        bias = "bullish" if net > 0 else "bearish" if net < 0 else "neutral"
        out = {
            "call_premium": round(tot_call, 2),
            "put_premium":  round(tot_put, 2),
            "net_call_put": round(net, 2),
            "bias":         bias,
        }
        _flow_cache[key] = out
        return out
    except Exception as e:
        logger.debug(f"Options flow error {symbol}: {e}")
        return None


def fetch_social_sentiment(symbol: str) -> Optional[dict]:
    """
    Fetch Reddit/StockTwits sentiment from Finnhub (when key configured).
    Returns {reddit_sentiment, stocktwits_sentiment, overall, buzz} or None.
    Used as a proxy for rumors/chatter.
    """
    if not config.FINNHUB_KEY:
        return None
    key = f"social_{symbol}"
    if key in _social_cache:
        return _social_cache[key]
    try:
        to_dt = date.today()
        from_dt = to_dt - timedelta(days=7)
        with httpx.Client(timeout=8) as client:
            resp = client.get(
                "https://finnhub.io/api/v1/stock/social-sentiment",
                params={
                    "symbol": symbol,
                    "from": from_dt.isoformat(),
                    "to": to_dt.isoformat(),
                    "token": config.FINNHUB_KEY,
                },
            )
        if resp.status_code != 200:
            return None
        data = resp.json()
        reddit = data.get("reddit") or []
        twits = data.get("stockTwits") or []
        if not isinstance(reddit, list):
            reddit = [reddit] if reddit else []
        if not isinstance(twits, list):
            twits = [twits] if twits else []

        def _agg(entries: list) -> tuple:
            mentions = sum(int(e.get("mention", 0) or 0) for e in entries if isinstance(e, dict))
            pos = sum(float(e.get("positiveScore", 0) or 0) for e in entries if isinstance(e, dict))
            neg = sum(float(e.get("negativeScore", 0) or 0) for e in entries if isinstance(e, dict))
            n = len(entries) or 1
            sent = (pos - neg) / n if n else 0.5
            return mentions, max(0, min(1, 0.5 + sent))

        rb, rr = _agg(reddit)
        tb, rt = _agg(twits)
        overall = (rr + rt) / 2.0 if (rr or rt) else 0.5
        buzz = rb + tb
        out = {
            "reddit_sentiment": round(rr, 3),
            "stocktwits_sentiment": round(rt, 3),
            "overall": round(overall, 3),
            "buzz": buzz,
            "buzz_label": "high" if buzz > 500 else "medium" if buzz > 100 else "low",
        }
        _social_cache[key] = out
        return out
    except Exception as e:
        logger.debug(f"Finnhub social sentiment error {symbol}: {e}")
        return None


def fetch_news(symbol: str) -> list[dict]:
    key = f"news_{symbol}"
    if key in _news_cache:
        return _news_cache[key]

    items: list[dict] = []

    def _add_item(title: str, source: str, url: str, published: str, desc: str = ""):
        score, label = _sentiment(title + " " + desc)
        tags = _categorize_news(title, desc)
        items.append({
            "title": title, "source": source, "url": url,
            "published_at": published, "description": desc,
            "sentiment": score, "sentiment_label": label,
            "tags": tags,
        })

    # Yahoo Finance RSS — free, no key needed
    rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    try:
        with httpx.Client(timeout=8) as client:
            resp = client.get(rss)
            if resp.status_code == 200:
                for entry in feedparser.parse(resp.text).entries[:10]:
                    title = entry.get("title", "")
                    _add_item(title, "Yahoo Finance", entry.get("link", ""),
                              entry.get("published", ""), entry.get("summary", ""))
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
                        desc = a.get("description") or ""
                        _add_item(title, a.get("source", {}).get("name", ""),
                                  a.get("url", ""), a.get("publishedAt", ""), desc)
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


_expiry_cache:  TTLCache = TTLCache(maxsize=300, ttl=3_600)   # 1-hour cache per symbol
_strikes_cache: TTLCache = TTLCache(maxsize=300, ttl=3_600)   # strikes per symbol+expiry
_chain_cache:   TTLCache = TTLCache(maxsize=200, ttl=3_600)   # full chain per symbol+expiry
_events_cache:  TTLCache = TTLCache(maxsize=400, ttl=3_600)   # earnings/dividends per symbol


def get_option_expiries(symbol: str) -> list:
    """
    Fetch real option expiration dates from yfinance (same chain Robinhood uses).
    Returns a list of datetime.date objects sorted ascending, or [] on failure.
    Cached for 1 hour — expiry chains don't change intraday.
    """
    key = f"expiries_{symbol}"
    if key in _expiry_cache:
        return _expiry_cache[key]
    try:
        raw = yf.Ticker(symbol).options          # tuple of "YYYY-MM-DD" strings
        expiries = sorted(
            date.fromisoformat(d) for d in raw
            if date.fromisoformat(d) >= date.today()
        )
        _expiry_cache[key] = expiries
        return expiries
    except Exception as e:
        logger.debug(f"Option expiries fetch failed for {symbol}: {e}")
        _expiry_cache[key] = []
        return []


def get_option_chain(symbol: str, expiry_date: date) -> dict:
    """
    Fetch full options chain (calls + puts DataFrames) for symbol and expiry.
    Returns {"calls": DataFrame, "puts": DataFrame}. Cached 1h.
    """
    expiry_str = expiry_date.isoformat()
    key = f"chain_{symbol}_{expiry_str}"
    if key in _chain_cache:
        return _chain_cache[key]
    try:
        chain = yf.Ticker(symbol).option_chain(expiry_str)
        result = {"calls": chain.calls if chain.calls is not None else pd.DataFrame(), "puts": chain.puts if chain.puts is not None else pd.DataFrame()}
    except Exception as e:
        logger.debug(f"Option chain fetch failed {symbol} {expiry_str}: {e}")
        result = {"calls": pd.DataFrame(), "puts": pd.DataFrame()}
    _chain_cache[key] = result
    return result


def get_real_strikes(symbol: str, expiry_date: date) -> dict:
    """
    Fetch real call and put strikes from the live options chain for a specific expiry.
    Returns {"calls": [float, ...], "puts": [float, ...]} sorted ascending.
    Uses get_option_chain so the full chain is cached for contract lookups.
    """
    chain = get_option_chain(symbol, expiry_date)
    expiry_str = expiry_date.isoformat()
    key = f"strikes_{symbol}_{expiry_str}"
    if key in _strikes_cache:
        return _strikes_cache[key]
    try:
        calls = sorted(float(s) for s in chain["calls"]["strike"].dropna().tolist()) if not chain["calls"].empty else []
        puts  = sorted(float(s) for s in chain["puts"]["strike"].dropna().tolist()) if not chain["puts"].empty else []
        result = {"calls": calls, "puts": puts}
    except Exception:
        result = {"calls": [], "puts": []}
    _strikes_cache[key] = result
    return result


def _row_for_strike(df: pd.DataFrame, strike: float) -> Optional[dict]:
    """Get first row of DataFrame where strike column equals strike (within 0.02)."""
    if df is None or df.empty:
        return None
    strike_col = None
    for c in df.columns:
        if str(c).lower() == "strike":
            strike_col = c
            break
    if strike_col is None:
        return None
    col = df[strike_col].astype(float)
    idx = (col - strike).abs().argmin()
    match = df.iloc[idx]
    if abs(float(match.get(strike_col, 0)) - strike) > 0.02:
        return None
    return match.to_dict()


def get_earnings_dividends(symbol: str) -> dict:
    """
    Next earnings date and next ex-dividend date (free, yfinance).
    Returns {next_earnings_date, next_ex_div_date, earnings_warning, ex_div_warning}.
    Important for options: avoid holding through earnings (IV crush), ex-div (assignment).
    """
    key = f"events_{symbol}"
    if key in _events_cache:
        return _events_cache[key]
    out = {"next_earnings_date": None, "next_ex_div_date": None, "earnings_warning": "", "ex_div_warning": ""}
    try:
        ticker = yf.Ticker(symbol)
        today = date.today()

        def _to_date(d) -> Optional[date]:
            if d is None:
                return None
            if hasattr(d, "date"):
                return d.date() if hasattr(d, "date") else d
            if isinstance(d, str):
                try:
                    return date.fromisoformat(d[:10])
                except Exception:
                    return None
            return None

        # Earnings
        try:
            eds = getattr(ticker, "get_earnings_dates", None)
            if callable(eds):
                df = eds(limit=8)
                if df is not None and not df.empty:
                    for idx in (df.index.tolist() if hasattr(df.index, "tolist") else []):
                        d = _to_date(idx)
                        if d and d >= today:
                            out["next_earnings_date"] = d.isoformat()
                            break
        except Exception:
            pass

        # Dividends (ex-date)
        try:
            divs = ticker.get_dividends()
            if divs is not None and len(divs) > 0:
                for idx in (divs.index.tolist()[-8:] if hasattr(divs.index, "tolist") else []):
                    d = _to_date(idx)
                    if d and d >= today:
                        out["next_ex_div_date"] = d.isoformat()
                        break
        except Exception:
            pass

        if out["next_earnings_date"]:
            out["earnings_warning"] = f"Earnings {out['next_earnings_date']} — IV crush risk"
        if out["next_ex_div_date"]:
            out["ex_div_warning"] = f"Ex-div {out['next_ex_div_date']} — call assignment risk"
    except Exception as e:
        logger.debug(f"Earnings/dividends fetch {symbol}: {e}")
    _events_cache[key] = out
    return out


def _nearest_strike(ideal: float, strikes: list, direction: str = "nearest") -> Optional[float]:
    """
    Snap an ideal price to the nearest available strike.
    direction: "nearest" | "above" | "below"
    """
    if not strikes:
        return None
    if direction == "above":
        above = [s for s in strikes if s >= ideal]
        return above[0] if above else strikes[-1]
    if direction == "below":
        below = [s for s in strikes if s <= ideal]
        return below[-1] if below else strikes[0]
    # nearest
    return min(strikes, key=lambda s: abs(s - ideal))


def _pick_expiry_from_chain(
    expiries: list,
    bucket: str,
) -> Optional[date]:
    """
    Select the best real expiry date from the live chain for a given bucket.

    Bucket → target DTE range:
      0dte      → today (0 DTE) or nearest next session
      2dte      → 1–4 DTE
      weeklies  → 5–10 DTE
      monthlies → 25–45 DTE
      yearly    → 60–120 DTE
    """
    if not expiries:
        return None

    today = date.today()

    targets: dict = {
        "0dte":      (0, 1),
        "2dte":      (1, 4),
        "weeklies":  (5, 14),
        "monthlies": (21, 50),
        "yearly":    (55, 130),
    }
    lo, hi = targets.get(bucket, (5, 14))

    # Prefer an expiry whose DTE falls in the ideal range
    in_range = [e for e in expiries if lo <= (e - today).days <= hi]
    if in_range:
        return in_range[0]   # nearest in range

    # Fall back to the nearest expiry at or after the lower bound
    at_or_after = [e for e in expiries if (e - today).days >= lo]
    if at_or_after:
        return at_or_after[0]

    # Last resort: the nearest available expiry
    return expiries[0]


def _generic_round(px: float) -> float:
    """Generic strike rounding used only when the live chain is unavailable."""
    if px < 5:     inc = 0.50
    elif px < 50:  inc = 1.0
    elif px < 200: inc = 5.0
    else:          inc = 10.0
    return round(round(px / inc) * inc, 2)


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
    symbol: str = "",
    iv_estimate: Optional[float] = None,
    flow_bias: Optional[str] = None,
) -> dict:
    """
    Recommend a concrete options play based on the signal setup.
    Uses real option expiry dates from the live options chain (same as Robinhood).

    Returns a dict with:
      strategy      — human label  e.g. "Long Call", "Call Debit Spread"
      contract      — short code   e.g. "3/6 $85c"
      strike        — float
      strike2       — float | None  (short leg for spreads)
      expiry_str    — "M/D" string matching the real chain
      expiry_date   — date ISO string
      rationale     — one-line reason
      max_profit    — "Unlimited" | "$X.XX − premium"
      max_loss      — "Premium paid" | "Net debit paid"
    """
    today = date.today()

    # ── 1. Fetch real expiry dates; fall back to synthetic if unavailable ─────
    real_expiries = get_option_expiries(symbol) if symbol else []
    expiry_date = _pick_expiry_from_chain(real_expiries, expiry_bucket)

    if expiry_date is None:
        # Synthetic fallback (same logic as before) when chain is unavailable
        def _next_friday(n: int = 0) -> date:
            d = 4 - today.weekday()
            if d <= 0:
                d += 7
            return today + timedelta(days=d + n * 7)

        def _monthly_opex(mo: int = 0) -> date:
            yr  = today.year + (today.month + mo - 1) // 12
            mn  = (today.month + mo - 1) % 12 + 1
            cal = calendar.monthcalendar(yr, mn)
            fri = [w[4] for w in cal if w[4] != 0]
            return date(yr, mn, fri[2])

        fallback_map: dict = {
            "0dte":      today if today.weekday() < 5 else _next_friday(0),
            "2dte":      _next_friday(0),
            "weeklies":  _next_friday(1),
            "monthlies": _monthly_opex(1),
            "yearly":    _monthly_opex(3),
        }
        expiry_date = fallback_map.get(expiry_bucket, _next_friday(1))

    expiry_str = f"{expiry_date.month}/{expiry_date.day}"
    dte = (expiry_date - today).days

    # ── 2. Fetch real strikes for this expiry; build a snap helper ─────────────
    real_chain  = get_real_strikes(symbol, expiry_date) if symbol else {"calls": [], "puts": []}
    call_strikes = real_chain["calls"]
    put_strikes  = real_chain["puts"]

    def _snap_call(ideal: float, prefer: str = "above") -> float:
        """Return nearest real call strike; fall back to generic rounding."""
        real = _nearest_strike(ideal, call_strikes, prefer)
        if real is not None:
            return real
        return _generic_round(ideal)

    def _snap_put(ideal: float, prefer: str = "below") -> float:
        """Return nearest real put strike; fall back to generic rounding."""
        real = _nearest_strike(ideal, put_strikes, prefer)
        if real is not None:
            return real
        return _generic_round(ideal)

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

    # ── 5. Set strikes — snapped to real chain strikes where available ───────────
    otm_buffer = atr * 0.25

    if strategy == "Long Call":
        # Slightly OTM call: snap to nearest call strike at or above ideal
        ideal   = entry + otm_buffer
        strike  = _snap_call(ideal, prefer="above")
        strike2 = None
        contract   = f"{expiry_str} ${strike:.2f}c".replace(".00c", "c")
        max_profit = "Unlimited"
        max_loss   = "Premium paid"
        target_move = round(tp1 - entry, 2)
        rationale = (
            f"OTM call on bullish breakout. Entry ${entry:.2f} → TP1 ${tp1:.2f} "
            f"(+${target_move:.2f}). Expires {expiry_str}."
        )

    elif strategy == "Long Put":
        # Slightly OTM put: snap to nearest put strike at or below ideal
        ideal   = entry - otm_buffer
        strike  = _snap_put(ideal, prefer="below")
        strike2 = None
        contract   = f"{expiry_str} ${strike:.2f}p".replace(".00p", "p")
        max_profit = f"~${strike:.0f} max (stock → $0)"
        max_loss   = "Premium paid"
        target_move = round(entry - tp1, 2)
        rationale = (
            f"OTM put on bearish breakdown. Entry ${entry:.2f} → TP1 ${tp1:.2f} "
            f"(−${target_move:.2f}). Expires {expiry_str}."
        )

    elif strategy == "Call Debit Spread":
        # Long leg slightly OTM, short leg near TP1
        ideal_long  = entry + atr * 0.20
        ideal_short = tp1
        strike  = _snap_call(ideal_long,  prefer="above")
        strike2 = _snap_call(ideal_short, prefer="above")
        if strike2 <= strike:          # ensure spread has positive width
            # move short leg one strike higher
            above_long = [s for s in call_strikes if s > strike]
            strike2 = above_long[0] if above_long else strike + _generic_round(price * 0.025)
        width      = round(abs(strike2 - strike), 2)
        contract   = f"{expiry_str} ${strike:.2f}/${strike2:.2f}c".replace(".00c", "c").replace(".00/", "/")
        max_profit = f"${width:.2f} − debit"
        max_loss   = "Net debit paid"
        rationale  = (
            f"Bull call spread — defined risk in high-IV environment. "
            f"Long ${strike:.2f} / Short ${strike2:.2f}. Expires {expiry_str}."
        )

    elif strategy == "Put Debit Spread":
        ideal_long  = entry - atr * 0.20
        ideal_short = tp1
        strike  = _snap_put(ideal_long,  prefer="below")
        strike2 = _snap_put(ideal_short, prefer="below")
        if strike2 >= strike:          # ensure spread has positive width
            below_long = [s for s in put_strikes if s < strike]
            strike2 = below_long[-1] if below_long else strike - _generic_round(price * 0.025)
        width      = round(abs(strike - strike2), 2)
        contract   = f"{expiry_str} ${strike:.2f}/${strike2:.2f}p".replace(".00p", "p").replace(".00/", "/")
        max_profit = f"${width:.2f} − debit"
        max_loss   = "Net debit paid"
        rationale  = (
            f"Bear put spread — defined risk in high-IV environment. "
            f"Long ${strike:.2f} / Short ${strike2:.2f}. Expires {expiry_str}."
        )

    elif strategy == "Straddle":
        # ATM: snap call strike nearest to current price (put chain is same strike)
        ideal   = price
        strike  = _snap_call(ideal, prefer="nearest")
        strike2 = _nearest_strike(ideal, put_strikes, "nearest") or strike
        contract   = f"{expiry_str} ${strike:.2f} straddle".replace(".00 straddle", " straddle")
        max_profit = "Unlimited"
        max_loss   = "Net debit (both premiums)"
        rationale  = (
            f"BB squeeze + volume surge — big move expected. "
            f"ATM straddle at ${strike:.2f}. Expires {expiry_str}."
        )

    else:
        # Generic fallback
        strike  = _snap_call(entry, prefer="nearest") if is_bull else _snap_put(entry, prefer="nearest")
        strike2 = None
        suffix  = "c" if is_bull else "p"
        contract   = f"{expiry_str} ${strike:.2f}{suffix}".replace(f".00{suffix}", suffix)
        max_profit = "Unlimited"
        max_loss   = "Premium paid"
        rationale  = f"Directional play on {direction} signal."

    # Append options flow note when available
    if flow_bias and flow_bias != "neutral":
        flow_confirms = (is_bull and flow_bias == "bullish") or (not is_bull and flow_bias == "bearish")
        if flow_confirms:
            rationale += " Options flow confirms."
        else:
            rationale += " Options flow opposes — caution."

    # ── 6. Contract details: IV, liquidity, expected move, option breakeven ─────
    contract_iv = None
    contract_vol = None
    contract_oi = None
    option_premium = None
    expected_move = None
    option_breakeven = None
    if symbol:
        chain = get_option_chain(symbol, expiry_date)
        df_side = chain["calls"] if is_bull or "Call" in strategy else chain["puts"]
        if not df_side.empty:
            row = _row_for_strike(df_side, strike)
            if row:
                def _get(key_aliases, default=None):
                    for k in key_aliases:
                        if k in row and row[k] is not None and (not isinstance(row[k], float) or not pd.isna(row[k])):
                            return row[k]
                    return default
                contract_iv   = _get(["impliedVolatility", "iv", "implied volatility"], iv_estimate)
                if contract_iv is not None:
                    contract_iv = float(contract_iv) * 100 if float(contract_iv) < 2 else float(contract_iv)
                contract_vol  = _get(["volume", "Volume"])
                contract_oi   = _get(["openInterest", "open interest", "oi"])
                bid           = _get(["bid", "Bid"])
                ask           = _get(["ask", "Ask"])
                option_premium = _get(["lastPrice", "last", "close"])
                if contract_iv is not None and dte and dte > 0:
                    iv_dec = float(contract_iv) / 100.0
                    expected_move = round(price * iv_dec * (dte / 365) ** 0.5, 2)
                if option_premium is not None:
                    prem = float(option_premium)
                    option_breakeven = round(strike + prem, 2) if is_bull or "Call" in strategy else round(strike - prem, 2)

    # Earnings / ex-div (for warnings in UI)
    events = get_earnings_dividends(symbol) if symbol else {}

    return {
        "strategy":        strategy,
        "contract":        contract,
        "strike":          strike,
        "strike2":         strike2,
        "expiry_str":      expiry_str,
        "expiry_date":     expiry_date.isoformat(),
        "rationale":       rationale,
        "max_profit":      max_profit,
        "max_loss":        max_loss,
        "iv_estimate":     iv_estimate,
        "dte":             dte,
        "contract_iv":     round(contract_iv, 1) if contract_iv is not None else None,
        "contract_volume": int(contract_vol) if contract_vol is not None else None,
        "contract_oi":     int(contract_oi) if contract_oi is not None else None,
        "expected_move":   expected_move,
        "option_breakeven": option_breakeven,
        "option_premium":  round(float(option_premium), 2) if option_premium is not None else None,
        "earnings_warning": events.get("earnings_warning", ""),
        "ex_div_warning":  events.get("ex_div_warning", ""),
        "next_earnings":   events.get("next_earnings_date"),
        "next_ex_div":     events.get("next_ex_div_date"),
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
