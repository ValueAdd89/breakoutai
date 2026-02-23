"""
Data ingestion layer — synchronous, cache-backed.
Works on Streamlit Cloud (no asyncio required).
"""
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
import feedparser
import yfinance as yf
import pandas as pd
from textblob import TextBlob
from cachetools import TTLCache

import config

logger = logging.getLogger(__name__)

_price_cache: TTLCache = TTLCache(maxsize=300, ttl=300)
_news_cache:  TTLCache = TTLCache(maxsize=100, ttl=300)

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
    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    _price_cache[key] = df
    return df


def get_quote(symbol: str) -> dict:
    key = f"quote_{symbol}"
    if key in _price_cache:
        return _price_cache[key]
    hist = yf.Ticker(symbol).history(period="5d", auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No quote data for {symbol}")
    current = float(hist["Close"].iloc[-1])
    prev    = float(hist["Close"].iloc[-2]) if len(hist) > 1 else current
    change_pct = (current - prev) / (prev + 1e-9) * 100
    result = {
        "symbol":     symbol,
        "name":       COMPANY_NAMES.get(symbol, symbol),
        "price":      round(current, 2),
        "change":     round(current - prev, 2),
        "change_pct": round(change_pct, 2),
        "volume":     int(hist["Volume"].iloc[-1]),
        "avg_volume": int(hist["Volume"].mean()),
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
