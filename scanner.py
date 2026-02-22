"""
Background scanner — APScheduler runs in a daemon thread every N minutes.
Thread-safe shared state lets the Streamlit UI read latest results without
blocking or async complexity.
"""
import logging
import threading
from datetime import datetime, timezone
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

import config
import database as db
from data_engine import get_price_data, get_quote, fetch_news, compute_sentiment_score, compute_indicators
from ml_model import get_confidence
from alerts import send_alert_email

logger = logging.getLogger(__name__)

_scheduler: Optional[BackgroundScheduler] = None
_sched_lock = threading.Lock()

_results: list[dict] = []
_results_lock = threading.Lock()
_last_scan_time: Optional[datetime] = None

_alert_callbacks: list[Callable] = []


# ── Public API ────────────────────────────────────────────────────────────────

def get_latest_results() -> list[dict]:
    with _results_lock:
        return list(_results)


def get_last_scan_time() -> Optional[datetime]:
    return _last_scan_time


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def force_scan_now() -> None:
    """Trigger a scan immediately in a background thread."""
    t = threading.Thread(target=_run_scan, daemon=True, name="forced_scan")
    t.start()


def start_scheduler() -> None:
    """Start APScheduler. Idempotent — safe to call multiple times."""
    global _scheduler
    with _sched_lock:
        if _scheduler is not None and _scheduler.running:
            return
        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_job(
            _run_scan,
            trigger="interval",
            minutes=config.SCAN_INTERVAL_MINUTES,
            id="breakout_scan",
            replace_existing=True,
            max_instances=1,
        )
        _scheduler.start()
        logger.info(f"Background scanner started — every {config.SCAN_INTERVAL_MINUTES} min")


def stop_scheduler() -> None:
    global _scheduler
    with _sched_lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None


# ── Internal scan logic ───────────────────────────────────────────────────────

def _analyze(symbol: str) -> Optional[dict]:
    try:
        df    = get_price_data(symbol, period="1y")
        quote = get_quote(symbol)
        news  = fetch_news(symbol)

        ind            = compute_indicators(df)
        sentiment_score = compute_sentiment_score(news)
        confidence     = get_confidence(df)

        # Momentum sub-score
        mom_score = 50.0
        if ind["macd_hist"] > 0: mom_score += 15
        if ind["sma20"] > ind["sma50"]: mom_score += 10
        if 50 < ind["rsi"] < 65: mom_score += 8
        mom_score = max(0, min(100, mom_score))

        # Volume sub-score
        vr = ind["vol_ratio"]
        vol_score = 50.0 if vr < 3.0 else 90.0
        if vr >= 2.0: vol_score = 78.0
        elif vr >= 1.5: vol_score = 65.0
        elif vr < 0.7: vol_score = 30.0

        rule_score = (
            ind["tech_score"]  * 0.40
            + mom_score        * 0.25
            + vol_score        * 0.20
            + sentiment_score  * 0.15
        )
        # Blend: 60% ML + 40% rule-based
        final_score = round(max(0, min(100, rule_score * 0.4 + confidence * 0.6)), 1)

        if final_score >= 65 or confidence >= 65:
            direction = "bullish"
        elif final_score <= 35 or confidence <= 35:
            direction = "bearish"
        else:
            direction = "neutral"

        # Catalysts
        catalysts = [s["name"] for s in ind["signals"] if s["type"] == "bullish"][:3]
        if ind["bb_squeeze"]:
            catalysts.append("Bollinger Band Squeeze — big move imminent")
        pos_news = [n for n in news if n["sentiment_label"] == "positive"]
        if pos_news:
            catalysts.append(f'Positive news: "{pos_news[0]["title"][:60]}"')

        return {
            "symbol":          symbol,
            "name":            quote.get("name", symbol),
            "price":           quote.get("price", 0.0),
            "change_pct":      quote.get("change_pct", 0.0),
            "confidence":      confidence,
            "rule_score":      round(rule_score, 1),
            "final_score":     final_score,
            "direction":       direction,
            "rsi":             ind["rsi"],
            "macd_hist":       ind["macd_hist"],
            "vol_ratio":       ind["vol_ratio"],
            "bb_squeeze":      ind["bb_squeeze"],
            "atr":             ind["atr"],
            "sma20":           ind["sma20"],
            "sma50":           ind["sma50"],
            "sma200":          ind["sma200"],
            "signals":         ind["signals"],
            "catalysts":       catalysts,
            "news":            news[:5],
            "sentiment_score": sentiment_score,
            "scanned_at":      datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.warning(f"Analysis failed for {symbol}: {e}")
        return None


def _run_scan() -> None:
    global _last_scan_time
    watchlist = db.get_watchlist()
    if not watchlist:
        logger.info("Watchlist empty — skipping scan")
        return

    logger.info(f"Scanning {len(watchlist)} symbols…")
    results = []

    for symbol in watchlist:
        result = _analyze(symbol)
        if result is None:
            continue
        results.append(result)

        db.log_scan(
            symbol=symbol, score=result["final_score"],
            confidence=result["confidence"], direction=result["direction"],
            price=result["price"],
        )

        # Fire alert?
        if (result["confidence"] >= config.ALERT_THRESHOLD
                and result["direction"] == "bullish"
                and not db.already_alerted_recently(symbol, hours=4)):

            logger.info(f"🚨 ALERT: {symbol} at {result['confidence']:.1f}% confidence")

            email_sent = send_alert_email(
                symbol=symbol, name=result["name"],
                price=result["price"], change_pct=result["change_pct"],
                confidence=result["confidence"], score=result["final_score"],
                direction=result["direction"], catalysts=result["catalysts"],
                signals=result["signals"],
            )
            db.save_alert(
                symbol=symbol, name=result["name"],
                price=result["price"], change_pct=result["change_pct"],
                score=result["final_score"], confidence=result["confidence"],
                direction=result["direction"], catalysts=result["catalysts"],
                email_sent=email_sent,
            )
            for cb in _alert_callbacks:
                try:
                    cb(result)
                except Exception:
                    pass

    results.sort(key=lambda x: x["confidence"], reverse=True)
    with _results_lock:
        _results.clear()
        _results.extend(results)

    _last_scan_time = datetime.now(timezone.utc)
    logger.info(f"Scan complete — {len(results)} symbols processed")
