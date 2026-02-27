"""
Background scanner — full market mode.

Architecture for speed:
  1. Universe fetcher builds ~3,000 symbol list (S&P 500 + NASDAQ-100 + R2000)
  2. Fast batch pre-screener cuts it to ~400-800 liquid candidates (5d OHLCV batch)
  3. ThreadPoolExecutor runs the ML + indicators pipeline in parallel (20 workers)
  4. Results sorted by confidence; top 200 stored in shared state
  5. APScheduler fires the full cycle every SCAN_INTERVAL_MINUTES

Performance targets:
  • Pre-screen 3,000 tickers  : ~2-3 min  (batch yfinance, one HTTP req per 100)
  • Analyze 400 candidates    : ~4-6 min  (20 threads × ~0.8s per symbol)
  • Total wall-clock           : ~6-9 min on first run, then cached repeats are faster
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Callable, Optional

from apscheduler.schedulers.background import BackgroundScheduler

import config
import database as db
from universe import get_screened_candidates, get_universe_names
from data_engine import get_price_data, get_quote, fetch_news, compute_sentiment_score, compute_indicators
from ml_model import get_confidence
from alerts import send_alert_email

logger = logging.getLogger(__name__)

# ── Shared state ──────────────────────────────────────────────────────────────

_scheduler: Optional[BackgroundScheduler] = None
_sched_lock = threading.Lock()

_results: list[dict] = []       # top N results sorted by confidence
_results_lock = threading.Lock()

_scan_progress: dict = {         # live progress for the UI progress bar
    "running": False,
    "total": 0,
    "done": 0,
    "phase": "idle",             # "screening" | "analyzing" | "idle"
    "started_at": None,
}
_progress_lock = threading.Lock()

_last_scan_time: Optional[datetime] = None
_alert_callbacks: list[Callable] = []

# Throttle: max parallel yfinance requests (be a good citizen)
_MAX_WORKERS = 20
# How many top results to keep in memory
_TOP_N = 200


# ── Public API ────────────────────────────────────────────────────────────────

def get_latest_results() -> list[dict]:
    with _results_lock:
        return list(_results)


def get_last_scan_time() -> Optional[datetime]:
    return _last_scan_time


def get_scan_progress() -> dict:
    with _progress_lock:
        return dict(_scan_progress)


def is_running() -> bool:
    return _scheduler is not None and _scheduler.running


def force_scan_now() -> None:
    """Trigger a full scan immediately in a background thread."""
    t = threading.Thread(target=_run_full_scan, daemon=True, name="forced_scan")
    t.start()


def start_scheduler() -> None:
    """Start APScheduler. Safe to call multiple times."""
    global _scheduler
    with _sched_lock:
        if _scheduler is not None and _scheduler.running:
            return
        _scheduler = BackgroundScheduler(timezone="UTC")
        _scheduler.add_job(
            _run_full_scan,
            trigger="interval",
            minutes=config.SCAN_INTERVAL_MINUTES,
            id="market_scan",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        _scheduler.start()
        logger.info(f"Full-market scanner started — every {config.SCAN_INTERVAL_MINUTES} min")


def stop_scheduler() -> None:
    global _scheduler
    with _sched_lock:
        if _scheduler and _scheduler.running:
            _scheduler.shutdown(wait=False)
            _scheduler = None


# ── Expiry signal assignment ───────────────────────────────────────────────────

def _assign_expiry_signal(
    price: float, sma20: float, sma50: float, sma200: float,
    bb_squeeze: bool, vol_ratio: float, confidence: float,
    catalysts: list[str], news: list, sentiment_score: float, macd_hist: float,
) -> str:
    """
    Assign expiry bucket based on technicals + events.
    Stocks on verge of breakout: 0DTE (imminent) → Yearly (long-term trend).
    """
    has_catalyst = len(catalysts) >= 2 or (news and abs(sentiment_score - 50) > 15)
    strong_momentum = abs(macd_hist) > 0.01 if macd_hist else False
    above_sma200 = price > sma200 if sma200 else False
    above_sma50 = price > sma50 if sma50 else False

    # 0DTE: Explosive setup — BB squeeze + surge vol + catalyst or very high confidence
    if bb_squeeze and vol_ratio >= 2.5 and (has_catalyst or confidence >= 90):
        return "0dte"

    # 2DTE: Imminent — squeeze or vol spike + catalyst/momentum
    if (bb_squeeze or vol_ratio >= 2.0) and confidence >= 80 and (has_catalyst or strong_momentum):
        return "2dte"

    # Weeklies: Standard breakout — volume confirmation, good confidence
    if vol_ratio >= 1.5 and confidence >= 70:
        return "weeklies"

    # Monthlies: Established setup — trend + moderate confidence
    if confidence >= 65 and (above_sma50 or vol_ratio >= 1.2):
        return "monthlies"

    # Yearly: Major trend — LEAPS candidate
    if confidence >= 60 and above_sma200:
        return "yearly"

    # Default to weeklies for decent setups
    return "weeklies" if confidence >= 65 else "monthlies"


# ── Per-symbol analysis ───────────────────────────────────────────────────────

def _analyze(symbol: str, name: str) -> Optional[dict]:
    """
    Full analysis pipeline for one symbol.
    Designed to be called from a thread pool.
    """
    try:
        df    = get_price_data(symbol, period="1y")
        quote = get_quote(symbol)
        news  = fetch_news(symbol)

        ind             = compute_indicators(df)
        sentiment_score = compute_sentiment_score(news)
        confidence      = get_confidence(df)

        # Momentum sub-score
        mom_score = 50.0
        if ind["macd_hist"] > 0:
            mom_score += 15
        if ind["sma20"] > ind["sma50"]:
            mom_score += 10
        if 50 < ind["rsi"] < 65:
            mom_score += 8
        mom_score = max(0.0, min(100.0, mom_score))

        # Volume sub-score
        vr = ind["vol_ratio"]
        if vr >= 3.0:
            vol_score = 90.0
        elif vr >= 2.0:
            vol_score = 78.0
        elif vr >= 1.5:
            vol_score = 65.0
        elif vr < 0.7:
            vol_score = 30.0
        else:
            vol_score = 50.0

        rule_score = (
            ind["tech_score"]  * 0.40
            + mom_score        * 0.25
            + vol_score        * 0.20
            + sentiment_score  * 0.15
        )
        final_score = round(max(0.0, min(100.0, rule_score * 0.4 + confidence * 0.6)), 1)

        if final_score >= 55 or confidence >= 58:
            direction = "bullish"
        elif final_score <= 42 or confidence <= 38:
            direction = "bearish"
        else:
            direction = "neutral"

        # Catalysts (bullish) vs risks (bearish)
        if direction == "bullish":
            catalysts = [s["name"] for s in ind["signals"] if s["type"] == "bullish"][:3]
            if ind["bb_squeeze"]:
                catalysts.append("Bollinger Band Squeeze — big move imminent")
            pos_news = [n for n in news if n["sentiment_label"] == "positive"]
            if pos_news:
                catalysts.append(f'Positive: "{pos_news[0]["title"][:55]}"')
        else:
            catalysts = [s["name"] for s in ind["signals"] if s["type"] == "bearish"][:3]
            if ind["bb_squeeze"]:
                catalysts.append("BB Squeeze — breakdown risk")
            neg_news = [n for n in news if n["sentiment_label"] == "negative"]
            if neg_news:
                catalysts.append(f'Negative: "{neg_news[0]["title"][:55]}"')

        # Profit-enhancing: stop-loss, take-profit, support/resistance, position size
        price = quote.get("price", 0.0)
        atr = ind["atr"] or (price * 0.02)
        bb_lo = ind.get("bb_lower") or ind["sma20"] - atr
        bb_up = ind.get("bb_upper") or ind["sma20"] + atr
        if direction == "bullish":
            stop_loss = round(price - 1.5 * atr, 2)
            take_profit_1 = round(price + 2.0 * atr, 2)
            take_profit_2 = round(price + 3.0 * atr, 2)
            support = round(min(bb_lo, ind["sma20"]), 2)
            resistance = round(max(bb_up, ind["sma50"]), 2)
        else:
            stop_loss = round(price + 1.5 * atr, 2)
            take_profit_1 = round(price - 2.0 * atr, 2)
            take_profit_2 = round(price - 3.0 * atr, 2)
            support = round(bb_lo, 2)
            resistance = round(max(ind["sma20"], bb_up), 2)
        atr_pct = (atr / (price + 1e-9)) * 100
        risk_reward = round(2.0 * atr / (1.5 * atr + 1e-9), 1)
        # Lower vol = larger suggested position; cap at 5%
        position_pct = min(5.0, max(1.0, round(3.0 / max(0.5, atr_pct), 1)))

        return {
            "symbol":          symbol,
            "name":            quote.get("name") or name,
            "price":           price,
            "change_pct":      quote.get("change_pct", 0.0),
            "volume":          quote.get("volume", 0),
            "avg_volume":      quote.get("avg_volume", 0),
            "confidence":      confidence,
            "rule_score":      round(rule_score, 1),
            "final_score":     final_score,
            "direction":       direction,
            "rsi":             ind["rsi"],
            "macd_hist":       ind["macd_hist"],
            "vol_ratio":       ind["vol_ratio"],
            "bb_squeeze":      ind["bb_squeeze"],
            "atr":             atr,
            "sma20":           ind["sma20"],
            "sma50":           ind["sma50"],
            "sma200":          ind["sma200"],
            "signals":         ind["signals"],
            "catalysts":       catalysts,
            "news":            news[:5],
            "sentiment_score": sentiment_score,
            "scanned_at":      datetime.now(timezone.utc).isoformat(),
            "stop_loss":       stop_loss,
            "take_profit_1":   take_profit_1,
            "take_profit_2":   take_profit_2,
            "support":         support,
            "resistance":      resistance,
            "risk_reward":     risk_reward,
            "position_pct":    position_pct,
            "expiry_signal":   _assign_expiry_signal(
                price=price, sma20=ind["sma20"], sma50=ind["sma50"], sma200=ind["sma200"],
                bb_squeeze=ind["bb_squeeze"], vol_ratio=ind["vol_ratio"],
                confidence=confidence, catalysts=catalysts, news=news,
                sentiment_score=sentiment_score, macd_hist=ind["macd_hist"],
            ),
        }
    except Exception as e:
        logger.debug(f"Analysis failed for {symbol}: {e}")
        return None


def _fire_alert_if_needed(result: dict) -> None:
    """Check threshold and send email alert + DB record. Fires for BOTH bullish and bearish at 95%+."""
    if (result["confidence"] >= config.ALERT_THRESHOLD
            and not db.already_alerted_recently(result["symbol"], result["direction"], hours=4)):

        logger.info(f"🚨 ALERT: {result['symbol']} {result['direction'].upper()} at {result['confidence']:.1f}%")
        email_sent = send_alert_email(
            symbol=result["symbol"], name=result["name"],
            price=result["price"], change_pct=result["change_pct"],
            confidence=result["confidence"], score=result["final_score"],
            direction=result["direction"], catalysts=result["catalysts"],
            signals=result["signals"],
            stop_loss=result.get("stop_loss"), take_profit_1=result.get("take_profit_1"),
            take_profit_2=result.get("take_profit_2"), support=result.get("support"),
            resistance=result.get("resistance"), risk_reward=result.get("risk_reward"),
            position_pct=result.get("position_pct"),
        )
        db.save_alert(
            symbol=result["symbol"], name=result["name"],
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


# ── Main scan loop ────────────────────────────────────────────────────────────

def _run_full_scan() -> None:
    global _last_scan_time

    with _progress_lock:
        if _scan_progress["running"]:
            logger.info("Scan already in progress — skipping overlap")
            return
        _scan_progress.update({"running": True, "total": 0, "done": 0,
                                "phase": "screening", "started_at": datetime.now(timezone.utc).isoformat()})

    try:
        # ── Stage 1: pre-screen ───────────────────────────────────────────────
        logger.info("Stage 1: pre-screening market universe…")
        candidates = get_screened_candidates(
            min_price=config.MIN_PRICE,
            min_avg_volume=config.MIN_AVG_VOLUME,
        )

        if not candidates:
            logger.warning("Pre-screener returned 0 candidates — check network")
            return

        logger.info(f"Stage 2: analyzing {len(candidates)} candidates in parallel…")
        names = get_universe_names()

        with _progress_lock:
            _scan_progress.update({"phase": "analyzing", "total": len(candidates), "done": 0})

        # ── Stage 2: parallel deep analysis ──────────────────────────────────
        batch_results: list[dict] = []
        done_count = 0

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="scan") as pool:
            future_map = {
                pool.submit(_analyze, sym, names.get(sym, sym)): sym
                for sym in candidates
            }

            for future in as_completed(future_map):
                sym = future_map[future]
                done_count += 1

                with _progress_lock:
                    _scan_progress["done"] = done_count

                try:
                    result = future.result(timeout=30)
                except Exception as e:
                    logger.debug(f"Future error {sym}: {e}")
                    continue

                if result is None:
                    continue

                batch_results.append(result)

                db.log_scan(
                    symbol=result["symbol"], score=result["final_score"],
                    confidence=result["confidence"], direction=result["direction"],
                    price=result["price"],
                )
                _fire_alert_if_needed(result)

        # ── Sort and store top N ──────────────────────────────────────────────
        batch_results.sort(key=lambda x: x["confidence"], reverse=True)
        top = batch_results[:_TOP_N]

        with _results_lock:
            _results.clear()
            _results.extend(top)

        _last_scan_time = datetime.now(timezone.utc)
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(_scan_progress["started_at"])).seconds
        logger.info(
            f"Scan complete: {len(batch_results)} analyzed, top {len(top)} stored "
            f"| {elapsed}s elapsed"
        )

    finally:
        with _progress_lock:
            _scan_progress.update({"running": False, "phase": "idle"})
