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
from data_engine import (
    get_price_data, get_quote, fetch_news, compute_sentiment_score, compute_indicators,
    compute_entry_exit, compute_option_play,
    fetch_options_flow, fetch_social_sentiment,
)
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
_results_loaded_from_db: bool = False

# Throttle: max parallel yfinance requests (be a good citizen)
_MAX_WORKERS = 20
# How many top results to keep in memory
_TOP_N = 200


# ── Public API ────────────────────────────────────────────────────────────────

def get_latest_results() -> list[dict]:
    global _results_loaded_from_db
    with _results_lock:
        # On first call after a restart, pre-populate from DB so results are
        # immediately available while the fresh scan runs in the background.
        if not _results_loaded_from_db:
            _results_loaded_from_db = True
            if not _results:
                persisted, scanned_at = db.load_scan_results()
                if persisted:
                    _results.extend(persisted)
                    logger.info(
                        f"Loaded {len(persisted)} persisted results from DB "
                        f"(as of {scanned_at})"
                    )
        return list(_results)


def get_last_scan_time() -> Optional[datetime]:
    if _last_scan_time:
        return _last_scan_time
    # Fall back to the timestamp from the persisted DB results
    try:
        _, scanned_at = db.load_scan_results()
        if scanned_at:
            return datetime.fromisoformat(scanned_at)
    except Exception:
        pass
    return None


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

        # Geopolitical & rumor news — always surface when present
        geo_news = [n for n in news if "geopolitical" in n.get("tags", [])]
        rumor_news = [n for n in news if "rumor" in n.get("tags", [])]
        for n in geo_news[:2]:
            catalysts.append(f'🌍 Geo: "{n["title"][:55]}"')
        for n in rumor_news[:2]:
            catalysts.append(f'💬 Rumor: "{n["title"][:55]}"')

        # Options flow (yfinance) + Finnhub social sentiment (rumors/chatter)
        unusual_flow = fetch_options_flow(symbol)
        social_sent = fetch_social_sentiment(symbol)
        flow_bias = unusual_flow["bias"] if unusual_flow else None
        if unusual_flow and unusual_flow.get("net_call_put"):
            net = unusual_flow["net_call_put"]
            direction_label = "call" if net > 0 else "put"
            catalysts.append(
                f"📈 Options flow: ${abs(net):,.0f} net {direction_label} premium"
            )
        if social_sent and social_sent.get("buzz", 0) > 100:
            catalysts.append(
                f"📊 Social buzz: {social_sent['buzz_label']} ({social_sent['buzz']} mentions)"
            )

        price = quote.get("price", 0.0)
        atr   = ind["atr"] or (price * 0.02)

        # Precise entry / exit levels derived from swing pivots, VWAP, BB, SMAs
        trade = compute_entry_exit(df, direction, price, atr)

        # Assign expiry bucket first so option play can use it
        expiry_signal = _assign_expiry_signal(
            price=price, sma20=ind["sma20"], sma50=ind["sma50"], sma200=ind["sma200"],
            bb_squeeze=ind["bb_squeeze"], vol_ratio=ind["vol_ratio"],
            confidence=confidence, catalysts=catalysts, news=news,
            sentiment_score=sentiment_score, macd_hist=ind["macd_hist"],
        )

        # Concrete options play recommendation — uses real expiry chain from yfinance
        # flow_bias from options flow can nudge strategy when smart money aligns
        option_play = compute_option_play(
            direction=direction,
            expiry_bucket=expiry_signal,
            price=price,
            entry=trade["entry"],
            tp1=trade["tp1"],
            tp2=trade["tp2"],
            stop_loss=trade["stop_loss"],
            atr=atr,
            confidence=confidence,
            bb_squeeze=ind["bb_squeeze"],
            vol_ratio=ind["vol_ratio"],
            rsi=ind["rsi"],
            symbol=symbol,
            flow_bias=flow_bias,
        )

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
            "atr":             round(atr, 2),
            "sma20":           ind["sma20"],
            "sma50":           ind["sma50"],
            "sma200":          ind["sma200"],
            "signals":         ind["signals"],
            "catalysts":       catalysts,
            "news":            news[:5],
            "sentiment_score": sentiment_score,
            "scanned_at":      datetime.now(timezone.utc).isoformat(),
            # ── Precise trade plan ──
            "entry":           trade["entry"],
            "stop_loss":       trade["stop_loss"],
            "take_profit_1":   trade["tp1"],
            "take_profit_2":   trade["tp2"],
            "take_profit_3":   trade["tp3"],
            "risk_per_share":  trade["risk_per_share"],
            "risk_pct":        trade["risk_pct"],
            "rr1":             trade["rr1"],
            "rr2":             trade["rr2"],
            "rr3":             trade["rr3"],
            "breakeven":       trade["breakeven"],
            "position_pct":    trade["position_pct"],
            "entry_reason":    trade["entry_reason"],
            "stop_reason":     trade["stop_reason"],
            "support":         trade["support"],
            "resistance":      trade["resistance"],
            "vwap":            trade["vwap"],
            # keep old keys for backward compat with persisted results
            "risk_reward":     trade["rr2"],
            "expiry_signal":   expiry_signal,
            # ── Options play ──
            "option_strategy": option_play["strategy"],
            "option_contract": option_play["contract"],
            "option_strike":   option_play["strike"],
            "option_strike2":  option_play["strike2"],
            "option_expiry":   option_play["expiry_str"],
            "option_dte":      option_play["dte"],
            "option_rationale": option_play["rationale"],
            "option_max_profit": option_play["max_profit"],
            "option_max_loss":  option_play["max_loss"],
            "iv_estimate":     option_play["iv_estimate"],
            "expected_move":   option_play.get("expected_move"),
            "option_breakeven": option_play.get("option_breakeven"),
            "option_premium":  option_play.get("option_premium"),
            "contract_iv":     option_play.get("contract_iv"),
            "contract_volume": option_play.get("contract_volume"),
            "contract_oi":     option_play.get("contract_oi"),
            "earnings_warning": option_play.get("earnings_warning", ""),
            "ex_div_warning":  option_play.get("ex_div_warning", ""),
            "next_earnings":   option_play.get("next_earnings"),
            "next_ex_div":     option_play.get("next_ex_div"),
            # ── Alternative data (options flow, social, geo, rumors) ──
            "unusual_flow":    unusual_flow,
            "social_sentiment": social_sent,
            "geo_news":        geo_news[:3],
            "rumor_news":      rumor_news[:3],
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

        # Persist to DB so results survive restarts and market-close periods
        try:
            db.save_scan_results(top)
        except Exception as e:
            logger.warning(f"Could not persist scan results: {e}")

        _last_scan_time = datetime.now(timezone.utc)
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(_scan_progress["started_at"])).seconds
        logger.info(
            f"Scan complete: {len(batch_results)} analyzed, top {len(top)} stored "
            f"| {elapsed}s elapsed"
        )

    finally:
        with _progress_lock:
            _scan_progress.update({"running": False, "phase": "idle"})
