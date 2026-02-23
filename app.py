"""
BreakoutAI — Full-Market Streamlit Dashboard.
Scans the entire investable universe (~3,000 tickers), pre-screens for
liquidity, then runs ML + technical analysis in parallel.
"""
import time
import threading
import logging
from datetime import datetime, timezone

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

import database as db
import config
import scanner
from ml_model import is_model_trained, train_model
from alerts import send_test_email
from data_engine import get_price_data
from universe import get_full_universe

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreakoutAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Robinhood-inspired CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Robinhood dark theme */
.stApp { background: #000000; font-family: 'Inter', -apple-system, sans-serif; }
[data-testid="stSidebar"] {
    background: #0C0C0E;
    border-right: 1px solid rgba(255,255,255,0.06);
}
.block-container { padding: 1.5rem 2rem 2rem; max-width: 1200px; }

/* Robinhood green #00C805, red #F23645 */
.metric-card {
    background: #0C0C0E;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: background 0.2s;
}
.metric-card:hover { background: #141418; }
.metric-value { font-size: 2rem; font-weight: 700; font-family: -apple-system, BlinkMacSystemFont, sans-serif; letter-spacing: -0.02em; }
.metric-label { font-size: 0.7rem; color: rgba(255,255,255,0.45); font-weight: 500; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 6px; }

/* High-confidence cards — clean Robinhood style */
.alert-card {
    background: #0C0C0E;
    border: 1px solid rgba(0,200,5,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: background 0.2s, border-color 0.2s;
}
.alert-card:hover { background: #141418; border-color: rgba(0,200,5,0.35); }
.alert-card-bear {
    background: #0C0C0E;
    border: 1px solid rgba(242,54,69,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.alert-card-bear:hover { background: #141418; border-color: rgba(242,54,69,0.35); }

/* Signal badges — minimal pills */
.sig-bull { background: rgba(0,200,5,0.12); color: #00C805; border-radius: 8px; padding: 4px 10px; font-size: 0.7rem; font-weight: 600; display: inline-block; margin: 2px; }
.sig-bear { background: rgba(242,54,69,0.12); color: #F23645; border-radius: 8px; padding: 4px 10px; font-size: 0.7rem; font-weight: 600; display: inline-block; margin: 2px; }
.sig-neut { background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.5); border-radius: 8px; padding: 4px 10px; font-size: 0.7rem; display: inline-block; margin: 2px; }

.live-dot { width: 6px; height: 6px; background: #00C805; border-radius: 50%; display: inline-block; margin-right: 8px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.4;} }

.progress-bar-outer { background: rgba(255,255,255,0.08); border-radius: 8px; height: 6px; overflow: hidden; margin: 8px 0; }
.progress-bar-inner { height: 100%; border-radius: 8px; background: #00C805; transition: width 0.5s ease; }

/* Robinhood-style stock row */
.stock-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-bottom: 1px solid rgba(255,255,255,0.06);
    transition: background 0.15s; cursor: pointer;
}
.stock-row:hover { background: rgba(255,255,255,0.03); }
.stock-row:last-child { border-bottom: none; }
.stock-symbol { font-size: 1.1rem; font-weight: 700; color: #fff; }
.stock-name { font-size: 0.8rem; color: rgba(255,255,255,0.45); margin-top: 2px; }
.stock-price { font-size: 1rem; font-weight: 600; color: #fff; text-align: right; }
.stock-change { font-size: 0.95rem; font-weight: 600; text-align: right; }
.stock-conf { font-size: 0.85rem; font-weight: 600; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0; border-bottom: 1px solid rgba(255,255,255,0.08); }
.stTabs [data-baseweb="tab"] { padding: 12px 20px; font-weight: 500; font-size: 0.9rem; }
.stTabs [aria-selected="true"] { border-bottom: 2px solid #00C805; color: #00C805; }

/* Buttons */
.stButton > button { border-radius: 12px; font-weight: 600; }
.stButton > button[kind="primary"] { background: #00C805; color: #000; border: none; }
.stButton > button[kind="primary"]:hover { background: #00a804; color: #000; }

/* Selectbox / inputs */
.stSelectbox > div, .stTextInput > div > div { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── One-time init (cached per process) ───────────────────────────────────────

@st.cache_resource
def _init() -> bool:
    db.init_db()
    scanner.start_scheduler()

    def _bg_train():
        try:
            train_model()
        except Exception as e:
            logging.getLogger(__name__).error(f"Model training: {e}")

    threading.Thread(target=_bg_train, daemon=True, name="trainer").start()
    scanner.force_scan_now()
    return True

_init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cc(c: float) -> str:
    if c >= 90: return "#00C805"   # Robinhood green
    if c >= 75: return "#22c55e"
    if c >= 60: return "#eab308"
    if c >= 45: return "#f97316"
    return "#F23645"  # Robinhood red

def _de(d: str) -> str:
    return "🟢" if d == "bullish" else "🔴" if d == "bearish" else "⚪"

def _fp(p: float) -> str:
    return f"+{p:.2f}%" if p >= 0 else f"{p:.2f}%"

def _bar(score: float, color: str) -> str:
    return (f'<div style="background:rgba(255,255,255,0.08);border-radius:4px;height:5px;'
            f'overflow:hidden;margin-top:3px;display:inline-block;width:60px;">'
            f'<div style="width:{min(100,score)}%;height:100%;background:{color};border-radius:4px;"></div></div>')

def _badges(signals: list[dict]) -> str:
    out = ""
    for s in signals[:5]:
        cls = "sig-bull" if s["type"]=="bullish" else "sig-bear" if s["type"]=="bearish" else "sig-neut"
        out += f'<span class="{cls}">{s["name"]}</span>'
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## BreakoutAI")
    st.markdown("*Full market scanner*")
    st.markdown("---")

    prog   = scanner.get_scan_progress()
    running = scanner.is_running()
    last_scan = scanner.get_last_scan_time()

    sc_color = "#00C805" if running else "rgba(255,255,255,0.5)"
    st.markdown(
        f'<span class="live-dot"></span> Scanner: <strong style="color:{sc_color};">{"Running" if running else "Idle"}</strong>',
        unsafe_allow_html=True,
    )
    if last_scan:
        st.caption(f"Last complete: {last_scan.strftime('%b %d %H:%M UTC')}")

    # Show live progress if scanning
    if prog["running"]:
        phase = prog["phase"]
        total = prog["total"] or 1
        done  = prog["done"]
        pct   = int(done / total * 100) if total else 0
        st.markdown(f"**Phase:** {phase.title()}")
        st.markdown(
            f'<div class="progress-bar-outer">'
            f'<div class="progress-bar-inner" style="width:{pct}%;"></div></div>'
            f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.5);">{done}/{total} symbols</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Filters
    st.markdown("### 🔧 Screener Filters")
    min_conf = st.slider("Min confidence %", 0, 100, 60, 5, key="min_conf")
    direction_filter = st.selectbox("Direction", ["All", "Bullish only", "Bearish only"], key="dir_filter")
    min_vol_ratio = st.slider("Min volume ratio", 0.0, 5.0, 0.0, 0.5, key="min_vol")
    max_results = st.select_slider("Show top N", [25, 50, 100, 200], value=50, key="top_n")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.caption(f"Alert threshold: **{config.ALERT_THRESHOLD:.0f}%**")
    st.caption(f"Scan interval: **{config.SCAN_INTERVAL_MINUTES} min**")
    st.caption(f"Min price: **${config.MIN_PRICE:.0f}** · Min vol: **{config.MIN_AVG_VOLUME//1000:,}k**")

    if st.button("🔄 Scan Now", use_container_width=True):
        scanner.force_scan_now()
        st.toast("Full-market scan triggered!", icon="🔍")

    st.markdown("---")

    if config.EMAIL_CONFIGURED:
        st.success("✅ Email alerts active")
        if st.button("📧 Test Email"):
            ok, msg = send_test_email()
            st.success(msg) if ok else st.error(f"Failed: {msg}")
    else:
        st.warning("⚠️ Email not configured")
        with st.expander("Setup"):
            st.markdown("""
Add to **Streamlit Cloud Secrets**:
```toml
ALERT_EMAIL_FROM     = "you@gmail.com"
ALERT_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"
ALERT_EMAIL_TO       = "you@gmail.com"
```
Use a Gmail **App Password**, not your normal password.
[Generate one here](https://myaccount.google.com/apppasswords)
            """)

    st.markdown("---")
    st.caption(f"ML model: {'✅' if is_model_trained() else '⏳ training…'}")
    st.caption("BreakoutAI · Educational use only")


# ── Apply filters ─────────────────────────────────────────────────────────────

all_results = scanner.get_latest_results()
filtered = [r for r in all_results if r["confidence"] >= min_conf]
if direction_filter == "Bullish only":
    filtered = [r for r in filtered if r["direction"] == "bullish"]
elif direction_filter == "Bearish only":
    filtered = [r for r in filtered if r["direction"] == "bearish"]
if min_vol_ratio > 0:
    filtered = [r for r in filtered if r["vol_ratio"] >= min_vol_ratio]
filtered = filtered[:max_results]


# ── Tabs ──────────────────────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs([
    "Scanner", "Alerts", "Chart", "Model",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
with t1:

    # KPI row
    high_conf = [r for r in all_results if r["confidence"] >= config.ALERT_THRESHOLD]
    bullish   = [r for r in all_results if r["direction"] == "bullish"]
    bearish   = [r for r in all_results if r["direction"] == "bearish"]
    top       = all_results[0] if all_results else None

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        universe_count = len(get_full_universe()) if all_results else 0
        universe_display = f"{universe_count:,}" if universe_count else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Universe</div>'
                    f'<div class="metric-value" style="color:rgba(255,255,255,0.6);">{universe_display}</div></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Scanned</div>'
                    f'<div class="metric-value" style="color:#fff;">{len(all_results):,}</div></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Bullish</div>'
                    f'<div class="metric-value" style="color:#00C805;">{len(bullish):,}</div></div>',
                    unsafe_allow_html=True)
    with k4:
        bull_95 = sum(1 for r in high_conf if r["direction"] == "bullish")
        bear_95 = sum(1 for r in high_conf if r["direction"] == "bearish")
        st.markdown(f'<div class="metric-card"><div class="metric-label">≥{config.ALERT_THRESHOLD:.0f}%</div>'
                    f'<div class="metric-value" style="color:#00C805;">{len(high_conf):,}</div>'
                    f'<div style="font-size:0.68rem;color:rgba(255,255,255,0.45);">{bull_95} bull · {bear_95} bear</div></div>',
                    unsafe_allow_html=True)
    with k5:
        tc  = f"{top['confidence']:.1f}%" if top else "—"
        tsym = top["symbol"] if top else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Top Pick</div>'
                    f'<div class="metric-value" style="color:#00C805;">{tc}</div>'
                    f'<div style="font-size:0.75rem;color:rgba(255,255,255,0.5);">{tsym}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Live scan progress — Robinhood-style minimal banner
    if prog["running"]:
        total = prog["total"] or 1
        done  = prog["done"]
        pct   = int(done / total * 100) if total else 0
        st.markdown(
            f'<div style="background:#0C0C0E;border:1px solid rgba(255,255,255,0.06);border-radius:16px;padding:16px 20px;">'
            f'<div style="color:rgba(255,255,255,0.8);font-size:0.9rem;font-weight:500;margin-bottom:8px;">'
            f'Scanning… {done:,} / {total:,} symbols</div>'
            f'<div class="progress-bar-outer"><div class="progress-bar-inner" style="width:{pct}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

    # High-confidence alert cards — Bullish & Bearish at 95%+
    if high_conf:
        st.markdown(f"### High-Confidence Alerts (Bullish & Bearish) ≥ {config.ALERT_THRESHOLD:.0f}%")
        for r in sorted(high_conf, key=lambda x: x["confidence"], reverse=True)[:10]:
            cls  = "alert-card" if r["direction"] == "bullish" else "alert-card-bear"
            col  = _cc(r["confidence"])
            cc   = "#00C805" if r["change_pct"] >= 0 else "#F23645"
            cats = "".join(
                f'<div style="font-size:0.76rem;color:rgba(255,255,255,0.6);margin-top:3px;">▸ {c}</div>'
                for c in r["catalysts"][:3]
            )
            cat_label = "Catalysts" if r["direction"] == "bullish" else "Risks"
            profit_row = ""
            if r.get("stop_loss") is not None and r.get("take_profit_1") is not None:
                profit_row = f"""
              <div style="margin-top:10px;padding:10px 14px;background:rgba(255,255,255,0.03);border-radius:10px;font-size:0.8rem;">
                <span style="color:rgba(255,255,255,0.5);">Trade plan:</span>
                <span style="color:#F23645;margin-left:8px;">SL ${r['stop_loss']:.2f}</span>
                <span style="color:#00C805;margin-left:12px;">TP1 ${r['take_profit_1']:.2f}</span>
                <span style="color:#00C805;margin-left:6px;">TP2 ${r['take_profit_2']:.2f}</span>
                <span style="color:rgba(255,255,255,0.5);margin-left:12px;">S/R ${r.get('support',0):.2f} / ${r.get('resistance',0):.2f}</span>
                <span style="color:rgba(255,255,255,0.5);margin-left:12px;">R:R 1:{r.get('risk_reward',2):.1f}</span>
                <span style="color:rgba(255,255,255,0.5);margin-left:8px;">· {r.get('position_pct',2):.1f}% max</span>
              </div>"""
            st.markdown(f"""
            <div class="{cls}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
                <div>
                  <span style="font-size:1.25rem;font-weight:800;color:white;font-family:monospace;">{r['symbol']}</span>
                  <span style="font-size:0.78rem;color:rgba(255,255,255,0.5);margin-left:8px;">{r.get('name','')[:30]}</span>
                  <span style="font-size:0.7rem;font-weight:600;color:{col};margin-left:8px;text-transform:uppercase;">{r['direction']}</span>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:1.5rem;font-weight:800;color:{col};font-family:monospace;">{r['confidence']:.1f}%</div>
                  <div style="font-size:0.65rem;color:rgba(255,255,255,0.45);letter-spacing:0.08em;">CONFIDENCE</div>
                </div>
              </div>
              <div style="margin-top:8px;display:flex;gap:16px;flex-wrap:wrap;">
                <span style="color:rgba(255,255,255,0.45);font-size:0.72rem;">PRICE</span> <span style="color:#fff;font-weight:600;">${r['price']:.2f}</span>
                <span style="color:rgba(255,255,255,0.45);font-size:0.72rem;">TODAY</span> <span style="color:{cc};font-weight:600;">{_fp(r['change_pct'])}</span>
                <span style="color:rgba(255,255,255,0.45);font-size:0.72rem;">SCORE</span> <span style="color:#fff;font-weight:600;">{r['final_score']:.0f}/100</span>
                <span style="color:rgba(255,255,255,0.45);font-size:0.72rem;">RSI</span>   <span style="color:#fff;font-weight:500;">{r['rsi']:.1f}</span>
                <span style="color:rgba(255,255,255,0.45);font-size:0.72rem;">VOL</span>   <span style="color:#fff;font-weight:500;">{r['vol_ratio']:.1f}x</span>
              </div>
              {profit_row}
              <div style="margin-top:6px;">{_badges(r['signals'])}</div>
              {('<div style="margin-top:6px;"><span style="font-size:0.7rem;color:rgba(255,255,255,0.45);">' + cat_label + ':</span>' + cats + '</div>') if r['catalysts'] else ''}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    # Full results table
    if not all_results:
        st.info(
            "🔄 Full-market scan in progress…\n\n"
            "**Stage 1** — Pre-screening ~3,000 tickers for liquidity (~2 min)\n\n"
            "**Stage 2** — Running ML + indicators on ~400-800 candidates in parallel (~5 min)\n\n"
            "Results will appear here automatically. You can also check the sidebar progress bar."
        )
    else:
        st.markdown(f"### 📊 Top Results  ·  *{len(filtered):,} shown, {len(all_results):,} scanned*")

        # Robinhood-style stock list
        rows_html = ""
        for r in filtered:
            col   = _cc(r["confidence"])
            cc    = "#00C805" if r["change_pct"] >= 0 else "#F23645"
            sqz   = ' <span class="sig-bull">SQZ</span>' if r.get("bb_squeeze") else ""
            badge = _badges(r["signals"][:3])
            rows_html += f"""
            <tr class="stock-row">
              <td style="padding:16px 20px;white-space:nowrap;border-bottom:1px solid rgba(255,255,255,0.06);">
                <div class="stock-symbol">{_de(r['direction'])} {r['symbol']}</div>
                <div class="stock-name">{r.get('name','')[:28]}</div>
              </td>
              <td style="padding:16px 12px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.06);">
                <div class="stock-price">${r['price']:.2f}</div>
              </td>
              <td style="padding:16px 12px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.06);">
                <div class="stock-change" style="color:{cc};">{_fp(r['change_pct'])}</div>
              </td>
              <td style="padding:16px 12px;border-bottom:1px solid rgba(255,255,255,0.06);">
                <div class="stock-conf" style="color:{col};">{r['confidence']:.1f}%</div>
                <div style="background:rgba(255,255,255,0.08);border-radius:4px;height:4px;margin-top:4px;width:60px;">
                  <div style="width:{min(100,r['confidence'])}%;height:100%;background:{col};border-radius:4px;"></div>
                </div>
              </td>
              <td style="padding:16px 12px;text-align:right;color:rgba(255,255,255,0.5);font-size:0.85rem;border-bottom:1px solid rgba(255,255,255,0.06);">{r['rsi']:.0f}</td>
              <td style="padding:16px 20px;font-size:0.8rem;border-bottom:1px solid rgba(255,255,255,0.06);">{badge}{sqz}</td>
            </tr>"""

        st.markdown(f"""
        <div style="overflow-x:auto;max-height:620px;overflow-y:auto;background:#0C0C0E;border:1px solid rgba(255,255,255,0.06);border-radius:16px;">
        <table style="width:100%;border-collapse:collapse;font-size:0.9rem;">
          <thead>
            <tr style="position:sticky;top:0;background:#0C0C0E;z-index:1;border-bottom:1px solid rgba(255,255,255,0.08);">
              <th style="padding:12px 20px;text-align:left;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">Symbol</th>
              <th style="padding:12px;text-align:right;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">Price</th>
              <th style="padding:12px;text-align:right;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">Chg%</th>
              <th style="padding:12px;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">Confidence</th>
              <th style="padding:12px;text-align:right;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">RSI</th>
              <th style="padding:12px 20px;color:rgba(255,255,255,0.5);font-weight:500;font-size:0.75rem;">Signals</th>
            </tr>
          </thead>
          <tbody style="background:#000;">{rows_html}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c_ar, c_ref = st.columns([3, 1])
    with c_ar:
        auto = st.checkbox("Auto-refresh every 60 s", value=False, key="auto_refresh")
    with c_ref:
        if st.button("🔄 Refresh"):
            st.rerun()
    if auto:
        time.sleep(60)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### Alert History")
    st.caption(f"Bullish & bearish alerts when confidence ≥ {config.ALERT_THRESHOLD:.0f}% · 4h throttle per symbol+direction")

    alerts = db.get_alerts(limit=200)
    if not alerts:
        st.info("No alerts yet. The scanner runs the full market every 30 minutes and will alert you when a breakout is detected.")
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Alerts", len(alerts))
        a2.metric("Emails Sent",  sum(1 for a in alerts if a["email_sent"]))
        a3.metric("Avg Confidence", f"{sum(a['confidence'] for a in alerts)/len(alerts):.1f}%")
        st.markdown("---")

        for alert in alerts:
            ts = alert["triggered_at"][:19].replace("T", " ")
            ei = "📧" if alert["email_sent"] else "🔕"
            with st.expander(
                f"{_de(alert['direction'])} **{alert['symbol']}** — {alert['confidence']:.1f}% · {ts} {ei}"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Price",  f"${alert['price']:.2f}")
                c2.metric("Change", _fp(alert["change_pct"]))
                c3.metric("Score",  f"{alert['score']:.0f}/100")
                if alert["catalysts"]:
                    label = "Catalysts" if alert["direction"] == "bullish" else "Risks"
                    st.markdown(f"**{label}:**")
                    for c in alert["catalysts"]:
                        st.markdown(f"- {c}")
                st.caption(f"Direction: {alert['direction'].upper()} · Email: {'sent' if alert['email_sent'] else 'not sent'}")

        st.markdown("---")
        df_exp = pd.DataFrame(alerts)
        df_exp["catalysts"] = df_exp["catalysts"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else x
        )
        st.download_button(
            "📥 Export CSV", df_exp.to_csv(index=False),
            file_name="breakout_alerts.csv", mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHART
# ═══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 📈 Price Chart + Indicators")

    live = scanner.get_latest_results()
    sym_opts = [r["symbol"] for r in live] if live else []

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        chart_sym = st.selectbox("Symbol", sym_opts or ["AAPL"], key="chart_sym")
    with sc2:
        chart_per = st.selectbox("Period", ["3mo","6mo","1y","2y"], index=1, key="chart_per")
    with sc3:
        custom_sym = st.text_input("Or type any ticker", placeholder="e.g. ORCL", key="custom_sym").upper().strip()

    sym = custom_sym if custom_sym else chart_sym

    if sym:
        with st.spinner(f"Loading {sym}…"):
            try:
                df = get_price_data(sym, period=chart_per).reset_index()
                df.columns = [c.lower() for c in df.columns]
                dc = df.columns[0]
                close = df["close"]
                df["sma20"]   = close.rolling(20).mean()
                df["sma50"]   = close.rolling(50).mean()
                df["sma200"]  = close.rolling(200).mean()
                df["vol_avg"] = df["volume"].rolling(20).mean()

                # Bollinger Bands
                std20 = close.rolling(20).std()
                df["bb_up"] = df["sma20"] + 2*std20
                df["bb_lo"] = df["sma20"] - 2*std20

                fig = go.Figure()
                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df[dc], open=df["open"], high=df["high"],
                    low=df["low"], close=close, name=sym,
                    increasing_line_color="#00C805", decreasing_line_color="#F23645",
                    increasing_fillcolor="#00C805", decreasing_fillcolor="#F23645",
                ))
                # MAs — Robinhood-style subtle lines
                fig.add_trace(go.Scatter(x=df[dc], y=df["sma20"],  name="SMA 20",
                                          line=dict(color="#00C805", width=1.5)))
                fig.add_trace(go.Scatter(x=df[dc], y=df["sma50"],  name="SMA 50",
                                          line=dict(color="#eab308", width=1.5, dash="dot")))
                fig.add_trace(go.Scatter(x=df[dc], y=df["sma200"], name="SMA 200",
                                          line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dash")))
                # Bollinger Bands
                fig.add_trace(go.Scatter(x=df[dc], y=df["bb_up"], name="BB Upper",
                                          line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=df[dc], y=df["bb_lo"], name="BB Lower",
                                          line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                                          fill="tonexty", fillcolor="rgba(255,255,255,0.03)"))

                # Annotate if high-confidence
                res = next((r for r in live if r["symbol"] == sym), None)
                if res and res["confidence"] >= config.ALERT_THRESHOLD:
                    fig.add_annotation(
                        x=df[dc].iloc[-1], y=float(close.iloc[-1]),
                        text=f"⚡ {res['confidence']:.0f}%",
                        showarrow=True, arrowhead=2,
                        arrowcolor="#00C805",
                        font=dict(color="#00C805", size=13),
                        bgcolor="rgba(0,200,5,0.1)", bordercolor="rgba(0,200,5,0.4)",
                    )

                fig.update_layout(
                    paper_bgcolor="#000000", plot_bgcolor="#0C0C0E",
                    font=dict(color="rgba(255,255,255,0.7)"), height=440,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", rangeslider_visible=False),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickprefix="$"),
                    legend=dict(bgcolor="#0C0C0E", bordercolor="rgba(255,255,255,0.08)", orientation="h", y=-0.15),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Volume
                vc = ["#00C805" if c >= o else "#F23645" for c, o in zip(df["close"], df["open"])]
                fv = go.Figure()
                fv.add_trace(go.Bar(x=df[dc], y=df["volume"], marker_color=vc, opacity=0.7, name="Volume"))
                fv.add_trace(go.Scatter(x=df[dc], y=df["vol_avg"], name="20d avg",
                                         line=dict(color="#eab308", width=1.5)))
                fv.update_layout(
                    paper_bgcolor="#000000", plot_bgcolor="#0C0C0E",
                    font=dict(color="rgba(255,255,255,0.7)"), height=140,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    margin=dict(l=10, r=10, t=5, b=10), showlegend=False,
                )
                st.plotly_chart(fv, use_container_width=True)

                if res:
                    st.markdown("---")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("ML Confidence",  f"{res['confidence']:.1f}%")
                    m2.metric("Breakout Score", f"{res['final_score']:.0f}/100")
                    m3.metric("RSI",            f"{res['rsi']:.1f}")
                    m4.metric("Volume Ratio",   f"{res['vol_ratio']:.2f}x")
                    m5.metric("Direction",      res["direction"].upper())
                    if res.get("stop_loss") is not None and res.get("take_profit_1") is not None:
                        st.markdown("**Trade plan (ATR-based):**")
                        p1, p2, p3, p4, p5 = st.columns(5)
                        p1.metric("Stop-loss", f"${res['stop_loss']:.2f}")
                        p2.metric("TP1", f"${res['take_profit_1']:.2f}")
                        p3.metric("TP2", f"${res['take_profit_2']:.2f}")
                        p4.metric("Support / Resistance", f"${res.get('support',0):.0f} / ${res.get('resistance',0):.0f}")
                        p5.metric("R:R · Position", f"1:{res.get('risk_reward',2):.1f} · {res.get('position_pct',2):.1f}%")
                    if res["catalysts"]:
                        st.markdown("**Catalysts / Risks:**")
                        for c in res["catalysts"]:
                            st.markdown(f"- {c}")

            except Exception as e:
                st.error(f"Chart error for {sym}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 🤖 ML Model")
    col_a, col_b = st.columns(2)
    with col_a:
        status = "✅ Trained & active" if is_model_trained() else "⏳ Training in background…"
        st.markdown(f"""
**Status:** {status}

**Algorithm:** Gradient Boosting + Platt Calibration  
**Training data:** 27 stocks × 2 years daily OHLCV  
**Breakout label:** +5% within 5 trading days  
**Features (18):** RSI+slope, MACD hist+slope, BB squeeze ratio,
BB %B, Volume ratio (1d+5d), Price vs SMA 20/50/200,
SMA slope, ATR%, 5d & 20d returns, distance from 52w high,
OBV slope, BB width  
**Validation:** 3-fold CV ROC-AUC
        """)

    with col_b:
        st.markdown(f"""
**Full-market scan pipeline:**

1. Fetch S&P 500 + NASDAQ-100 + Russell 2000 (~3,000 tickers)
2. Batch pre-screen: price ≥ ${config.MIN_PRICE:.0f}, avg volume ≥ {config.MIN_AVG_VOLUME//1000:,}k
3. Deep analysis on ~400-800 candidates with **{scanner._MAX_WORKERS} parallel threads**
4. Results sorted by ML confidence, top 200 stored

**Alert fires when ALL:**  
✅ Confidence ≥ {config.ALERT_THRESHOLD:.0f}%  
✅ Direction is bullish  
✅ Not alerted same symbol in last 4 hours

**Confidence = 60% ML + 40% rule-based**  
(RSI, MACD, volume, trend)
        """)
        st.markdown("---")
        from ml_model import TRAINING_SYMBOLS
        st.caption("Training symbols: " + ", ".join(TRAINING_SYMBOLS))
        if st.button("🔁 Retrain Model (~3 min)"):
            with st.spinner("Training…"):
                try:
                    train_model(force=True)
                    st.success("Model retrained!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer:** BreakoutAI is for educational and research purposes only. "
        "ML scores do not guarantee profitable trades. Always do your own due diligence."
    )
