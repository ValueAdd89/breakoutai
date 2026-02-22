"""
BreakoutAI — Streamlit dashboard.

Deployable to Streamlit Community Cloud from a GitHub repo.
Background scanner runs every N minutes and sends email alerts
when ML confidence ≥ threshold (default 95%).
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
from data_engine import get_price_data, COMPANY_NAMES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreakoutAI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0e1a; }
[data-testid="stSidebar"] { background-color: #0f1629; border-right: 1px solid #1c2840; }
.block-container { padding-top: 1.2rem; padding-bottom: 1rem; }

.metric-card {
    background: #0f1629; border: 1px solid #1c2840;
    border-radius: 12px; padding: 16px 20px; text-align: center;
}
.metric-value { font-size: 1.9rem; font-weight: 800; font-family: 'Courier New', monospace; }
.metric-label { font-size: 0.7rem; color: #64748b; font-weight: 600;
                letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; }

.alert-card {
    background: linear-gradient(135deg,#0f2a1e,#0a1f16);
    border: 1px solid #00ff8840; border-left: 4px solid #00ff88;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
}
.alert-card-bear {
    background: linear-gradient(135deg,#2a0f0f,#1f0a0a);
    border: 1px solid #ff456040; border-left: 4px solid #ff4560;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
}

.sig-bull { background:#00ff8820; color:#00ff88; border:1px solid #00ff8840;
             border-radius:20px; padding:2px 10px; font-size:0.72rem;
             font-weight:700; display:inline-block; margin:2px; }
.sig-bear { background:#ff456020; color:#ff4560; border:1px solid #ff456040;
             border-radius:20px; padding:2px 10px; font-size:0.72rem;
             font-weight:700; display:inline-block; margin:2px; }
.sig-neut { background:#33333550; color:#94a3b8; border:1px solid #ffffff15;
             border-radius:20px; padding:2px 10px; font-size:0.72rem;
             display:inline-block; margin:2px; }

.live-dot { width:8px; height:8px; background:#00ff88; border-radius:50%;
            display:inline-block; margin-right:6px;
            animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
</style>
""", unsafe_allow_html=True)


# ── One-time initialisation (cached per process) ──────────────────────────────

@st.cache_resource
def _init() -> bool:
    """Runs exactly once per Streamlit server process."""
    db.init_db()
    db.bulk_init_watchlist(config.DEFAULT_WATCHLIST)
    scanner.start_scheduler()

    def _bg_train():
        try:
            train_model()
        except Exception as e:
            logging.getLogger(__name__).error(f"Model training failed: {e}")

    threading.Thread(target=_bg_train, daemon=True, name="model_trainer").start()
    scanner.force_scan_now()
    return True


_init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conf_color(c: float) -> str:
    if c >= 90: return "#00ff88"
    if c >= 75: return "#4ade80"
    if c >= 60: return "#ffd60a"
    if c >= 45: return "#fb923c"
    return "#ff4560"

def _dir_emoji(d: str) -> str:
    return "🟢" if d == "bullish" else "🔴" if d == "bearish" else "⚪"

def _fmt_pct(p: float) -> str:
    return f"+{p:.2f}%" if p >= 0 else f"{p:.2f}%"

def _bar(score: float, color: str) -> str:
    return (f'<div style="background:#1c2840;border-radius:4px;height:6px;overflow:hidden;margin-top:4px;">'
            f'<div style="width:{score}%;height:100%;background:{color};border-radius:4px;box-shadow:0 0 6px {color}66;"></div></div>')

def _badges(signals: list[dict]) -> str:
    out = ""
    for s in signals:
        cls = "sig-bull" if s["type"] == "bullish" else "sig-bear" if s["type"] == "bearish" else "sig-neut"
        out += f'<span class="{cls}">{s["name"]}</span>'
    return out


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ BreakoutAI")
    st.markdown("---")

    running   = scanner.is_running()
    last_scan = scanner.get_last_scan_time()
    sc        = "#00ff88" if running else "#ff4560"
    st.markdown(
        f'<span class="live-dot"></span> Scanner: <strong style="color:{sc};">{"Running" if running else "Stopped"}</strong>',
        unsafe_allow_html=True,
    )
    if last_scan:
        st.caption(f"Last scan: {last_scan.strftime('%H:%M:%S UTC')}")

    st.markdown("---")
    st.markdown("### 📋 Watchlist")
    watchlist = db.get_watchlist()
    st.caption(f"{len(watchlist)} symbols monitored")

    add_sym = st.text_input("Add ticker", placeholder="e.g. NVDA", key="add_sym").upper().strip()
    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ Add", use_container_width=True) and add_sym:
            db.add_to_watchlist(add_sym)
            st.rerun()
    with c2:
        if st.button("🔄 Scan Now", use_container_width=True):
            scanner.force_scan_now()
            st.toast("Scan triggered!", icon="🔍")

    if watchlist:
        rm = st.selectbox("Remove", ["—"] + watchlist, key="remove_sym")
        if rm != "—" and st.button("🗑️ Remove", use_container_width=True):
            db.remove_from_watchlist(rm)
            st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.caption(f"Alert threshold: **{config.ALERT_THRESHOLD:.0f}%** confidence")
    st.caption(f"Scan interval: **{config.SCAN_INTERVAL_MINUTES} min**")

    if config.EMAIL_CONFIGURED:
        st.success("✅ Email alerts enabled")
        if st.button("📧 Send Test Email"):
            ok, msg = send_test_email()
            st.success(msg) if ok else st.error(f"Failed: {msg}")
    else:
        st.warning("⚠️ Email not configured")
        with st.expander("How to enable email alerts"):
            st.markdown("""
**Streamlit Cloud** → App Settings → Secrets, add:
```toml
ALERT_EMAIL_FROM = "you@gmail.com"
ALERT_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"
ALERT_EMAIL_TO = "you@gmail.com"
```
Use a **Gmail App Password** (not your normal password).  
Generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)

**Locally** → create `.streamlit/secrets.toml` with the same keys.
            """)

    st.markdown("---")
    model_status = "✅ Trained" if is_model_trained() else "⏳ Training…"
    st.caption(f"ML model: {model_status}")
    st.caption("BreakoutAI · Educational use only")


# ── Tabs ──────────────────────────────────────────────────────────────────────

t1, t2, t3, t4 = st.tabs([
    "🔍 Live Scanner", "🚨 Alert History", "📈 Chart", "🤖 Model",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
with t1:
    results   = scanner.get_latest_results()
    high_conf = [r for r in results if r["confidence"] >= config.ALERT_THRESHOLD]
    bullish   = [r for r in results if r["direction"] == "bullish"]
    top       = results[0] if results else None

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Symbols</div>'
                    f'<div class="metric-value" style="color:#00b4d8;">{len(results)}</div></div>',
                    unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">≥{config.ALERT_THRESHOLD:.0f}% Confidence</div>'
                    f'<div class="metric-value" style="color:#00ff88;">{len(high_conf)}</div></div>',
                    unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Bullish Setups</div>'
                    f'<div class="metric-value" style="color:#4ade80;">{len(bullish)}</div></div>',
                    unsafe_allow_html=True)
    with k4:
        tc = f"{top['confidence']:.1f}%" if top else "—"
        ts = top["symbol"] if top else "—"
        st.markdown(f'<div class="metric-card"><div class="metric-label">Top Confidence</div>'
                    f'<div class="metric-value" style="color:#00ff88;">{tc}</div>'
                    f'<div style="font-size:0.75rem;color:#64748b;margin-top:2px;">{ts}</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # High-confidence alert cards
    if high_conf:
        st.markdown(f"### 🚨 High-Confidence Breakouts ≥ {config.ALERT_THRESHOLD:.0f}%")
        for r in sorted(high_conf, key=lambda x: x["confidence"], reverse=True):
            cls   = "alert-card" if r["direction"] == "bullish" else "alert-card-bear"
            col   = _conf_color(r["confidence"])
            cc    = "#00ff88" if r["change_pct"] >= 0 else "#ff4560"
            cats  = "".join(f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:3px;">▸ {c}</div>' for c in r["catalysts"][:3])
            st.markdown(f"""
            <div class="{cls}">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div>
                  <span style="font-size:1.3rem;font-weight:800;color:white;font-family:monospace;">{r['symbol']}</span>
                  <span style="font-size:0.8rem;color:#64748b;margin-left:8px;">{r.get('name','')}</span>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:1.6rem;font-weight:800;color:{col};font-family:monospace;">{r['confidence']:.1f}%</div>
                  <div style="font-size:0.7rem;color:#64748b;letter-spacing:0.1em;">ML CONFIDENCE</div>
                </div>
              </div>
              <div style="margin-top:8px;display:flex;gap:20px;flex-wrap:wrap;">
                <div><span style="color:#64748b;font-size:0.75rem;">PRICE </span><span style="color:white;font-weight:700;font-family:monospace;">${r['price']:.2f}</span></div>
                <div><span style="color:#64748b;font-size:0.75rem;">TODAY </span><span style="color:{cc};font-weight:700;">{_fmt_pct(r['change_pct'])}</span></div>
                <div><span style="color:#64748b;font-size:0.75rem;">SCORE </span><span style="color:#00b4d8;font-weight:700;">{r['final_score']:.0f}/100</span></div>
                <div><span style="color:#64748b;font-size:0.75rem;">RSI </span><span style="color:white;font-family:monospace;">{r['rsi']:.1f}</span></div>
                <div><span style="color:#64748b;font-size:0.75rem;">VOL </span><span style="color:white;font-family:monospace;">{r['vol_ratio']:.2f}x</span></div>
              </div>
              <div style="margin-top:8px;">{_badges(r['signals'])}</div>
              {('<div style="margin-top:8px;">' + cats + '</div>') if r['catalysts'] else ''}
            </div>
            """, unsafe_allow_html=True)
        st.markdown("---")

    # Full table
    if not results:
        st.info("🔄 Scanner warming up — first scan triggers automatically. Click **Scan Now** in the sidebar.")
        if st.button("⚡ Trigger Scan Now"):
            scanner.force_scan_now()
            with st.spinner("Scanning… (~30s per batch)"):
                time.sleep(10)
            st.rerun()
    else:
        st.markdown("### 📊 All Symbols")
        fdir = st.selectbox("Filter", ["All", "Bullish only", "Bearish only"], key="fdir")
        rows = results
        if fdir == "Bullish only":  rows = [r for r in results if r["direction"] == "bullish"]
        elif fdir == "Bearish only": rows = [r for r in results if r["direction"] == "bearish"]

        for r in rows:
            col  = _conf_color(r["confidence"])
            cc   = "#00ff88" if r["change_pct"] >= 0 else "#ff4560"
            sqz  = '<span class="sig-bull">🔥 SQUEEZE</span>' if r.get("bb_squeeze") else ""
            st.markdown(f"""
            <div style="background:#0f1629;border:1px solid #1c2840;border-radius:10px;
                        padding:12px 16px;margin-bottom:8px;border-left:3px solid {col};">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">
                <div style="display:flex;align-items:center;gap:12px;">
                  <span style="font-size:1.05rem;font-weight:800;color:white;font-family:monospace;">{_dir_emoji(r['direction'])} {r['symbol']}</span>
                  <span style="font-size:0.8rem;font-weight:700;color:{col};">{r['confidence']:.1f}%</span>
                  {_bar(r['confidence'], col)}
                </div>
                <div style="display:flex;gap:16px;align-items:center;">
                  <span style="color:white;font-family:monospace;">${r['price']:.2f}</span>
                  <span style="color:{cc};font-weight:600;">{_fmt_pct(r['change_pct'])}</span>
                  <span style="color:#64748b;font-size:0.78rem;">RSI {r['rsi']:.0f} · Vol {r['vol_ratio']:.1f}x</span>
                </div>
              </div>
              <div style="margin-top:6px;">{_badges(r['signals'][:4])}{sqz}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    ar_col, ref_col = st.columns([3, 1])
    with ar_col:
        auto = st.checkbox("Auto-refresh every 60 s", value=False)
    with ref_col:
        if st.button("🔄 Refresh"):
            st.rerun()
    if auto:
        time.sleep(60)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ALERT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
with t2:
    st.markdown("### 🚨 Alert History")
    st.caption(f"Fired when ML confidence ≥ {config.ALERT_THRESHOLD:.0f}% · email de-duped per 4 h")

    alerts = db.get_alerts(limit=200)
    if not alerts:
        st.info("No alerts yet. The scanner is running and will notify you when a high-confidence setup is detected.")
    else:
        a1, a2, a3 = st.columns(3)
        a1.metric("Total Alerts", len(alerts))
        a2.metric("Emails Sent", sum(1 for a in alerts if a["email_sent"]))
        a3.metric("Avg Confidence", f"{sum(a['confidence'] for a in alerts)/len(alerts):.1f}%")
        st.markdown("---")

        for alert in alerts:
            ts  = alert["triggered_at"][:19].replace("T", " ")
            ei  = "📧" if alert["email_sent"] else "🔕"
            with st.expander(f"{_dir_emoji(alert['direction'])} **{alert['symbol']}** — {alert['confidence']:.1f}% · {ts} {ei}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Price", f"${alert['price']:.2f}")
                c2.metric("Change", _fmt_pct(alert["change_pct"]))
                c3.metric("Score", f"{alert['score']:.0f}/100")
                if alert["catalysts"]:
                    st.markdown("**Catalysts:**")
                    for c in alert["catalysts"]:
                        st.markdown(f"- {c}")
                st.caption(f"Direction: {alert['direction'].upper()} · Email: {'sent' if alert['email_sent'] else 'not sent'}")

        st.markdown("---")
        df_exp = pd.DataFrame(alerts)
        df_exp["catalysts"] = df_exp["catalysts"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) else x
        )
        st.download_button(
            "📥 Export to CSV", df_exp.to_csv(index=False),
            file_name="breakout_alerts.csv", mime="text/csv",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CHART
# ═══════════════════════════════════════════════════════════════════════════════
with t3:
    st.markdown("### 📈 Price Chart + Indicators")

    live = scanner.get_latest_results()
    opts = [r["symbol"] for r in live] if live else list(COMPANY_NAMES.keys())[:20]

    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sym = st.selectbox("Symbol", opts, key="chart_sym")
    with sc2:
        per = st.selectbox("Period", ["3mo", "6mo", "1y", "2y"], index=1, key="chart_per")

    if sym:
        with st.spinner(f"Loading {sym}…"):
            try:
                df = get_price_data(sym, period=per).reset_index()
                df.columns = [c.lower() for c in df.columns]
                dc = df.columns[0]   # date or datetime
                close = df["close"]
                df["sma20"]   = close.rolling(20).mean()
                df["sma50"]   = close.rolling(50).mean()
                df["vol_avg"] = df["volume"].rolling(20).mean()

                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df[dc], open=df["open"], high=df["high"],
                    low=df["low"], close=close, name=sym,
                    increasing_line_color="#00ff88", decreasing_line_color="#ff4560",
                    increasing_fillcolor="#00ff8830", decreasing_fillcolor="#ff456030",
                ))
                fig.add_trace(go.Scatter(x=df[dc], y=df["sma20"], name="SMA 20",
                                          line=dict(color="#00b4d8", width=1.5)))
                fig.add_trace(go.Scatter(x=df[dc], y=df["sma50"], name="SMA 50",
                                          line=dict(color="#ffd60a", width=1.5, dash="dot")))

                res = next((r for r in live if r["symbol"] == sym), None)
                if res and res["confidence"] >= config.ALERT_THRESHOLD:
                    fig.add_annotation(
                        x=df[dc].iloc[-1], y=float(close.iloc[-1]),
                        text=f"⚡ {res['confidence']:.0f}%",
                        showarrow=True, arrowhead=2,
                        arrowcolor="#00ff88",
                        font=dict(color="#00ff88", size=13, family="monospace"),
                        bgcolor="#00ff8815", bordercolor="#00ff8840",
                    )

                fig.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
                    font=dict(color="#94a3b8"), height=420,
                    xaxis=dict(gridcolor="#1c2840", rangeslider_visible=False),
                    yaxis=dict(gridcolor="#1c2840", tickprefix="$"),
                    legend=dict(bgcolor="#0f1629", bordercolor="#1c2840"),
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Volume bars
                vc = ["#00ff88" if c >= o else "#ff4560" for c, o in zip(df["close"], df["open"])]
                fv = go.Figure()
                fv.add_trace(go.Bar(x=df[dc], y=df["volume"], marker_color=vc, opacity=0.7, name="Volume"))
                fv.add_trace(go.Scatter(x=df[dc], y=df["vol_avg"], name="20d avg",
                                         line=dict(color="#ffd60a", width=1.5)))
                fv.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#0f1629",
                    font=dict(color="#94a3b8"), height=150,
                    xaxis=dict(gridcolor="#1c2840"),
                    yaxis=dict(gridcolor="#1c2840"),
                    margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                )
                st.plotly_chart(fv, use_container_width=True)

                if res:
                    st.markdown("---")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("ML Confidence",   f"{res['confidence']:.1f}%")
                    m2.metric("Breakout Score",  f"{res['final_score']:.0f}/100")
                    m3.metric("RSI",             f"{res['rsi']:.1f}")
                    m4.metric("Volume Ratio",    f"{res['vol_ratio']:.2f}x")
                    m5.metric("Direction",       res["direction"].upper())
                    if res["catalysts"]:
                        for c in res["catalysts"]:
                            st.markdown(f"- {c}")

            except Exception as e:
                st.error(f"Chart error for {sym}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
with t4:
    st.markdown("### 🤖 ML Model")

    col_a, col_b = st.columns(2)
    with col_a:
        status = "✅ Trained & active" if is_model_trained() else "⏳ Training in background…"
        st.markdown(f"""
**Status:** {status}

**Algorithm:** Gradient Boosting + Platt Calibration  
**Training universe:** 27 stocks × 2 years daily bars  
**Breakout label:** +5% price move within 5 trading days  
**18 features:** RSI + slope, MACD histogram + slope,  
BB squeeze ratio + %B, Volume ratio + 5d ratio,  
Price vs SMA 20/50/200 + SMA slope, ATR %,  
5d & 20d returns, distance from 52-week high,  
OBV slope, BB width  
**Validation:** 3-fold CV ROC-AUC (logged on training)
        """)

    with col_b:
        st.markdown("""
**What "95% confidence" means:**

The model is *calibrated* — a 95% output means ~95% of historical
similar setups produced a +5% breakout within 5 days.

Calibration is done with Platt Scaling (sigmoid fit on held-out
CV predictions), which corrects raw overconfident GBM outputs.

**Final confidence blend:**  
60% ML probability + 40% rule-based score  
(RSI, MACD, volume, trend alignment)

**Alert fires when ALL of:**  
✅ Confidence ≥ threshold (default 95%)  
✅ Direction is bullish  
✅ No alert for same symbol in last 4 hours
        """)

        st.markdown("---")
        from ml_model import TRAINING_SYMBOLS
        st.caption("Training symbols: " + ", ".join(TRAINING_SYMBOLS))

        if st.button("🔁 Retrain Model (~3 min)"):
            with st.spinner("Training… please wait"):
                try:
                    train_model(force=True)
                    st.success("Model retrained!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer:** BreakoutAI is for educational and research purposes only. "
        "ML confidence scores do not guarantee profitable trades. Markets are unpredictable. "
        "Always conduct your own due diligence before making any investment decisions."
    )
