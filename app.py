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
from plotly.subplots import make_subplots

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BreakoutAI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ── */
.stApp { background: #050507; font-family: 'Inter', -apple-system, sans-serif; }
[data-testid="stSidebar"] {
    background: #0A0A0D;
    border-right: 1px solid rgba(255,255,255,0.05);
}
.block-container { padding: 0 2rem 2rem; max-width: 1280px; }
h1,h2,h3,h4 { font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }

/* ── Page header strip ── */
.page-header {
    background: linear-gradient(135deg, #0A0A0D 0%, #0d0d12 100%);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 18px 0 16px;
    margin-bottom: 24px;
}
.page-header-title {
    font-size: 1.6rem; font-weight: 800; color: #fff;
    letter-spacing: -0.03em; line-height: 1;
}
.page-header-sub { font-size: 0.78rem; color: rgba(255,255,255,0.4); margin-top: 4px; }

/* ── KPI cards ── */
.metric-card {
    background: #0C0C10;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 16px;
    text-align: center;
    transition: background 0.2s, border-color 0.2s;
    height: 100%;
}
.metric-card:hover { background: #141418; border-color: rgba(255,255,255,0.10); }
.metric-value {
    font-size: 1.9rem; font-weight: 800;
    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
    letter-spacing: -0.03em; line-height: 1.1;
}
.metric-label {
    font-size: 0.65rem; color: rgba(255,255,255,0.38); font-weight: 600;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;
}
.metric-delta {
    font-size: 0.7rem; color: rgba(255,255,255,0.4); margin-top: 4px;
}

/* ── Section header ── */
.section-header {
    display: flex; align-items: center; gap: 10px;
    padding: 20px 0 12px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
    margin-bottom: 16px;
}
.section-title { font-size: 0.95rem; font-weight: 700; color: #fff; }
.section-count {
    font-size: 0.72rem; color: rgba(255,255,255,0.4);
    background: rgba(255,255,255,0.06); border-radius: 20px;
    padding: 2px 9px; font-weight: 500;
}

/* ── Alert / breakout cards ── */
.alert-card {
    background: #0C0C10;
    border: 1px solid rgba(0,200,5,0.2);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 10px;
    transition: background 0.2s, border-color 0.2s;
}
.alert-card:hover { background: #101014; border-color: rgba(0,200,5,0.35); }
.alert-card-bear {
    background: #0C0C10;
    border: 1px solid rgba(242,54,69,0.2);
    border-radius: 14px; padding: 18px 20px; margin-bottom: 10px;
    transition: background 0.2s, border-color 0.2s;
}
.alert-card-bear:hover { background: #101014; border-color: rgba(242,54,69,0.35); }

/* ── Signal badges ── */
.sig-bull {
    background: rgba(0,200,5,0.1); color: #00C805;
    border-radius: 6px; padding: 3px 9px; font-size: 0.68rem; font-weight: 600;
    display: inline-block; margin: 2px; border: 1px solid rgba(0,200,5,0.2);
}
.sig-bear {
    background: rgba(242,54,69,0.1); color: #F23645;
    border-radius: 6px; padding: 3px 9px; font-size: 0.68rem; font-weight: 600;
    display: inline-block; margin: 2px; border: 1px solid rgba(242,54,69,0.2);
}
.sig-neut {
    background: rgba(255,255,255,0.05); color: rgba(255,255,255,0.45);
    border-radius: 6px; padding: 3px 9px; font-size: 0.68rem;
    display: inline-block; margin: 2px; border: 1px solid rgba(255,255,255,0.08);
}

/* ── Live dot ── */
.live-dot {
    width: 7px; height: 7px; background: #00C805; border-radius: 50%;
    display: inline-block; margin-right: 7px;
    box-shadow: 0 0 6px rgba(0,200,5,0.7);
    animation: pulse-dot 2s infinite;
}
.idle-dot {
    width: 7px; height: 7px; background: rgba(255,255,255,0.3); border-radius: 50%;
    display: inline-block; margin-right: 7px;
}
@keyframes pulse-dot { 0%,100%{opacity:1;box-shadow:0 0 6px rgba(0,200,5,0.7);}
                        50%{opacity:0.5;box-shadow:0 0 3px rgba(0,200,5,0.3);} }

/* ── Progress bar ── */
.progress-bar-outer {
    background: rgba(255,255,255,0.07); border-radius: 8px;
    height: 5px; overflow: hidden; margin: 8px 0;
}
.progress-bar-inner { height: 100%; border-radius: 8px; background: #00C805; transition: width 0.6s ease; }
@keyframes shimmer {
    0%{background-position:-200% center;}
    100%{background-position:200% center;}
}
.progress-shimmer {
    height: 100%; border-radius: 8px;
    background: linear-gradient(90deg,#00C805 25%,#22c55e 50%,#00C805 75%);
    background-size: 200% auto;
    animation: shimmer 1.8s linear infinite;
}

/* ── Stock table rows ── */
.stock-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 20px; border-bottom: 1px solid rgba(255,255,255,0.05);
    transition: background 0.15s;
}
.stock-row:hover { background: rgba(255,255,255,0.025); }
.stock-row:last-child { border-bottom: none; }
.stock-symbol { font-size: 1rem; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.stock-name { font-size: 0.75rem; color: rgba(255,255,255,0.38); margin-top: 1px; }
.stock-price { font-size: 0.95rem; font-weight: 600; color: #fff; text-align: right; }
.stock-change { font-size: 0.9rem; font-weight: 600; text-align: right; }
.stock-conf { font-size: 0.82rem; font-weight: 700; }

/* ── Expiry lane ── */
.expiry-lane {
    border-radius: 12px; padding: 16px 18px; margin-bottom: 8px;
    transition: opacity 0.2s;
}
.expiry-lane:hover { opacity: 0.92; }

/* ── Heatmap cell ── */
.heat-cell {
    border-radius: 10px; padding: 10px 8px; text-align: center;
    transition: transform 0.15s;
    cursor: default;
}
.heat-cell:hover { transform: scale(1.04); }

/* ── Alert timeline row ── */
.alert-row {
    display: flex; align-items: center; gap: 14px;
    padding: 12px 16px; border-radius: 10px; margin-bottom: 6px;
    background: #0C0C10; border: 1px solid rgba(255,255,255,0.05);
    transition: background 0.15s;
}
.alert-row:hover { background: #101014; }

/* ── Sub-tabs (direction) ── */
.dir-tab-active {
    background: rgba(0,200,5,0.12); color: #00C805;
    border: 1px solid rgba(0,200,5,0.3); border-radius: 20px;
    padding: 5px 16px; font-size: 0.8rem; font-weight: 600; cursor: pointer;
}
.dir-tab-inactive {
    background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.08); border-radius: 20px;
    padding: 5px 16px; font-size: 0.8rem; font-weight: 500; cursor: pointer;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid rgba(255,255,255,0.07);
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    padding: 12px 22px; font-weight: 500; font-size: 0.88rem;
    color: rgba(255,255,255,0.5); border-radius: 0;
}
.stTabs [aria-selected="true"] {
    border-bottom: 2px solid #00C805 !important;
    color: #fff !important;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px; font-weight: 600; font-size: 0.88rem;
    transition: all 0.15s;
}
.stButton > button:hover { transform: translateY(-1px); }

/* ── Sidebar stat pill ── */
.stat-pill {
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,0.04); border-radius: 8px;
    padding: 8px 12px; margin-bottom: 6px;
    font-size: 0.82rem;
}
.stat-pill-label { color: rgba(255,255,255,0.45); }
.stat-pill-val { color: #fff; font-weight: 600; }

/* ── Trade plan box ── */
.trade-plan {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px; padding: 10px 14px; margin-top: 10px;
    font-size: 0.79rem; display: flex; gap: 16px; flex-wrap: wrap;
    align-items: center;
}

/* ── Direction badge ── */
.dir-bull {
    background: rgba(0,200,5,0.1); color: #00C805;
    border-radius: 5px; padding: 2px 8px; font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.dir-bear {
    background: rgba(242,54,69,0.1); color: #F23645;
    border-radius: 5px; padding: 2px 8px; font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
}
.dir-neut {
    background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.5);
    border-radius: 5px; padding: 2px 8px; font-size: 0.67rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* Selectbox / inputs */
.stSelectbox > div > div, .stTextInput > div > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── One-time init ─────────────────────────────────────────────────────────────

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
    if c >= 90: return "#00C805"
    if c >= 75: return "#22c55e"
    if c >= 60: return "#eab308"
    if c >= 45: return "#f97316"
    return "#F23645"

def _fp(p: float) -> str:
    return f"+{p:.2f}%" if p >= 0 else f"{p:.2f}%"

def _badges(signals: list[dict]) -> str:
    out = ""
    for s in signals[:5]:
        cls = "sig-bull" if s["type"] == "bullish" else "sig-bear" if s["type"] == "bearish" else "sig-neut"
        out += f'<span class="{cls}">{s["name"]}</span>'
    return out

def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "255,255,255"

def _dir_badge(d: str) -> str:
    if d == "bullish": return '<span class="dir-bull">Bullish</span>'
    if d == "bearish": return '<span class="dir-bear">Bearish</span>'
    return '<span class="dir-neut">Neutral</span>'

def _trade_plan_html(r: dict) -> str:
    """Full precision entry / exit panel embedded in a breakout card."""
    entry = r.get("entry") or r.get("price")
    sl    = r.get("stop_loss")
    if entry is None or sl is None:
        return ""

    is_bull     = r.get("direction", "bullish") == "bullish"
    tp1         = r.get("take_profit_1") or r.get("tp1")
    tp2         = r.get("take_profit_2") or r.get("tp2")
    tp3         = r.get("take_profit_3") or r.get("tp3")
    rr1         = r.get("rr1", 1.5)
    rr2         = r.get("rr2", 2.5)
    rr3         = r.get("rr3", 4.0)
    risk        = r.get("risk_per_share") or abs(entry - sl)
    rpct        = r.get("risk_pct", round(risk / (entry + 1e-9) * 100, 2))
    pos         = r.get("position_pct", 2.0)
    be          = r.get("breakeven")
    er          = r.get("entry_reason", "")
    sr          = r.get("stop_reason", "")
    vwap        = r.get("vwap")
    arrow       = "▲" if is_bull else "▼"
    entry_color = "#00C805" if is_bull else "#F23645"
    tp1_str     = f"${tp1:.2f}"  if tp1  else "—"
    tp2_str     = f"${tp2:.2f}"  if tp2  else "—"
    tp3_str     = f"${tp3:.2f}"  if tp3  else "—"
    be_str      = f"${be:.2f}"   if be   else "—"
    vwap_str    = f"${vwap:.2f}" if vwap else "—"

    def cell(label: str, val: str, color: str, sub: str = "") -> str:
        sub_html = (
            f'<div style="font-size:0.58rem;color:rgba(255,255,255,0.28);'
            f'margin-top:2px;line-height:1.3;">{sub}</div>'
        ) if sub else ""
        return (
            f'<div style="display:flex;flex-direction:column;gap:1px;">'
            f'<div style="font-size:0.58rem;color:rgba(255,255,255,0.32);'
            f'letter-spacing:0.07em;text-transform:uppercase;">{label}</div>'
            f'<div style="font-size:0.9rem;font-weight:800;color:{color};">{val}</div>'
            + sub_html +
            f'</div>'
        )

    sep  = '<div style="padding:11px 13px;border-right:1px solid rgba(255,255,255,0.05);">'
    sepl = '<div style="padding:11px 13px;">'

    header = (
        '<div style="margin-top:14px;border-top:1px solid rgba(255,255,255,0.06);padding-top:14px;">'
        f'<div style="font-size:0.62rem;font-weight:700;color:rgba(255,255,255,0.35);'
        f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;">{arrow} Trade Plan</div>'
    )

    row1 = (
        '<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0;'
        'background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.06);'
        'border-radius:10px 10px 0 0;overflow:hidden;">'
        + sep  + cell("Entry",      f"${entry:.2f}",              entry_color,             er[:40] if er else "") + "</div>"
        + sep  + cell("Stop Loss",  f"${sl:.2f}",                 "#F23645",               sr[:40] if sr else "") + "</div>"
        + sep  + cell("Breakeven",  be_str,                       "rgba(255,255,255,0.6)")                        + "</div>"
        + sep  + cell("VWAP",       vwap_str,                     "rgba(255,255,255,0.5)")                        + "</div>"
        + sepl + cell("Risk/Share", f"${risk:.2f} ({rpct:.1f}%)", "#eab308")                                      + "</div>"
        + "</div>"
    )

    row2 = (
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0;'
        'background:rgba(0,200,5,0.03);border:1px solid rgba(255,255,255,0.06);'
        'border-top:none;border-radius:0 0 10px 10px;overflow:hidden;">'
        + sep  + cell("TP 1",          tp1_str, "#00C805",             f"R:R 1:{rr1} — 50% exit") + "</div>"
        + sep  + cell("TP 2",          tp2_str, "#00C805",             f"R:R 1:{rr2} — 30% exit") + "</div>"
        + sep  + cell("TP 3 (runner)", tp3_str, "#22c55e",             f"R:R 1:{rr3} — 20% exit") + "</div>"
        + sepl + cell("Position Size", f"{pos:.1f}% of portfolio", "rgba(255,255,255,0.6)", "1% portfolio risk rule") + "</div>"
        + "</div>"
    )

    return header + row1 + row2 + "</div>"

def _stat_cell(label: str, value: str, color: str = "#fff") -> str:
    return (
        f'<div style="display:flex;flex-direction:column;gap:3px;">'
        f'  <div style="font-size:0.6rem;color:rgba(255,255,255,0.35);'
        f'letter-spacing:0.08em;text-transform:uppercase;">{label}</div>'
        f'  <div style="font-size:0.92rem;font-weight:700;color:{color};">{value}</div>'
        f'</div>'
    )


def _option_play_html(r: dict) -> str:
    """Render a compact option play recommendation chip + detail rows."""
    strategy = r.get("option_strategy")
    contract = r.get("option_contract")
    if not strategy or not contract:
        return ""

    dte        = r.get("option_dte", "")
    rationale  = r.get("option_rationale", "")
    max_profit = r.get("option_max_profit", "—")
    max_loss   = r.get("option_max_loss", "—")
    iv_est     = r.get("iv_estimate")
    iv_str     = f"{iv_est:.0f}%" if iv_est else "—"
    contract_iv = r.get("contract_iv")
    if contract_iv is not None:
        iv_str = f"{contract_iv:.0f}%"

    if isinstance(dte, int) and dte >= 0:
        dte_label = f"{dte} DTE"
        chain_badge = (
            '<span style="font-size:0.55rem;background:rgba(0,200,5,0.12);'
            'border:1px solid rgba(0,200,5,0.3);color:#00C805;padding:1px 6px;'
            'border-radius:10px;margin-left:4px;">live chain</span>'
        )
    else:
        dte_label   = ""
        chain_badge = ""

    strat_colors: dict = {
        "Long Call":         "#00C805",
        "Long Put":          "#F23645",
        "Call Debit Spread": "#22c55e",
        "Put Debit Spread":  "#ef4444",
        "Straddle":          "#eab308",
    }
    chip_color = strat_colors.get(strategy, "#fff")

    # Trader-focused details: expected move, option BE, premium, liquidity, warnings
    exp_move = r.get("expected_move")
    opt_be   = r.get("option_breakeven")
    premium  = r.get("option_premium")
    cvol     = r.get("contract_volume")
    coi      = r.get("contract_oi")
    earn_w   = r.get("earnings_warning", "")
    exdiv_w  = r.get("ex_div_warning", "")

    detail_cells = []
    if exp_move is not None:
        detail_cells.append(
            f'<div style="padding:6px 8px;border-right:1px solid rgba(255,255,255,0.05);">'
            f'<div style="font-size:0.52rem;color:rgba(255,255,255,0.25);text-transform:uppercase;">Expected move (1σ)</div>'
            f'<div style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.85);">±${exp_move:.2f}</div></div>'
        )
    if opt_be is not None:
        detail_cells.append(
            f'<div style="padding:6px 8px;border-right:1px solid rgba(255,255,255,0.05);">'
            f'<div style="font-size:0.52rem;color:rgba(255,255,255,0.25);text-transform:uppercase;">Option BE</div>'
            f'<div style="font-size:0.75rem;font-weight:700;color:#0a84ff;">${opt_be:.2f}</div></div>'
        )
    if premium is not None:
        detail_cells.append(
            f'<div style="padding:6px 8px;border-right:1px solid rgba(255,255,255,0.05);">'
            f'<div style="font-size:0.52rem;color:rgba(255,255,255,0.25);text-transform:uppercase;">Premium</div>'
            f'<div style="font-size:0.75rem;font-weight:700;">${premium:.2f}</div></div>'
        )
    if (cvol is not None and cvol >= 0) or (coi is not None and coi >= 0):
        liq = []
        if cvol is not None: liq.append(f"Vol {cvol:,}")
        if coi is not None:  liq.append(f"OI {coi:,}")
        detail_cells.append(
            '<div style="padding:6px 8px;">'
            '<div style="font-size:0.52rem;color:rgba(255,255,255,0.25);text-transform:uppercase;">Liquidity</div>'
            f'<div style="font-size:0.75rem;font-weight:700;">{" · ".join(liq)}</div></div>'
        )
    details_row = ""
    if detail_cells:
        details_row = (
            '<div style="display:grid;grid-template-columns:repeat(' + str(len(detail_cells)) + ',1fr);gap:0;'
            'background:rgba(0,0,0,0.15);border:1px solid rgba(255,255,255,0.04);'
            'border-top:none;border-radius:0 0 8px 8px;overflow:hidden;">'
            + "".join(detail_cells) + "</div>"
        )
    warnings_html = ""
    if earn_w or exdiv_w:
        warns = [w for w in [earn_w, exdiv_w] if w]
        warnings_html = (
            '<div style="margin-top:6px;padding:6px 8px;background:rgba(234,179,8,0.08);'
            'border:1px solid rgba(234,179,8,0.2);border-radius:6px;font-size:0.65rem;'
            'color:#eab308;">⚠ ' + " | ".join(warns) + "</div>"
        )

    return (
        '<div style="margin-top:12px;padding-top:12px;'
        'border-top:1px solid rgba(255,255,255,0.05);">'

        '<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:8px;">'
        '<span style="font-size:0.6rem;font-weight:700;color:rgba(255,255,255,0.3);'
        'letter-spacing:0.1em;text-transform:uppercase;">Options Play</span>'
        f'<span style="background:{chip_color}22;border:1px solid {chip_color}55;'
        f'color:{chip_color};font-size:0.68rem;font-weight:700;padding:2px 8px;'
        f'border-radius:20px;">{strategy}</span>'
        f'<span style="font-family:monospace;font-size:1rem;font-weight:900;'
        f'color:#fff;letter-spacing:0.02em;">{contract}</span>'
        + (f'<span style="font-size:0.65rem;color:rgba(255,255,255,0.35);">{dte_label}</span>' if dte_label else "")
        + chain_badge +
        '</div>'

        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0;'
        'background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);'
        'border-radius:8px 8px 0 0;overflow:hidden;">'

        '<div style="padding:8px 10px;border-right:1px solid rgba(255,255,255,0.05);">'
        '<div style="font-size:0.55rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">Max Profit</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#00C805;margin-top:2px;">{max_profit}</div>'
        '</div>'

        '<div style="padding:8px 10px;border-right:1px solid rgba(255,255,255,0.05);">'
        '<div style="font-size:0.55rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">Max Loss</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#F23645;margin-top:2px;">{max_loss}</div>'
        '</div>'

        '<div style="padding:8px 10px;">'
        '<div style="font-size:0.55rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">IV</div>'
        f'<div style="font-size:0.82rem;font-weight:700;color:#eab308;margin-top:2px;">{iv_str}</div>'
        '</div>'

        '</div>'
        + details_row
        + warnings_html
        + (
            f'<div style="font-size:0.68rem;color:rgba(255,255,255,0.38);'
            f'margin-top:7px;line-height:1.45;">{rationale}</div>'
            if rationale else ""
        ) +

        '</div>'
    )


def _alt_data_html(r: dict) -> str:
    """Render Options flow, Social/Rumors, and Geopolitical when present."""
    parts = []
    flow = r.get("unusual_flow")
    if flow and flow.get("net_call_put") is not None:
        net = flow["net_call_put"]
        bias = flow.get("bias", "neutral")
        bias_col = "#00C805" if bias == "bullish" else "#F23645" if bias == "bearish" else "#888"
        parts.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;">📈 Flow</span>'
            f'<span style="font-size:0.8rem;font-weight:700;color:{bias_col};">'
            f'${net:,.0f} net {"calls" if net > 0 else "puts"}</span></div>'
        )
    social = r.get("social_sentiment")
    if social and social.get("buzz", 0) > 50:
        buzz = social["buzz"]
        overall = social.get("overall", 0.5)
        sent_col = "#00C805" if overall > 0.55 else "#F23645" if overall < 0.45 else "#eab308"
        parts.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<span style="font-size:0.6rem;color:rgba(255,255,255,0.3);text-transform:uppercase;">📊 Social</span>'
            f'<span style="font-size:0.8rem;font-weight:700;color:{sent_col};">'
            f'{buzz} mentions · {overall*100:.0f}% bullish</span></div>'
        )
    geo = r.get("geo_news") or []
    rumor = r.get("rumor_news") or []
    if geo or rumor:
        items = []
        for n in geo[:1]:
            items.append(f'🌍 {n["title"][:50]}…')
        for n in rumor[:1]:
            items.append(f'💬 {n["title"][:50]}…')
        if items:
            parts.append(
                '<div style="font-size:0.68rem;color:rgba(255,255,255,0.4);line-height:1.4;">'
                + " | ".join(items) + '</div>'
            )
    if not parts:
        return ""
    return (
        '<div style="margin-top:10px;padding:10px 12px;background:rgba(255,255,255,0.02);'
        'border:1px solid rgba(255,255,255,0.06);border-radius:8px;">'
        '<div style="font-size:0.58rem;color:rgba(255,255,255,0.28);letter-spacing:0.08em;'
        'text-transform:uppercase;margin-bottom:8px;">Flow · Rumors · Geo</div>'
        + "".join(parts) +
        '</div>'
    )


def _conf_bar(confidence: float, color: str) -> str:
    pct = min(100, confidence)
    return (
        f'<div style="background:rgba(255,255,255,0.07);border-radius:4px;'
        f'height:3px;margin-top:6px;width:100%;">'
        f'  <div style="width:{pct}%;height:100%;background:{color};'
        f'border-radius:4px;transition:width 0.4s ease;"></div>'
        f'</div>'
    )


def _breakout_card_html(r: dict) -> str:
    """Return the HTML string for a single breakout card (no st.markdown call)."""
    is_bull      = r["direction"] == "bullish"
    col          = _cc(r["confidence"])
    cc           = "#00C805" if r["change_pct"] >= 0 else "#F23645"
    border_color = "rgba(0,200,5,0.22)" if is_bull else "rgba(242,54,69,0.22)"
    glow_color   = "rgba(0,200,5,0.04)"  if is_bull else "rgba(242,54,69,0.04)"
    cat_label    = "Catalysts" if is_bull else "Risks"
    rsi_color    = "#F23645" if r["rsi"] > 70 else "#00C805" if r["rsi"] < 35 else "#fff"
    vol_color    = "#eab308" if r["vol_ratio"] >= 2 else "#fff"

    cats_html = "".join(
        f'<div style="font-size:0.73rem;color:rgba(255,255,255,0.5);'
        f'display:flex;align-items:flex-start;gap:5px;margin-top:4px;">'
        f'<span style="color:{col};margin-top:1px;">▸</span>{c}</div>'
        for c in r["catalysts"][:6]
    )
    cats_section = (
        f'<div style="margin-top:10px;padding-top:10px;'
        f'border-top:1px solid rgba(255,255,255,0.05);">'
        f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.3);'
        f'letter-spacing:0.07em;text-transform:uppercase;margin-bottom:4px;">{cat_label}</div>'
        + cats_html + '</div>'
    ) if r["catalysts"] else ""

    # Alternative data panel: Options flow, Social/Rumors, Geopolitical
    alt_html = _alt_data_html(r)

    sep = '<div style="padding:10px 12px;border-right:1px solid rgba(255,255,255,0.05);">'

    return (
        f'<div style="background:#0C0C10;border:1px solid {border_color};border-radius:16px;'
        f'padding:20px 22px;box-shadow:0 0 24px {glow_color};">'

        f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px;">'
        f'<div style="min-width:0;">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<span style="font-size:1.25rem;font-weight:800;color:#fff;letter-spacing:-0.01em;">{r["symbol"]}</span>'
        f'{_dir_badge(r["direction"])}'
        f'</div>'
        f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.38);margin-top:3px;">{r.get("name","")[:36]}</div>'
        f'</div>'
        f'<div style="text-align:right;flex-shrink:0;">'
        f'<div style="font-size:1.7rem;font-weight:800;color:{col};letter-spacing:-0.02em;line-height:1;">{r["confidence"]:.1f}%</div>'
        f'<div style="font-size:0.58rem;color:rgba(255,255,255,0.3);letter-spacing:0.1em;text-transform:uppercase;margin-top:2px;">Confidence</div>'
        + _conf_bar(r["confidence"], col) +
        f'</div>'
        f'</div>'

        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin-top:16px;'
        f'background:rgba(255,255,255,0.03);border-radius:10px;'
        f'border:1px solid rgba(255,255,255,0.05);overflow:hidden;">'
        + sep + _stat_cell("Price",  f'${r["price"]:.2f}')                     + "</div>"
        + sep + _stat_cell("Today",  _fp(r["change_pct"]),   cc)                + "</div>"
        + sep + _stat_cell("Score",  f'{r["final_score"]:.0f}/100', col)        + "</div>"
        + sep + _stat_cell("RSI",    f'{r["rsi"]:.1f}',      rsi_color)         + "</div>"
        + f'<div style="padding:10px 12px;">'
        + _stat_cell("Volume", f'{r["vol_ratio"]:.1f}x', vol_color) + "</div>"
        + f'</div>'

        + _trade_plan_html(r)
        + _option_play_html(r)
        + alt_html
        + f'<div style="margin-top:10px;">{_badges(r["signals"])}</div>'
        + cats_section
        + f'</div>'
    )


def _render_breakout_card(r: dict) -> None:
    """Render a single breakout card (used when a grid wrapper is not needed)."""
    st.markdown(_breakout_card_html(r), unsafe_allow_html=True)

def _kpi(label: str, value: str, color: str = "#fff", sub: str = "") -> str:
    sub_html = f'<div class="metric-delta">{sub}</div>' if sub else ""
    return (f'<div class="metric-card">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color};">{value}</div>'
            f'{sub_html}</div>')


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div style="padding:4px 0 16px;">'
        '<div style="font-size:1.35rem;font-weight:800;color:#fff;letter-spacing:-0.02em;">BreakoutAI</div>'
        '<div style="font-size:0.72rem;color:rgba(255,255,255,0.35);margin-top:3px;">Full-market breakout scanner</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    prog    = scanner.get_scan_progress()
    running = scanner.is_running()
    last_scan = scanner.get_last_scan_time()

    # Scanner status pill
    if prog["running"]:
        phase = prog.get("phase", "screening")
        total = prog["total"] or 0
        done  = prog["done"]
        pct   = int(done / total * 100) if total else 0
        dot   = '<span class="live-dot"></span>'
        label = "Screening…" if (phase == "screening" or total == 0) else f"Analyzing {done:,}/{total:,}"
        bar_inner = '<div class="progress-shimmer" style="width:100%;"></div>' if total == 0 else \
                    f'<div class="progress-bar-inner" style="width:{pct}%;"></div>'
        st.markdown(
            f'<div style="background:rgba(0,200,5,0.06);border:1px solid rgba(0,200,5,0.18);'
            f'border-radius:10px;padding:10px 12px;margin-bottom:12px;">'
            f'{dot}<span style="color:#00C805;font-size:0.82rem;font-weight:600;">{label}</span>'
            f'<div class="progress-bar-outer" style="margin-top:8px;">{bar_inner}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        dot = '<span class="idle-dot"></span>'
        status_color = "rgba(255,255,255,0.5)"
        last_scan_html = (
            f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.3);margin-top:4px;">'
            f'Last: {last_scan.strftime("%b %d %H:%M UTC")}</div>'
            if last_scan else ""
        )
        st.markdown(
            f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);'
            f'border-radius:10px;padding:10px 12px;margin-bottom:12px;">'
            f'{dot}<span style="color:{status_color};font-size:0.82rem;">Scanner idle</span>'
            f'{last_scan_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick stats
    _all = scanner.get_latest_results()
    _bull = sum(1 for r in _all if r["direction"] == "bullish")
    _bear = sum(1 for r in _all if r["direction"] == "bearish")
    _hi   = sum(1 for r in _all if r["confidence"] >= config.ALERT_THRESHOLD)

    st.markdown(
        f'<div class="stat-pill"><span class="stat-pill-label">Scanned</span>'
        f'<span class="stat-pill-val">{len(_all):,}</span></div>'
        f'<div class="stat-pill"><span class="stat-pill-label">Bullish</span>'
        f'<span class="stat-pill-val" style="color:#00C805;">{_bull:,}</span></div>'
        f'<div class="stat-pill"><span class="stat-pill-label">Bearish</span>'
        f'<span class="stat-pill-val" style="color:#F23645;">{_bear:,}</span></div>'
        f'<div class="stat-pill"><span class="stat-pill-label">High Conf ≥{config.ALERT_THRESHOLD:.0f}%</span>'
        f'<span class="stat-pill-val" style="color:#eab308;">{_hi:,}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div style="font-size:0.75rem;font-weight:600;color:rgba(255,255,255,0.4);'
                'letter-spacing:0.06em;text-transform:uppercase;margin-bottom:10px;">Filters</div>',
                unsafe_allow_html=True)
    min_conf        = st.slider("Min confidence %", 0, 100, 40, 5, key="min_conf")
    direction_filter = st.selectbox("Direction", ["All", "Bullish only", "Bearish only"], key="dir_filter")
    min_vol_ratio   = st.slider("Min volume ratio", 0.0, 5.0, 0.0, 0.5, key="min_vol")
    max_results     = st.select_slider("Show top N", [25, 50, 100, 200], value=50, key="top_n")

    st.markdown("---")
    col_scan, col_ref = st.columns(2)
    with col_scan:
        if st.button("⚡ Scan Now", use_container_width=True):
            scanner.force_scan_now()
            st.toast("Full-market scan triggered!", icon="🔍")
    with col_ref:
        if st.button("↺ Refresh", use_container_width=True):
            st.rerun()

    st.markdown("---")

    if config.EMAIL_CONFIGURED:
        st.success("✅ Email alerts active")
        if st.button("📧 Test Email", use_container_width=True):
            ok, msg = send_test_email()
            st.success(msg) if ok else st.error(f"Failed: {msg}")
    else:
        st.warning("⚠️ Email not configured")
        with st.expander("Setup email alerts"):
            st.markdown("""
Add to **Streamlit Secrets**:
```toml
ALERT_EMAIL_FROM     = "you@gmail.com"
ALERT_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"
ALERT_EMAIL_TO       = "you@gmail.com"
```
Use a Gmail **App Password**.
[Generate one here](https://myaccount.google.com/apppasswords)
            """)

    st.markdown("---")
    st.markdown(
        f'<div style="font-size:0.7rem;color:rgba(255,255,255,0.3);line-height:1.8;">'
        f'ML model: {"✅ trained" if is_model_trained() else "⏳ training…"}<br>'
        f'Alert threshold: {config.ALERT_THRESHOLD:.0f}%<br>'
        f'Scan interval: {config.SCAN_INTERVAL_MINUTES} min<br>'
        f'Min price: ${config.MIN_PRICE:.0f} · Min vol: {config.MIN_AVG_VOLUME//1000:,}k'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="font-size:0.65rem;color:rgba(255,255,255,0.18);margin-top:12px;">Educational use only · Not financial advice</div>',
                unsafe_allow_html=True)


# ── Auto-refresh while scanning ───────────────────────────────────────────────
_prog = scanner.get_scan_progress()
if _prog["running"]:
    time.sleep(5)
    st.rerun()


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


# ── Page header ───────────────────────────────────────────────────────────────
now_utc = datetime.now(timezone.utc)
_wday = now_utc.weekday()
_hour = now_utc.hour
market_open = _wday < 5 and 14 <= _hour < 21   # Mon–Fri 14:30–21:00 UTC ≈ NYSE
market_str  = '<span style="color:#00C805;">● Market Open</span>' if market_open else \
              '<span style="color:rgba(255,255,255,0.35);">○ Market Closed</span>'
high_conf_all = [r for r in all_results if r["confidence"] >= config.ALERT_THRESHOLD]

_last_scan = scanner.get_last_scan_time()
if _last_scan:
    _age_mins = int((now_utc - _last_scan.replace(tzinfo=timezone.utc)
                     if _last_scan.tzinfo is None else
                     now_utc - _last_scan).total_seconds() / 60)
    _as_of = _last_scan.strftime("%b %d %H:%M UTC")
    if not market_open and all_results:
        _data_note = (
            f'<span style="color:rgba(255,159,10,0.9);">⏱ Data as of {_as_of} '
            f'({_age_mins:,} min ago)</span>'
        )
    else:
        _data_note = f'<span style="color:rgba(255,255,255,0.3);">Last scan {_as_of}</span>'
else:
    _data_note = ""

st.markdown(
    f'<div class="page-header">'
    f'<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
    f'<div>'
    f'  <div class="page-header-title">BreakoutAI</div>'
    f'  <div class="page-header-sub">'
    f'    {market_str}&nbsp;&nbsp;·&nbsp;&nbsp;{now_utc.strftime("%b %d, %Y  %H:%M UTC")}'
    f'    {"&nbsp;&nbsp;·&nbsp;&nbsp;" + _data_note if _data_note else ""}'
    f'  </div>'
    f'</div>'
    f'<div style="display:flex;gap:20px;flex-wrap:wrap;">'
    f'  <div style="text-align:center;">'
    f'    <div style="font-size:1.4rem;font-weight:800;color:#fff;">{len(all_results):,}</div>'
    f'    <div style="font-size:0.62rem;color:rgba(255,255,255,0.35);letter-spacing:0.07em;text-transform:uppercase;">Scanned</div>'
    f'  </div>'
    f'  <div style="text-align:center;">'
    f'    <div style="font-size:1.4rem;font-weight:800;color:#00C805;">{sum(1 for r in all_results if r["direction"]=="bullish"):,}</div>'
    f'    <div style="font-size:0.62rem;color:rgba(255,255,255,0.35);letter-spacing:0.07em;text-transform:uppercase;">Bullish</div>'
    f'  </div>'
    f'  <div style="text-align:center;">'
    f'    <div style="font-size:1.4rem;font-weight:800;color:#F23645;">{sum(1 for r in all_results if r["direction"]=="bearish"):,}</div>'
    f'    <div style="font-size:0.62rem;color:rgba(255,255,255,0.35);letter-spacing:0.07em;text-transform:uppercase;">Bearish</div>'
    f'  </div>'
    f'  <div style="text-align:center;">'
    f'    <div style="font-size:1.4rem;font-weight:800;color:#eab308;">{len(high_conf_all):,}</div>'
    f'    <div style="font-size:0.62rem;color:rgba(255,255,255,0.35);letter-spacing:0.07em;text-transform:uppercase;">High Conf</div>'
    f'  </div>'
    f'</div>'
    f'</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ── Tabs ──────────────────────────────────────────────────────────────────────

t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊  Scanner",
    "⏱  Expiry Signals",
    "🔔  Alerts",
    "📈  Chart",
    "🌡  Heatmap",
    "🤖  Model",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ═══════════════════════════════════════════════════════════════════════════════
with t1:

    # Scan progress banner
    if prog["running"]:
        total = prog["total"] or 0
        done  = prog["done"]
        phase = prog.get("phase", "screening")
        pct   = int(done / total * 100) if total else 0
        if phase == "screening" or total == 0:
            phase_label = "Building universe & pre-screening tickers…"
            bar_inner = '<div class="progress-shimmer" style="width:100%;"></div>'
        else:
            phase_label = f"Analyzing {done:,} / {total:,} symbols"
            bar_inner = f'<div class="progress-bar-inner" style="width:{pct}%;"></div>'
        st.markdown(
            f'<div style="background:rgba(0,200,5,0.05);border:1px solid rgba(0,200,5,0.15);'
            f'border-radius:12px;padding:14px 18px;margin-bottom:20px;">'
            f'<span class="live-dot"></span>'
            f'<span style="color:rgba(255,255,255,0.8);font-size:0.88rem;font-weight:500;">{phase_label}</span>'
            f'<div class="progress-bar-outer">{bar_inner}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    elif not all_results:
        st.markdown(
            '<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);border-radius:14px;'
            'padding:40px 20px;text-align:center;margin-bottom:20px;">'
            '<div style="font-size:2rem;margin-bottom:12px;">📡</div>'
            '<div style="color:rgba(255,255,255,0.6);font-size:0.95rem;font-weight:500;">No results yet.</div>'
            '<div style="color:rgba(255,255,255,0.35);font-size:0.82rem;margin-top:6px;">'
            'Press <strong style="color:#fff;">⚡ Scan Now</strong> in the sidebar to start the full-market scan.</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # High-confidence cards
    if high_conf_all:
        st.markdown(
            f'<div class="section-header">'
            f'<span class="section-title">High-Confidence Alerts</span>'
            f'<span class="section-count">≥ {config.ALERT_THRESHOLD:.0f}% · {len(high_conf_all)} found</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        hc_bull = [r for r in high_conf_all if r["direction"] == "bullish"]
        hc_bear = [r for r in high_conf_all if r["direction"] == "bearish"]

        # Interleave bull/bear so the grid reads left-to-right naturally
        top_bull = sorted(hc_bull, key=lambda x: x["confidence"], reverse=True)[:5]
        top_bear = sorted(hc_bear, key=lambda x: x["confidence"], reverse=True)[:5]
        hc_cards_html = "".join(_breakout_card_html(r) for r in top_bull + top_bear)
        st.markdown(
            '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));'
            'gap:14px;align-items:start;">'
            + hc_cards_html +
            '</div>',
            unsafe_allow_html=True,
        )

    # Full results — sub-tabs by direction
    if all_results:
        bull_filtered = [r for r in filtered if r["direction"] == "bullish"]
        bear_filtered = [r for r in filtered if r["direction"] == "bearish"]
        neut_filtered = [r for r in filtered if r["direction"] == "neutral"]

        st.markdown(
            f'<div class="section-header">'
            f'<span class="section-title">All Results</span>'
            f'<span class="section-count">{len(filtered):,} shown · {len(all_results):,} scanned</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        sub1, sub2, sub3 = st.tabs([
            f"🟢 Bullish ({len(bull_filtered)})",
            f"🔴 Bearish ({len(bear_filtered)})",
            f"⚪ Neutral ({len(neut_filtered)})",
        ])

        def _stock_table(rows: list) -> None:
            if not rows:
                st.markdown(
                    '<div style="color:rgba(255,255,255,0.3);font-size:0.85rem;'
                    'text-align:center;padding:24px 0;">No results match current filters.</div>',
                    unsafe_allow_html=True,
                )
                return
            rows_html = ""
            for r in rows:
                col  = _cc(r["confidence"])
                cc   = "#00C805" if r["change_pct"] >= 0 else "#F23645"
                sqz  = ' <span class="sig-bull">SQZ</span>' if r.get("bb_squeeze") else ""
                badge = _badges(r["signals"][:3])
                rows_html += f"""
                <tr class="stock-row">
                  <td style="padding:14px 18px;white-space:nowrap;border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div class="stock-symbol">{r['symbol']}</div>
                    <div class="stock-name">{r.get('name','')[:28]}</div>
                  </td>
                  <td style="padding:14px 10px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div class="stock-price">${r['price']:.2f}</div>
                  </td>
                  <td style="padding:14px 10px;text-align:right;border-bottom:1px solid rgba(255,255,255,0.05);">
                    <div class="stock-change" style="color:{cc};">{_fp(r['change_pct'])}</div>
                  </td>
                  <td style="padding:14px 10px;border-bottom:1px solid rgba(255,255,255,0.05);min-width:90px;">
                    <div class="stock-conf" style="color:{col};">{r['confidence']:.1f}%</div>
                    <div style="background:rgba(255,255,255,0.07);border-radius:3px;height:3px;margin-top:5px;width:70px;">
                      <div style="width:{min(100,r['confidence'])}%;height:100%;background:{col};border-radius:3px;"></div>
                    </div>
                  </td>
                  <td style="padding:14px 10px;text-align:center;color:rgba(255,255,255,0.5);
                             font-size:0.82rem;border-bottom:1px solid rgba(255,255,255,0.05);">{r['rsi']:.0f}</td>
                  <td style="padding:14px 10px;text-align:center;color:rgba(255,255,255,0.5);
                             font-size:0.82rem;border-bottom:1px solid rgba(255,255,255,0.05);">{r['vol_ratio']:.1f}x</td>
                  <td style="padding:14px 18px;font-size:0.78rem;border-bottom:1px solid rgba(255,255,255,0.05);">{badge}{sqz}</td>
                </tr>"""

            st.markdown(f"""
            <div style="overflow-x:auto;max-height:560px;overflow-y:auto;
                        background:#0A0A0D;border:1px solid rgba(255,255,255,0.06);border-radius:14px;">
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem;">
              <thead>
                <tr style="position:sticky;top:0;background:#0A0A0D;z-index:1;
                           border-bottom:1px solid rgba(255,255,255,0.07);">
                  <th style="padding:11px 18px;text-align:left;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Symbol</th>
                  <th style="padding:11px 10px;text-align:right;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Price</th>
                  <th style="padding:11px 10px;text-align:right;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Chg%</th>
                  <th style="padding:11px 10px;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Confidence</th>
                  <th style="padding:11px 10px;text-align:center;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">RSI</th>
                  <th style="padding:11px 10px;text-align:center;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Vol</th>
                  <th style="padding:11px 18px;color:rgba(255,255,255,0.38);
                             font-weight:600;font-size:0.68rem;letter-spacing:0.07em;text-transform:uppercase;">Signals</th>
                </tr>
              </thead>
              <tbody style="background:#050507;">{rows_html}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)

        with sub1:
            _stock_table(bull_filtered)
        with sub2:
            _stock_table(bear_filtered)
        with sub3:
            _stock_table(neut_filtered)

    # Auto-refresh row
    st.markdown("<br>", unsafe_allow_html=True)
    c_ar, c_ref = st.columns([3, 1])
    with c_ar:
        auto = st.checkbox("Auto-refresh every 60 s", value=False, key="auto_refresh")
    with c_ref:
        if st.button("↺ Refresh", key="tab1_refresh"):
            st.rerun()
    if auto:
        time.sleep(60)
        st.rerun()

    # Diagnostics expander
    with st.expander("🔍 Scanner diagnostics"):
        _p = scanner.get_scan_progress()
        _model_ready = is_model_trained()
        st.markdown(f"""
| Setting | Value |
|---------|-------|
| ML model | {"✅ trained (ML scores active)" if _model_ready else "⏳ training in background (rule-based fallback active)"} |
| Scanner state | {"🟢 Running — phase: **" + _p.get("phase","?") + "**" if _p["running"] else "⚪ Idle"} |
| Results in memory | {len(all_results):,} total · {len(filtered):,} matching current filters |
| Min confidence filter | {min_conf}% (lower this if no results show) |
| Direction filter | {direction_filter} |
| Min vol ratio | {min_vol_ratio}x |
| Bullish in memory | {sum(1 for r in all_results if r["direction"]=="bullish"):,} |
| Bearish in memory | {sum(1 for r in all_results if r["direction"]=="bearish"):,} |
| Neutral in memory | {sum(1 for r in all_results if r["direction"]=="neutral"):,} |
| Confidence range | {f'{min(r["confidence"] for r in all_results):.1f}% – {max(r["confidence"] for r in all_results):.1f}%' if all_results else "—"} |
        """)
        if all_results and not filtered:
            st.warning(
                f"⚠️ There are **{len(all_results):,}** results in memory but **0** pass the current filters. "
                f"Try lowering the **Min confidence %** slider (currently {min_conf}%) or changing the direction filter."
            )
        if not _model_ready:
            st.info(
                "The ML model is still training in the background (takes 2–3 min on first run). "
                "Until then, confidence scores are rule-based and tend to cluster around 50–65%. "
                "Results will improve automatically once training completes."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPIRY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
EXPIRY_ORDER = ["0dte", "2dte", "weeklies", "monthlies", "yearly"]
EXPIRY_META = {
    "0dte":     ("0 DTE",     "Same-day · explosive setup",        "rgba(242,54,69,0.08)",   "#F23645"),
    "2dte":     ("2 DTE",     "1–2 days · imminent move",          "rgba(255,159,10,0.08)",   "#ff9f0a"),
    "weeklies": ("Weeklies",  "5–7 days · breakout play",          "rgba(0,200,5,0.08)",      "#00C805"),
    "monthlies":("Monthlies", "~30 days · swing setup",            "rgba(10,132,255,0.08)",   "#0a84ff"),
    "yearly":   ("Yearly",    "~365 days · LEAPS trend play",      "rgba(148,163,184,0.08)",  "#94a3b8"),
}

with t2:
    st.markdown(
        '<div style="padding:8px 0 20px;">'
        '<div style="font-size:0.88rem;color:rgba(255,255,255,0.45);max-width:600px;">'
        'Stocks on the verge of a breakout, grouped by recommended options expiry. '
        'Assigned based on BB squeeze strength, volume surge, catalysts, and ML confidence.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    expiry_groups: dict[str, list] = {e: [] for e in EXPIRY_ORDER}
    for r in all_results:
        if r["confidence"] < 60:
            continue
        exp = r.get("expiry_signal", "weeklies")
        if exp in expiry_groups:
            expiry_groups[exp].append(r)

    # Summary bar
    summary_html = ""
    for key in EXPIRY_ORDER:
        label, _, _, color = EXPIRY_META[key]
        cnt = len(expiry_groups[key])
        summary_html += (
            f'<div style="text-align:center;padding:12px 16px;background:rgba(255,255,255,0.03);'
            f'border:1px solid rgba(255,255,255,0.06);border-radius:10px;">'
            f'<div style="font-size:1.3rem;font-weight:800;color:{color};">{cnt}</div>'
            f'<div style="font-size:0.65rem;color:rgba(255,255,255,0.38);margin-top:2px;'
            f'letter-spacing:0.07em;text-transform:uppercase;">{label}</div>'
            f'</div>'
        )
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:24px;">'
        f'{summary_html}</div>',
        unsafe_allow_html=True,
    )

    def _expiry_card(r: dict, color: str) -> str:
        cc        = "#00C805" if r["change_pct"] >= 0 else "#F23645"
        rsi_c     = "#F23645" if r["rsi"] > 70 else "#00C805" if r["rsi"] < 35 else "rgba(255,255,255,0.7)"
        vol_color = "#eab308" if r["vol_ratio"] >= 2 else "rgba(255,255,255,0.7)"
        sqz       = '<span class="sig-bull">SQZ</span>' if r.get("bb_squeeze") else ""
        cat       = (r["catalysts"][0][:72] + "…") if r["catalysts"] else ""
        is_bull   = r.get("direction") == "bullish"
        cat_html  = (
            f'<div style="font-size:0.72rem;color:rgba(255,255,255,0.42);'
            f'margin-top:8px;line-height:1.4;">'
            f'<span style="color:{color};margin-right:4px;">▸</span>{cat}</div>'
        ) if cat else ""

        def sc(lbl: str, val: str, clr: str) -> str:
            return (
                f'<div style="display:flex;flex-direction:column;gap:2px;">'
                f'<span style="font-size:0.58rem;color:rgba(255,255,255,0.3);'
                f'letter-spacing:0.07em;text-transform:uppercase;">{lbl}</span>'
                f'<span style="font-size:0.88rem;font-weight:700;color:{clr};">{val}</span>'
                f'</div>'
            )

        # ── trade plan section ─────────────────────────────────────────────
        entry = r.get("entry") or r.get("price")
        sl    = r.get("stop_loss")
        tp1   = r.get("take_profit_1") or r.get("tp1")
        tp2   = r.get("take_profit_2") or r.get("tp2")
        tp3   = r.get("take_profit_3") or r.get("tp3")
        rr1   = r.get("rr1", 1.5)
        rr2   = r.get("rr2", 2.5)
        rr3   = r.get("rr3", 4.0)
        risk  = r.get("risk_per_share") or (abs(entry - sl) if entry and sl else None)
        rpct  = r.get("risk_pct")
        er    = r.get("entry_reason", "")
        sr    = r.get("stop_reason", "")

        entry_clr = "#00C805" if is_bull else "#F23645"
        arrow     = "▲" if is_bull else "▼"

        def tp_sc(lbl: str, val, rr, note: str) -> str:
            val_str = f"${val:.2f}" if val else "—"
            return (
                f'<div style="display:flex;flex-direction:column;gap:2px;">'
                f'<span style="font-size:0.56rem;color:rgba(255,255,255,0.28);'
                f'letter-spacing:0.07em;text-transform:uppercase;">{lbl}</span>'
                f'<span style="font-size:0.85rem;font-weight:700;color:#00C805;">{val_str}</span>'
                f'<span style="font-size:0.55rem;color:rgba(255,255,255,0.25);">{note} · R:R 1:{rr}</span>'
                f'</div>'
            )

        if entry and sl:
            risk_str  = f"${risk:.2f} ({rpct:.1f}%)" if risk and rpct else (f"${risk:.2f}" if risk else "—")
            er_short  = er[:38] if er else ""
            sr_short  = sr[:38] if sr else ""

            plan_row1 = (
                '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:2px;">'
                + f'<div style="display:flex;flex-direction:column;gap:2px;">'
                  f'<span style="font-size:0.56rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">Entry</span>'
                  f'<span style="font-size:0.92rem;font-weight:800;color:{entry_clr};">${entry:.2f}</span>'
                  + (f'<span style="font-size:0.55rem;color:rgba(255,255,255,0.25);">{er_short}</span>' if er_short else "")
                  + '</div>'
                + f'<div style="display:flex;flex-direction:column;gap:2px;">'
                  f'<span style="font-size:0.56rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">Stop Loss</span>'
                  f'<span style="font-size:0.92rem;font-weight:800;color:#F23645;">${sl:.2f}</span>'
                  + (f'<span style="font-size:0.55rem;color:rgba(255,255,255,0.25);">{sr_short}</span>' if sr_short else "")
                  + '</div>'
                + f'<div style="display:flex;flex-direction:column;gap:2px;">'
                  f'<span style="font-size:0.56rem;color:rgba(255,255,255,0.28);letter-spacing:0.07em;text-transform:uppercase;">Risk/Share</span>'
                  f'<span style="font-size:0.85rem;font-weight:700;color:#eab308;">{risk_str}</span>'
                  f'</div>'
                + '</div>'
            )

            plan_row2 = (
                '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-top:8px;">'
                + tp_sc("TP 1", tp1, rr1, "50% exit")
                + tp_sc("TP 2", tp2, rr2, "30% exit")
                + tp_sc("TP 3 runner", tp3, rr3, "20% exit")
                + '</div>'
            )

            trade_plan_html = (
                f'<div style="margin-top:10px;padding-top:10px;'
                f'border-top:1px solid rgba(255,255,255,0.05);">'
                f'<div style="font-size:0.58rem;font-weight:700;color:rgba(255,255,255,0.28);'
                f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;">'
                f'{arrow} Trade Plan</div>'
                + plan_row1 + plan_row2 +
                f'</div>'
            )
        else:
            trade_plan_html = ""

        return (
            f'<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);'
            f'border-left:3px solid {color};border-radius:12px;'
            f'padding:14px 16px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'flex-wrap:wrap;gap:10px;">'
            f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
            f'<span style="font-size:1.05rem;font-weight:800;color:#fff;">{r["symbol"]}</span>'
            f'<span style="font-size:0.68rem;color:rgba(255,255,255,0.35);">{r.get("name","")[:22]}</span>'
            f'{_dir_badge(r["direction"])}{sqz}'
            f'</div>'
            f'<span style="font-size:1.35rem;font-weight:800;color:{color};">{r["confidence"]:.1f}%</span>'
            f'</div>'
            f'<div style="display:flex;gap:16px;margin-top:10px;flex-wrap:wrap;">'
            + sc("Price",  f'${r["price"]:.2f}',          "#fff")
            + sc("Today",  _fp(r["change_pct"]),           cc)
            + sc("Volume", f'{r["vol_ratio"]:.1f}x',       vol_color)
            + sc("RSI",    f'{r["rsi"]:.0f}',              rsi_c)
            + sc("Score",  f'{r["final_score"]:.0f}/100',  "rgba(255,255,255,0.7)")
            + f'</div>'
            + cat_html
            + trade_plan_html
            + _option_play_html(r)
            + _alt_data_html(r)
            + f'<div style="margin-top:8px;">{_badges(r["signals"][:4])}</div>'
            + f'</div>'
        )

    any_results = False
    for exp_key in EXPIRY_ORDER:
        rows = expiry_groups[exp_key]
        if not rows:
            continue
        any_results = True
        label, desc, bg, color = EXPIRY_META[exp_key]
        sorted_rows = sorted(rows, key=lambda x: x["confidence"], reverse=True)[:12]

        expander_label = f"{label}  ·  {desc}  ·  {len(rows)} stocks"
        with st.expander(expander_label, expanded=(exp_key in ("0dte", "2dte", "weeklies"))):
            inner = "".join(_expiry_card(r, color) for r in sorted_rows)
            st.markdown(
                '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));'
                'gap:12px;align-items:start;">'
                + inner +
                '</div>',
                unsafe_allow_html=True,
            )

    if not any_results:
        st.info("Run a scan to populate expiry signals. Stocks are assigned to 0DTE → Yearly based on technical setup + catalysts.")

    with st.expander("How expiry signals work"):
        st.markdown("""
        | Bucket | Criteria |
        |--------|----------|
        | **0 DTE** | BB squeeze + volume ≥ 2.5x + catalyst or 90%+ confidence |
        | **2 DTE** | Squeeze or vol ≥ 2x + catalyst/momentum + 80%+ confidence |
        | **Weeklies** | Volume ≥ 1.5x + 70%+ confidence |
        | **Monthlies** | 65%+ confidence + price > SMA50 |
        | **Yearly** | 60%+ confidence + price > SMA200 (LEAPS) |

        *Catalysts = news sentiment, multiple technical signals, or event-driven factors.*
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ALERTS
# ═══════════════════════════════════════════════════════════════════════════════
with t3:
    alerts = db.get_alerts(limit=200)

    if not alerts:
        st.markdown(
            '<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);border-radius:14px;'
            'padding:40px 20px;text-align:center;">'
            '<div style="font-size:2rem;margin-bottom:12px;">🔕</div>'
            '<div style="color:rgba(255,255,255,0.5);font-size:0.9rem;">No alerts yet.</div>'
            '<div style="color:rgba(255,255,255,0.3);font-size:0.8rem;margin-top:6px;">'
            f'Alerts fire when confidence ≥ {config.ALERT_THRESHOLD:.0f}%. '
            'The scanner runs every 30 minutes.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # KPI row
        avg_conf = sum(a["confidence"] for a in alerts) / len(alerts)
        emails_sent = sum(1 for a in alerts if a["email_sent"])
        bull_alerts = sum(1 for a in alerts if a["direction"] == "bullish")
        bear_alerts = sum(1 for a in alerts if a["direction"] == "bearish")

        k1, k2, k3, k4 = st.columns(4)
        k1.markdown(_kpi("Total Alerts", str(len(alerts))), unsafe_allow_html=True)
        k2.markdown(_kpi("Avg Confidence", f"{avg_conf:.1f}%", _cc(avg_conf)), unsafe_allow_html=True)
        k3.markdown(_kpi("Bullish", str(bull_alerts), "#00C805"), unsafe_allow_html=True)
        k4.markdown(_kpi("Bearish", str(bear_alerts), "#F23645", f"{emails_sent} emails sent"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Build a lookup of the latest scan results so we can enrich alerts
        # with trade plan + option play data from the most recent scan.
        _latest = scanner.get_latest_results()
        _scan_lookup: dict = {r["symbol"]: r for r in _latest}

        # Timeline
        for alert in alerts:
            ts  = alert["triggered_at"][:19].replace("T", " ")
            ei  = "📧" if alert["email_sent"] else ""

            with st.expander(
                f"{'▲' if alert['direction']=='bullish' else '▼'}  {alert['symbol']}  —  "
                f"{alert['confidence']:.1f}%  ·  {ts}  {ei}"
            ):
                # Merge alert data with the latest scan result for this symbol
                # (scan result has trade plan + option play; alert has the
                # snapshot price/score at trigger time which we preserve).
                scan_r = _scan_lookup.get(alert["symbol"], {})
                merged = {
                    **scan_r,
                    # Always show the alert-time values for the key fields
                    "symbol":     alert["symbol"],
                    "name":       alert.get("name", scan_r.get("name", alert["symbol"])),
                    "price":      alert["price"],
                    "change_pct": alert["change_pct"],
                    "confidence": alert["confidence"],
                    "final_score": alert.get("score", scan_r.get("final_score", 0.0)),
                    "direction":  alert["direction"],
                    "catalysts":  alert.get("catalysts") or scan_r.get("catalysts", []),
                    "signals":    scan_r.get("signals", []),
                    "rsi":        scan_r.get("rsi", 50.0),
                    "vol_ratio":  scan_r.get("vol_ratio", 1.0),
                }

                if scan_r:
                    # Full rich card with trade plan + option play
                    st.markdown(_breakout_card_html(merged), unsafe_allow_html=True)
                else:
                    # Fallback — scan result not available, show plain metrics
                    col = _cc(alert["confidence"])
                    cc  = "#00C805" if alert["change_pct"] >= 0 else "#F23645"
                    st.markdown(
                        f'<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);'
                        f'border-radius:12px;padding:16px 18px;">'
                        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:center;">'
                        f'<div>{_dir_badge(alert["direction"])}</div>'
                        f'<div style="display:flex;flex-direction:column;">'
                        f'<span style="font-size:0.58rem;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.07em;">Price at trigger</span>'
                        f'<span style="font-size:0.95rem;font-weight:700;color:#fff;">${alert["price"]:.2f} '
                        f'<span style="color:{cc};font-size:0.8rem;">{_fp(alert["change_pct"])}</span></span>'
                        f'</div>'
                        f'<div style="display:flex;flex-direction:column;">'
                        f'<span style="font-size:0.58rem;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.07em;">Score</span>'
                        f'<span style="font-size:0.95rem;font-weight:700;color:#fff;">{alert.get("score",0):.0f}/100</span>'
                        f'</div>'
                        f'<div style="display:flex;flex-direction:column;">'
                        f'<span style="font-size:0.58rem;color:rgba(255,255,255,0.3);text-transform:uppercase;letter-spacing:0.07em;">Confidence</span>'
                        f'<span style="font-size:0.95rem;font-weight:800;color:{col};">{alert["confidence"]:.1f}%</span>'
                        f'</div>'
                        f'</div>'
                        + (
                            '<div style="margin-top:10px;font-size:0.75rem;color:rgba(255,255,255,0.35);">'
                            'Run a new scan to load full trade plan &amp; option play for this symbol.</div>'
                        ) +
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    if alert["catalysts"]:
                        label = "Catalysts" if alert["direction"] == "bullish" else "Risks"
                        st.markdown(f"**{label}:**")
                        for cat in alert["catalysts"]:
                            st.markdown(f"- {cat}")

                st.caption(
                    f"Triggered: {ts} UTC · "
                    f"Email: {'sent ✓' if alert['email_sent'] else 'not sent'}"
                )

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
# TAB 4 — CHART
# ═══════════════════════════════════════════════════════════════════════════════
with t4:
    live = scanner.get_latest_results()
    sym_opts = [r["symbol"] for r in live] if live else []

    sc1, sc2, sc3 = st.columns([2, 1, 1])
    with sc1:
        chart_sym = st.selectbox("Symbol", sym_opts or ["AAPL"], key="chart_sym")
    with sc2:
        chart_per = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y"], index=2, key="chart_per")
    with sc3:
        custom_sym = st.text_input("Or type any ticker", placeholder="e.g. ORCL", key="custom_sym").upper().strip()

    sym = custom_sym if custom_sym else chart_sym

    if sym:
        with st.spinner(f"Loading {sym}…"):
            try:
                df = get_price_data(sym, period=chart_per).reset_index()
                df.columns = [c.lower() for c in df.columns]
                dc    = df.columns[0]
                close = df["close"]

                df["sma20"]   = close.rolling(20).mean()
                df["sma50"]   = close.rolling(50).mean()
                df["sma200"]  = close.rolling(200).mean()
                df["vol_avg"] = df["volume"].rolling(20).mean()

                std20 = close.rolling(20).std()
                df["bb_up"] = df["sma20"] + 2 * std20
                df["bb_lo"] = df["sma20"] - 2 * std20

                # RSI
                delta = close.diff()
                gain  = delta.clip(lower=0).rolling(14).mean()
                loss  = (-delta.clip(upper=0)).rolling(14).mean()
                rs    = gain / loss.replace(0, float("nan"))
                df["rsi"] = 100 - (100 / (1 + rs))

                # ── Figure: price + volume + RSI ──────────────────────────────
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.60, 0.20, 0.20],
                    vertical_spacing=0.03,
                )

                # Candlesticks
                fig.add_trace(go.Candlestick(
                    x=df[dc], open=df["open"], high=df["high"],
                    low=df["low"], close=close, name=sym,
                    increasing_line_color="#00C805", decreasing_line_color="#F23645",
                    increasing_fillcolor="#00C805", decreasing_fillcolor="#F23645",
                ), row=1, col=1)

                for name, col_name, clr, dash in [
                    ("SMA 20", "sma20", "#00C805", "solid"),
                    ("SMA 50", "sma50", "#eab308", "dot"),
                    ("SMA 200","sma200","rgba(255,255,255,0.35)","dash"),
                ]:
                    fig.add_trace(go.Scatter(
                        x=df[dc], y=df[col_name], name=name,
                        line=dict(color=clr, width=1.5, dash=dash),
                    ), row=1, col=1)

                fig.add_trace(go.Scatter(
                    x=df[dc], y=df["bb_up"], name="BB Upper",
                    line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=df[dc], y=df["bb_lo"], name="BB Lower",
                    line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(255,255,255,0.025)",
                ), row=1, col=1)

                res = next((r for r in live if r["symbol"] == sym), None)
                is_bull_chart = res["direction"] == "bullish" if res else True
                entry_price = res.get("entry") if res else None
                sl_price    = res.get("stop_loss") if res else None
                tp1_price   = res.get("take_profit_1") if res else None
                tp2_price   = res.get("take_profit_2") if res else None
                tp3_price   = res.get("take_profit_3") if res else None

                # ── Horizontal level lines on price chart ─────────────────────
                x_range = [df[dc].iloc[0], df[dc].iloc[-1]]

                def _hline(price_val, color, label, dash="solid", width=1.5):
                    if price_val is None:
                        return
                    fig.add_shape(type="line", x0=x_range[0], x1=x_range[1],
                                  y0=price_val, y1=price_val,
                                  line=dict(color=color, width=width, dash=dash),
                                  row=1, col=1)
                    fig.add_annotation(
                        x=x_range[1], y=price_val,
                        text=f"  {label} ${price_val:.2f}",
                        showarrow=False, xanchor="left",
                        font=dict(color=color, size=10),
                        row=1, col=1,
                    )

                if entry_price:
                    _hline(entry_price, "#00C805",  "ENTRY", dash="solid",  width=2)
                if sl_price:
                    _hline(sl_price,    "#F23645",  "STOP",  dash="dash",   width=1.5)
                if tp1_price:
                    _hline(tp1_price,   "#22c55e",  "TP1",   dash="dot",    width=1.2)
                if tp2_price:
                    _hline(tp2_price,   "#00C805",  "TP2",   dash="dot",    width=1.2)
                if tp3_price:
                    _hline(tp3_price,   "#86efac",  "TP3",   dash="dot",    width=1)

                # Confidence annotation on last candle
                if res and res["confidence"] >= config.ALERT_THRESHOLD:
                    fig.add_annotation(
                        x=df[dc].iloc[-1], y=float(close.iloc[-1]),
                        text=f"⚡ {res['confidence']:.0f}%",
                        showarrow=True, arrowhead=2, arrowcolor="#00C805",
                        font=dict(color="#00C805", size=12),
                        bgcolor="rgba(0,200,5,0.1)", bordercolor="rgba(0,200,5,0.35)",
                        row=1, col=1,
                    )

                # Volume
                vc = ["#00C805" if c >= o else "#F23645"
                      for c, o in zip(df["close"], df["open"])]
                fig.add_trace(go.Bar(
                    x=df[dc], y=df["volume"], marker_color=vc, opacity=0.6, name="Volume",
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=df[dc], y=df["vol_avg"], name="Vol 20d avg",
                    line=dict(color="#eab308", width=1.5),
                ), row=2, col=1)

                # RSI
                fig.add_trace(go.Scatter(
                    x=df[dc], y=df["rsi"], name="RSI 14",
                    line=dict(color="#0a84ff", width=1.5),
                ), row=3, col=1)
                for lvl, clr in [(70, "rgba(242,54,69,0.4)"), (30, "rgba(0,200,5,0.4)")]:
                    fig.add_hline(y=lvl, line_dash="dot", line_color=clr,
                                  line_width=1, row=3, col=1)

                axis_style = dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False)
                fig.update_layout(
                    paper_bgcolor="#050507", plot_bgcolor="#0A0A0D",
                    font=dict(color="rgba(255,255,255,0.6)", family="Inter, sans-serif"),
                    height=620,
                    xaxis=dict(**axis_style, rangeslider_visible=False),
                    xaxis2=dict(**axis_style),
                    xaxis3=dict(**axis_style),
                    yaxis=dict(**axis_style, tickprefix="$"),
                    yaxis2=dict(**axis_style),
                    yaxis3=dict(**axis_style, range=[0, 100]),
                    legend=dict(bgcolor="#0A0A0D", bordercolor="rgba(255,255,255,0.07)",
                                orientation="h", y=-0.06, font=dict(size=11)),
                    margin=dict(l=8, r=60, t=20, b=8),
                )
                st.plotly_chart(fig, use_container_width=True)

                # ── Precision trade panel below chart ─────────────────────────
                if res:
                    st.markdown("---")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("ML Confidence",  f"{res['confidence']:.1f}%")
                    m2.metric("Breakout Score", f"{res['final_score']:.0f}/100")
                    m3.metric("RSI",            f"{res['rsi']:.1f}")
                    m4.metric("Volume Ratio",   f"{res['vol_ratio']:.2f}x")
                    m5.metric("Direction",      res["direction"].upper())

                    if res.get("entry") is not None:
                        st.markdown(
                            _trade_plan_html(res) + _option_play_html(res)
                            + _alt_data_html(res),
                            unsafe_allow_html=True,
                        )

                    if res.get("catalysts"):
                        st.markdown("**Catalysts / Risks:**")
                        for c in res["catalysts"]:
                            st.markdown(f"- {c}")

            except Exception as e:
                st.error(f"Chart error for {sym}: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
SECTOR_MAP: dict[str, str] = {
    "AAPL":"Technology","MSFT":"Technology","GOOGL":"Technology","META":"Technology",
    "NVDA":"Technology","AMD":"Technology","INTC":"Technology","QCOM":"Technology",
    "AMZN":"Consumer","TSLA":"Consumer","NFLX":"Consumer","SHOP":"Consumer",
    "JPM":"Financials","BAC":"Financials","GS":"Financials","MS":"Financials","V":"Financials",
    "XOM":"Energy","CVX":"Energy","COP":"Energy","SLB":"Energy",
    "JNJ":"Healthcare","PFE":"Healthcare","ABBV":"Healthcare","MRK":"Healthcare",
    "LLY":"Healthcare","UNH":"Healthcare",
    "SPY":"ETFs","QQQ":"ETFs","IWM":"ETFs","DIA":"ETFs",
    "COIN":"Crypto","MSTR":"Crypto","RIOT":"Crypto","MARA":"Crypto",
    "PLTR":"Technology","SOFI":"Financials","HOOD":"Financials",
    "TSLA":"Consumer","RIVN":"Consumer","LCID":"Consumer","NIO":"Consumer",
    "SNAP":"Technology","UBER":"Consumer","LYFT":"Consumer",
}
DEFAULT_SECTOR = "Other"

with t5:
    st.markdown(
        '<div style="padding:4px 0 16px;">'
        '<div style="font-size:0.88rem;color:rgba(255,255,255,0.45);">'
        'Confidence heatmap for scanned stocks, colored by ML confidence and grouped by sector.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    if not all_results:
        st.info("Run a scan to populate the heatmap.")
    else:
        # Group by sector
        sector_buckets: dict[str, list] = {}
        for r in all_results:
            sec = SECTOR_MAP.get(r["symbol"], DEFAULT_SECTOR)
            sector_buckets.setdefault(sec, []).append(r)

        # Sort sectors by average confidence
        sorted_sectors = sorted(
            sector_buckets.items(),
            key=lambda x: sum(r["confidence"] for r in x[1]) / len(x[1]),
            reverse=True,
        )

        hm_filter = st.selectbox(
            "Filter by direction",
            ["All", "Bullish", "Bearish"],
            key="hm_dir",
        )

        for sector_name, sec_rows in sorted_sectors:
            if hm_filter == "Bullish":
                sec_rows = [r for r in sec_rows if r["direction"] == "bullish"]
            elif hm_filter == "Bearish":
                sec_rows = [r for r in sec_rows if r["direction"] == "bearish"]
            if not sec_rows:
                continue

            avg_conf = sum(r["confidence"] for r in sec_rows) / len(sec_rows)
            sec_col  = _cc(avg_conf)

            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;margin-top:18px;">'
                f'<span style="font-size:0.82rem;font-weight:700;color:#fff;">{sector_name}</span>'
                f'<span style="font-size:0.68rem;color:rgba(255,255,255,0.35);">'
                f'{len(sec_rows)} stocks · avg {avg_conf:.0f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Grid: 8 columns
            cols_per_row = 8
            rows_data = [sec_rows[i:i+cols_per_row]
                         for i in range(0, len(sec_rows), cols_per_row)]

            for row_data in rows_data:
                grid_cols = st.columns(cols_per_row)
                for idx, r in enumerate(row_data):
                    c = _cc(r["confidence"])
                    bg_alpha = max(0.06, r["confidence"] / 100 * 0.3)
                    cc_txt = "#00C805" if r["change_pct"] >= 0 else "#F23645"
                    with grid_cols[idx]:
                        st.markdown(
                            f'<div class="heat-cell" style="background:rgba({_hex_to_rgb(c)},{bg_alpha:.2f});">'
                            f'<div style="font-size:0.8rem;font-weight:700;color:#fff;">{r["symbol"]}</div>'
                            f'<div style="font-size:0.95rem;font-weight:800;color:{c};">{r["confidence"]:.0f}%</div>'
                            f'<div style="font-size:0.68rem;color:{cc_txt};margin-top:2px;">{_fp(r["change_pct"])}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.72rem;color:rgba(255,255,255,0.3);">'
            '🟢 ≥90%&nbsp;&nbsp;🟩 ≥75%&nbsp;&nbsp;🟡 ≥60%&nbsp;&nbsp;🟠 ≥45%&nbsp;&nbsp;🔴 &lt;45%'
            '</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL
# ═══════════════════════════════════════════════════════════════════════════════
with t6:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            '<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);'
            'border-radius:14px;padding:22px 20px;">'
            '<div style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:14px;">Algorithm</div>',
            unsafe_allow_html=True,
        )
        status = "✅ Trained & active" if is_model_trained() else "⏳ Training in background…"
        st.markdown(f"""
**Status:** {status}

**Algorithm:** Gradient Boosting + Platt Calibration  
**Training data:** 27 stocks × 2 years daily OHLCV  
**Breakout label:** +5% within 5 trading days  

**18 Features:** RSI + slope, MACD hist + slope, BB squeeze ratio, BB %B,
Volume ratio (1d + 5d), Price vs SMA 20/50/200, SMA slope, ATR%, 5d & 20d returns,
distance from 52-week high, OBV slope, BB width  

**Validation:** 3-fold CV ROC-AUC  
**Confidence formula:** 60% ML + 40% rule-based (RSI, MACD, volume, trend)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(
            '<div style="background:#0C0C10;border:1px solid rgba(255,255,255,0.06);'
            'border-radius:14px;padding:22px 20px;">'
            '<div style="font-size:0.75rem;font-weight:700;color:rgba(255,255,255,0.4);'
            'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:14px;">Pipeline</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"""
1. Fetch S&P 500 + NASDAQ-100 + Russell 2000 (~3,000 tickers)
2. Batch pre-screen: price ≥ ${config.MIN_PRICE:.0f}, avg volume ≥ {config.MIN_AVG_VOLUME//1000:,}k
3. Deep analysis on ~400–800 candidates with **{scanner._MAX_WORKERS} parallel threads**
4. Results sorted by ML confidence, top 200 stored

**Alert fires when ALL:**  
✅ Confidence ≥ {config.ALERT_THRESHOLD:.0f}%  
✅ Direction is bullish  
✅ Not alerted same symbol in last 4 hours
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        from ml_model import TRAINING_SYMBOLS
        with st.expander("Training symbols"):
            st.caption(", ".join(TRAINING_SYMBOLS))

        if st.button("🔁 Retrain Model", use_container_width=True):
            with st.spinner("Training… (~3 min)"):
                try:
                    train_model(force=True)
                    st.success("Model retrained successfully!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.markdown("---")
    st.info(
        "⚠️ **Disclaimer:** BreakoutAI is for educational and research purposes only. "
        "ML confidence scores do not guarantee profitable trades. "
        "Always do your own due diligence before making any investment decision."
    )


