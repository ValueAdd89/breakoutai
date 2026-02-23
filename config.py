"""
Configuration loader.

Priority order (highest → lowest):
  1. Streamlit secrets  (st.secrets) — used on Streamlit Cloud
  2. Environment variables            — used with a local .env file
  3. Hard-coded defaults
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()


# ── Email ─────────────────────────────────────────────────────────────────────
ALERT_EMAIL_FROM: str     = _get("ALERT_EMAIL_FROM")
ALERT_EMAIL_PASSWORD: str = _get("ALERT_EMAIL_PASSWORD")
ALERT_EMAIL_TO: list[str] = [e.strip() for e in _get("ALERT_EMAIL_TO").split(",") if e.strip()]
SMTP_HOST: str            = _get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int            = int(_get("SMTP_PORT", "587"))

# ── APIs ──────────────────────────────────────────────────────────────────────
NEWS_API_KEY: str = _get("NEWS_API_KEY")
FINNHUB_KEY: str  = _get("FINNHUB_KEY")

# ── Scanner behaviour ─────────────────────────────────────────────────────────
# How often (minutes) to re-run the full-market scan
SCAN_INTERVAL_MINUTES: int = int(_get("SCAN_INTERVAL_MINUTES", "30"))

# ML confidence % required to fire an alert
ALERT_THRESHOLD: float = float(_get("ALERT_THRESHOLD", "95"))

# ── Pre-screener filters ──────────────────────────────────────────────────────
# Minimum stock price — eliminates penny stocks
MIN_PRICE: float = float(_get("MIN_PRICE", "2.0"))

# Minimum average daily volume — eliminates illiquid names
MIN_AVG_VOLUME: int = int(_get("MIN_AVG_VOLUME", "500000"))

# ── Derived ───────────────────────────────────────────────────────────────────
EMAIL_CONFIGURED: bool = bool(ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD and ALERT_EMAIL_TO)
