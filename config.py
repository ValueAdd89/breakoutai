"""
Configuration loader.

Priority order (highest → lowest):
  1. Streamlit secrets  (st.secrets) — used on Streamlit Cloud
  2. Environment variables            — used with a local .env file
  3. Hard-coded defaults

This means the same code runs identically locally and on the cloud.
"""
import os
from pathlib import Path

# Load .env when running locally (no-op if the file doesn't exist)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def _get(key: str, default: str = "") -> str:
    """Read a config value from st.secrets first, then env vars, then default."""
    # Try Streamlit secrets (only available when Streamlit is actually running)
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()


# ── Email ─────────────────────────────────────────────────────────────────────
ALERT_EMAIL_FROM: str = _get("ALERT_EMAIL_FROM")
ALERT_EMAIL_PASSWORD: str = _get("ALERT_EMAIL_PASSWORD")
ALERT_EMAIL_TO: list[str] = [
    e.strip() for e in _get("ALERT_EMAIL_TO").split(",") if e.strip()
]
SMTP_HOST: str = _get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(_get("SMTP_PORT", "587"))

# ── APIs ─────────────────────────────────────────────────────────────────────
NEWS_API_KEY: str = _get("NEWS_API_KEY")
FINNHUB_KEY: str = _get("FINNHUB_KEY")

# ── Scanner ───────────────────────────────────────────────────────────────────
SCAN_INTERVAL_MINUTES: int = int(_get("SCAN_INTERVAL_MINUTES", "15"))
ALERT_THRESHOLD: float = float(_get("ALERT_THRESHOLD", "95"))

DEFAULT_WATCHLIST: list[str] = [
    s.strip().upper()
    for s in _get(
        "DEFAULT_WATCHLIST",
        "AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,AMD,PLTR,COIN,SOFI,HOOD,MSTR,RIVN,SNAP,UBER,SHOP,PYPL,ROKU,SQ",
    ).split(",")
    if s.strip()
]

EMAIL_CONFIGURED: bool = bool(ALERT_EMAIL_FROM and ALERT_EMAIL_PASSWORD and ALERT_EMAIL_TO)
