# ⚡ BreakoutAI

> **Predict stock breakouts before they happen.**  
> Always-on Streamlit app — deploy once from GitHub, runs 24/7 for free.

Combines a calibrated ML model (Gradient Boosting + Platt Scaling) with
technical indicators and news sentiment to flag high-confidence breakout
setups and **email you automatically** when confidence ≥ 95%.

---

## Live Features

| Feature | Detail |
|---|---|
| **Background Scanner** | APScheduler scans your watchlist every 15 min (configurable) |
| **ML Confidence Score** | GBM + Platt Calibration — calibrated 0–100% probability |
| **Email Alerts** | HTML email sent automatically when confidence ≥ threshold |
| **Technical Signals** | RSI, MACD, Bollinger Squeeze, Volume Surge, 52-week high, Golden/Death Cross |
| **News Sentiment** | Yahoo Finance RSS + optional NewsAPI (TextBlob NLP) |
| **Candlestick Charts** | Plotly charts with SMA 20/50 overlay and signal annotations |
| **Alert History** | SQLite log with CSV export |
| **Watchlist** | Add/remove symbols any time from the UI |

---

## Deploy to Streamlit Community Cloud (Free, Always-On)

This is the recommended way to run the app 24/7 without keeping your computer on.

### Step 1 — Fork the repo

1. Go to `https://github.com/YOUR_USERNAME/breakoutai` (after pushing — see below)
2. Or click **Use this template** / **Fork**

### Step 2 — Push to your GitHub account

```bash
git clone https://github.com/YOUR_USERNAME/breakoutai
# OR start fresh:
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/breakoutai.git
git push -u origin main
```

### Step 3 — Create the Streamlit Cloud app

1. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in with GitHub
2. Click **New app**
3. Select your repo, branch `main`, main file `app.py`
4. Click **Deploy**

### Step 4 — Add your secrets

In the Streamlit Cloud dashboard → your app → **⚙️ Settings → Secrets**, paste:

```toml
# Required for email alerts
ALERT_EMAIL_FROM     = "you@gmail.com"
ALERT_EMAIL_PASSWORD = "xxxx xxxx xxxx xxxx"   # Gmail App Password
ALERT_EMAIL_TO       = "you@gmail.com"

# Optional: comma-separate multiple recipients
# ALERT_EMAIL_TO = "you@gmail.com,partner@gmail.com"

# Optional enrichment (free API keys)
NEWS_API_KEY = ""     # https://newsapi.org/register
FINNHUB_KEY  = ""     # https://finnhub.io/

# Tuning (these have sensible defaults)
SCAN_INTERVAL_MINUTES = "15"
ALERT_THRESHOLD       = "95"
DEFAULT_WATCHLIST     = "AAPL,MSFT,NVDA,TSLA,META,GOOGL,AMZN,AMD,PLTR,COIN,SOFI,HOOD,MSTR,RIVN,SNAP,UBER,SHOP,PYPL,ROKU,SQ"
```

> **Gmail App Password** — do NOT use your normal Gmail password.  
> Generate one at [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords).  
> Select "Mail" + "Other (Custom name)" → copy the 16-character password.

### Step 5 — Done!

Your app is live at `https://YOUR_APP.streamlit.app`.  
The ML model trains in the background on first boot (~3 min).  
The scanner starts immediately and runs every 15 minutes.

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/breakoutai
cd breakoutai

# Create and activate virtualenv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux

# Install dependencies
pip install -r requirements.txt

# Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your values

# Run
streamlit run app.py
```

Open **http://localhost:8501**

---

## Repo Structure

```
breakoutai/
├── app.py              ← Streamlit app (single entry point)
├── config.py           ← Reads from st.secrets OR .env, with defaults
├── paths.py            ← Writable data paths (local: data/ | cloud: /tmp/)
├── database.py         ← SQLite: alerts, scan log, watchlist
├── data_engine.py      ← yfinance + Yahoo RSS + NewsAPI + indicators
├── ml_model.py         ← GBM + Platt calibration training & inference
├── scanner.py          ← APScheduler background scan loop + alert logic
├── alerts.py           ← SMTP HTML email sender
├── requirements.txt
├── .gitignore          ← Excludes .env, secrets.toml, data/, model.pkl
├── .streamlit/
│   ├── config.toml               ← Dark theme + server settings
│   └── secrets.toml.example      ← Template (never committed)
└── .github/
    └── workflows/ci.yml          ← GitHub Actions CI
```

---

## How the Score Works

```
Final confidence = 60% ML probability + 40% rule-based score

Rule-based score = 40% Technical + 25% Momentum + 20% Volume + 15% Sentiment

Alert fires when:
  • Confidence ≥ threshold (default 95%)
  • Direction is bullish
  • Same symbol not alerted in last 4 hours
```

### ML Model

| Setting | Value |
|---|---|
| Algorithm | Gradient Boosting Classifier |
| Calibration | Platt Scaling (sigmoid, 3-fold CV) |
| Training data | 27 stocks × 2 years daily OHLCV |
| Breakout label | +5% price move within 5 trading days |
| Features | 18 (RSI, MACD, BB squeeze, volume, momentum, OBV, etc.) |
| Validation | 3-fold ROC-AUC logged at training time |

---

## Important Notes

- **Data persistence on Streamlit Cloud:** The SQLite database and trained model are stored in `/tmp/breakoutai/` on Streamlit Cloud. They persist for the lifetime of the process. If the app restarts (e.g. after a code push or daily reboot), the model retrains automatically (~3 min) and the alert history resets. For permanent history, use the CSV export button.

- **Streamlit Cloud free tier** keeps your app alive as long as it has traffic at least once every 7 days. The background scanner keeps running as long as the Streamlit server process is alive.

- **Scanner during off-hours:** APScheduler runs inside the same process as Streamlit. If no one visits the app for a while, Streamlit Cloud may put it to sleep. For truly 24/7 operation with guaranteed uptime, consider deploying to a VPS (e.g. Railway, Render, or a $5/month DigitalOcean droplet).

---

## Disclaimer

BreakoutAI is for **educational and research purposes only**.  
A 95% ML confidence score does not guarantee a profitable trade.  
Markets can behave unexpectedly. Always do your own due diligence.
