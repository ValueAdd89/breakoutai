"""
ML breakout confidence model.

Trains a GradientBoosting classifier with Platt calibration on 2 years of
daily OHLCV data. Label = 1 if +5% move occurs within 5 trading days.

The model is persisted to paths.MODEL_PATH so it survives process restarts.
On Streamlit Cloud this lives in /tmp/breakoutai/ for the process lifetime.
"""
import logging
import pickle
import threading
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

from paths import MODEL_PATH

logger = logging.getLogger(__name__)

_model: Optional[Pipeline] = None
_model_lock = threading.Lock()

TRAINING_SYMBOLS = [
    "AAPL", "MSFT", "NVDA", "TSLA", "META", "GOOGL", "AMZN",
    "AMD",  "NFLX", "PYPL", "SQ",   "SHOP", "COIN",  "UBER", "SNAP",
    "PLTR", "SOFI", "HOOD", "ROKU", "TWLO",
    "JPM",  "BAC",  "GS",   "XOM",  "CVX",  "SPY",   "QQQ",
]

BREAKOUT_PCT  = 0.05   # +5% within LOOKFORWARD days = breakout
LOOKFORWARD   = 5      # trading days


# ── Feature extraction ────────────────────────────────────────────────────────

def _rsi(series: pd.Series, window: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    return float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9)))


def extract_features(df: pd.DataFrame) -> Optional[np.ndarray]:
    """18-feature vector from OHLCV DataFrame. Returns None if insufficient data."""
    if len(df) < 60:
        return None
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        rsi_now   = _rsi(close, 14)
        rsi_5d    = _rsi(close.iloc[:-5], 14) if len(close) > 19 else rsi_now
        rsi_slope = rsi_now - rsi_5d

        ema12      = close.ewm(span=12, adjust=False).mean()
        ema26      = close.ewm(span=26, adjust=False).mean()
        macd       = ema12 - ema26
        macd_sig   = macd.ewm(span=9, adjust=False).mean()
        macd_hist  = float(macd.iloc[-1] - macd_sig.iloc[-1])
        macd_hist2 = float(macd.iloc[-2] - macd_sig.iloc[-2])
        macd_slope = macd_hist - macd_hist2

        sma20       = close.rolling(20).mean()
        std20       = close.rolling(20).std()
        bb_up       = sma20 + 2 * std20
        bb_lo       = sma20 - 2 * std20
        bb_width    = float((bb_up.iloc[-1] - bb_lo.iloc[-1]) / (sma20.iloc[-1] + 1e-9))
        bb_avg_w    = float(((bb_up - bb_lo) / (sma20 + 1e-9)).rolling(20).mean().iloc[-1])
        bb_squeeze  = bb_width / (bb_avg_w + 1e-9)
        bb_pct      = float((close.iloc[-1] - bb_lo.iloc[-1]) / (bb_up.iloc[-1] - bb_lo.iloc[-1] + 1e-9))

        vol_ma20    = volume.rolling(20).mean()
        vol_ratio   = float(volume.iloc[-1] / (vol_ma20.iloc[-1] + 1e-9))
        vol_5d_ratio= float(volume.iloc[-5:].mean() / (vol_ma20.iloc[-1] + 1e-9))

        sma50       = close.rolling(50).mean()
        sma200_s    = close.rolling(200).mean()
        sma200_val  = sma200_s.iloc[-1] if not pd.isna(sma200_s.iloc[-1]) else sma50.iloc[-1]
        p_vs_20     = float(close.iloc[-1] / (sma20.iloc[-1] + 1e-9) - 1)
        p_vs_50     = float(close.iloc[-1] / (sma50.iloc[-1] + 1e-9) - 1)
        p_vs_200    = float(close.iloc[-1] / (sma200_val + 1e-9) - 1)
        sma20_slope = float((sma20.iloc[-1] - sma20.iloc[-5]) / (sma20.iloc[-5] + 1e-9))

        tr          = pd.concat([high - low,
                                  (high - close.shift(1)).abs(),
                                  (low  - close.shift(1)).abs()], axis=1).max(axis=1)
        atr_pct     = float(tr.rolling(14).mean().iloc[-1]) / (float(close.iloc[-1]) + 1e-9)

        ret5d       = float((close.iloc[-1] - close.iloc[-6])  / (close.iloc[-6]  + 1e-9))
        ret20d      = float((close.iloc[-1] - close.iloc[-21]) / (close.iloc[-21] + 1e-9))

        high52      = float(high.rolling(min(252, len(high))).max().iloc[-1])
        pct52       = (float(close.iloc[-1]) - high52) / (high52 + 1e-9)

        obv         = (volume * close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))).cumsum()
        obv_slope   = float((obv.iloc[-1] - obv.iloc[-6]) / (abs(obv.iloc[-6]) + 1e-9))

        feat = np.array([
            rsi_now, rsi_slope,
            macd_hist, macd_slope,
            bb_squeeze, bb_pct,
            vol_ratio, vol_5d_ratio,
            p_vs_20, p_vs_50, p_vs_200, sma20_slope,
            atr_pct,
            ret5d, ret20d,
            pct52,
            obv_slope,
            bb_width,
        ], dtype=np.float64)

        return feat if not (np.any(np.isnan(feat)) or np.any(np.isinf(feat))) else None
    except Exception as e:
        logger.debug(f"Feature extraction error: {e}")
        return None


# ── Training ──────────────────────────────────────────────────────────────────

def _build_dataset() -> tuple[np.ndarray, np.ndarray]:
    X_list, y_list = [], []
    for sym in TRAINING_SYMBOLS:
        try:
            df = yf.Ticker(sym).history(period="2y", auto_adjust=True)
            if df.empty or len(df) < 100:
                continue
            df.columns = [c.lower() for c in df.columns]
            df = df[["open","high","low","close","volume"]].dropna().reset_index(drop=True)
            for i in range(60, len(df) - LOOKFORWARD):
                feat = extract_features(df.iloc[:i+1])
                if feat is None:
                    continue
                future_max = df["close"].iloc[i+1:i+1+LOOKFORWARD].max()
                label = 1 if (future_max - df["close"].iloc[i]) / (df["close"].iloc[i] + 1e-9) >= BREAKOUT_PCT else 0
                X_list.append(feat)
                y_list.append(label)
            logger.info(f"Built features for {sym} — {len(X_list)} samples total")
        except Exception as e:
            logger.warning(f"Dataset error {sym}: {e}")
    if len(X_list) < 100:
        raise RuntimeError("Insufficient training data collected")
    return np.array(X_list), np.array(y_list)


def train_model(force: bool = False) -> Pipeline:
    global _model
    if not force and MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            logger.info("Loaded pre-trained model from disk")
            return _model
        except Exception as e:
            logger.warning(f"Could not load saved model: {e} — retraining")

    logger.info("Training breakout ML model (this takes 2-3 minutes)…")
    X, y = _build_dataset()
    logger.info(f"Dataset: {len(X)} samples, {y.mean():.1%} positive breakouts")

    base = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=20, random_state=42,
    )
    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    pipeline   = Pipeline([("scaler", StandardScaler()), ("clf", calibrated)])
    pipeline.fit(X, y)

    cv = cross_val_score(pipeline, X, y, cv=3, scoring="roc_auc")
    logger.info(f"Model ROC-AUC CV: {cv.mean():.3f} ± {cv.std():.3f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)

    _model = pipeline
    logger.info("Model saved to disk")
    return _model


# ── Inference ─────────────────────────────────────────────────────────────────

def get_confidence(df: pd.DataFrame) -> float:
    """Return calibrated breakout probability 0–100."""
    global _model
    with _model_lock:
        if _model is None:
            try:
                _model = train_model()
            except Exception as e:
                logger.error(f"Model unavailable: {e}")
                return 0.0

    feat = extract_features(df)
    if feat is None:
        return 0.0
    try:
        prob = float(_model.predict_proba([feat])[0][1]) * 100
        return round(min(99.9, max(0.1, prob)), 1)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return 0.0


def is_model_trained() -> bool:
    return MODEL_PATH.exists()
