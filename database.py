"""
SQLite persistence — alert history, scan log, watchlist.
Uses the path from paths.py so it works locally and on Streamlit Cloud.
"""
import sqlite3
import json
import threading
from datetime import datetime, timezone, timedelta
from typing import Optional

from paths import DB_PATH

_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS alerts (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol       TEXT    NOT NULL,
            name         TEXT,
            price        REAL,
            change_pct   REAL,
            score        REAL,
            confidence   REAL,
            direction    TEXT,
            catalysts    TEXT,
            email_sent   INTEGER DEFAULT 0,
            triggered_at TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scan_log (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     TEXT NOT NULL,
            score      REAL,
            confidence REAL,
            direction  TEXT,
            price      REAL,
            scanned_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol   TEXT PRIMARY KEY,
            added_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_alerts_symbol   ON alerts(symbol);
        CREATE INDEX IF NOT EXISTS idx_alerts_time     ON alerts(triggered_at);
        CREATE INDEX IF NOT EXISTS idx_scan_symbol     ON scan_log(symbol);
        """)


def save_alert(
    symbol: str, name: Optional[str], price: float, change_pct: float,
    score: float, confidence: float, direction: str,
    catalysts: list[str], email_sent: bool,
) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            """INSERT INTO alerts
               (symbol,name,price,change_pct,score,confidence,direction,catalysts,email_sent,triggered_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (symbol, name, price, change_pct, score, confidence,
             direction, json.dumps(catalysts), int(email_sent),
             datetime.now(timezone.utc).isoformat()),
        )
        return cur.lastrowid  # type: ignore


def get_alerts(limit: int = 100) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM alerts ORDER BY triggered_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["catalysts"] = json.loads(d.get("catalysts") or "[]")
        out.append(d)
    return out


def already_alerted_recently(symbol: str, hours: int = 4) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT id FROM alerts WHERE symbol=? AND triggered_at > ? LIMIT 1",
            (symbol, cutoff),
        ).fetchone()
    return row is not None


def log_scan(symbol: str, score: float, confidence: float, direction: str, price: float) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO scan_log (symbol,score,confidence,direction,price,scanned_at) VALUES (?,?,?,?,?,?)",
            (symbol, score, confidence, direction, price,
             datetime.now(timezone.utc).isoformat()),
        )


def get_scan_history(symbol: str, limit: int = 50) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT * FROM scan_log WHERE symbol=? ORDER BY scanned_at DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_watchlist() -> list[str]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT symbol FROM watchlist ORDER BY added_at").fetchall()
    return [r["symbol"] for r in rows]


def add_to_watchlist(symbol: str) -> None:
    with _lock, _conn() as c:
        c.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?,?)",
            (symbol, datetime.now(timezone.utc).isoformat()),
        )


def remove_from_watchlist(symbol: str) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM watchlist WHERE symbol=?", (symbol,))


def bulk_init_watchlist(symbols: list[str]) -> None:
    """Seed watchlist on first run if empty."""
    with _lock, _conn() as c:
        existing = c.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        if existing == 0:
            now = datetime.now(timezone.utc).isoformat()
            c.executemany(
                "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?,?)",
                [(s, now) for s in symbols],
            )
