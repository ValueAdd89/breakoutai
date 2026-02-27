"""
SQLite persistence — alert history and scan log.
The watchlist concept is replaced by the full-market universe scanner.
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
        CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol);
        CREATE INDEX IF NOT EXISTS idx_alerts_time   ON alerts(triggered_at);
        CREATE INDEX IF NOT EXISTS idx_scan_symbol   ON scan_log(symbol);
        """)
        # Migrate: add scan_results table if it doesn't exist yet
        c.execute("""
        CREATE TABLE IF NOT EXISTS scan_results (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol     TEXT NOT NULL,
            payload    TEXT NOT NULL,
            scanned_at TEXT NOT NULL
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_symbol ON scan_results(symbol)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_results_time   ON scan_results(scanned_at)")

        # Evict persisted results that pre-date the entry/tp3/rr fields so
        # stale rows don't suppress the new trade plan UI.
        try:
            row = c.execute(
                "SELECT payload FROM scan_results LIMIT 1"
            ).fetchone()
            if row:
                sample = json.loads(row["payload"])
                if "entry" not in sample or "take_profit_3" not in sample:
                    c.execute("DELETE FROM scan_results")
        except Exception:
            pass


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


def get_alerts(limit: int = 200) -> list[dict]:
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


def already_alerted_recently(symbol: str, direction: str = "bullish", hours: int = 4) -> bool:
    """Check if we already alerted this symbol+direction recently (avoids spam, allows bull+bear separately)."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT id FROM alerts WHERE symbol=? AND direction=? AND triggered_at > ? LIMIT 1",
            (symbol, direction, cutoff),
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


# ── Scan result persistence (survive restarts / market close) ─────────────────

def save_scan_results(results: list[dict]) -> None:
    """
    Persist the full list of scan results to SQLite.
    Replaces all previous rows — we only keep the most recent scan.
    Each result is stored as a JSON blob keyed by symbol + timestamp.
    """
    if not results:
        return
    now = datetime.now(timezone.utc).isoformat()
    with _lock, _conn() as c:
        c.execute("DELETE FROM scan_results")
        c.executemany(
            "INSERT INTO scan_results (symbol, payload, scanned_at) VALUES (?, ?, ?)",
            [(r["symbol"], json.dumps(r), now) for r in results],
        )


def load_scan_results() -> tuple[list[dict], Optional[str]]:
    """
    Load the most recently persisted scan results from SQLite.
    Returns (results_list, scanned_at_iso_string).
    Returns ([], None) if nothing is stored yet or the table doesn't exist.
    """
    try:
        with _lock, _conn() as c:
            # Ensure table exists even if init_db wasn't called yet
            c.execute("""
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                payload TEXT NOT NULL,
                scanned_at TEXT NOT NULL
            )""")
            rows = c.execute(
                "SELECT payload, scanned_at FROM scan_results ORDER BY id ASC"
            ).fetchall()
    except Exception:
        return [], None
    if not rows:
        return [], None
    results = []
    for row in rows:
        try:
            results.append(json.loads(row["payload"]))
        except Exception:
            pass
    scanned_at = rows[-1]["scanned_at"] if rows else None
    return results, scanned_at
