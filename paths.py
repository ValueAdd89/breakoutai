"""
Centralised path resolution.

On Streamlit Cloud the filesystem is ephemeral — the repo root is read-only
but /tmp is writable. We store runtime data (SQLite DB, trained model) in
a writable directory that works on both local and cloud environments.
"""
import os
from pathlib import Path

# Repo root — always the directory this file lives in
REPO_ROOT = Path(__file__).parent

# Writable data directory:
#   - Locally: <repo>/data/
#   - Streamlit Cloud: /tmp/breakoutai/  (persists for the lifetime of the process)
_is_cloud = os.getenv("STREAMLIT_SHARING_MODE") == "streamlit" or os.getenv("HOME") == "/home/appuser"

if _is_cloud:
    DATA_DIR = Path("/tmp/breakoutai")
else:
    DATA_DIR = REPO_ROOT / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH    = DATA_DIR / "breakout.db"
MODEL_PATH = DATA_DIR / "model.pkl"
