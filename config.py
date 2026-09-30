"""Central configuration for QueryPilot.

All settings come from environment variables (optionally loaded from a local
`.env` file). Secrets are never logged or shown to the user.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "querypilot.db"
# Private copies of user-uploaded databases (never the sample database's folder itself).
DEFAULT_UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None
    gemini_model: str
    db_path: Path
    max_retries: int
    max_rows: int
    query_timeout_seconds: float
    upload_dir: Path = DEFAULT_UPLOAD_DIR
    max_upload_mb: int = 50


def _path_env(name: str, default: Path) -> Path:
    path = Path(os.getenv(name) or default)
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_settings() -> Settings:
    db_path = _path_env("QUERYPILOT_DB_PATH", DEFAULT_DB_PATH)
    return Settings(
        gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
        gemini_model=os.getenv("GEMINI_MODEL") or "gemini-3.5-flash-lite",
        db_path=db_path,
        max_retries=min(5, max(0, _int_env("QUERYPILOT_MAX_RETRIES", 2))),
        max_rows=max(1, _int_env("QUERYPILOT_MAX_ROWS", 200)),
        query_timeout_seconds=max(0.1, _float_env("QUERYPILOT_QUERY_TIMEOUT_SECONDS", 5.0)),
        upload_dir=_path_env("QUERYPILOT_UPLOAD_DIR", DEFAULT_UPLOAD_DIR),
        max_upload_mb=min(200, max(1, _int_env("QUERYPILOT_MAX_UPLOAD_MB", 50))),
    )
