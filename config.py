"""Settings loaded from environment variables / a `.env` file next to this module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    comfyui_url: str
    workflow_path: Path
    output_dir: Path
    timeout: float
    comfyui_output_dir: Path | None
    cache_dir: Path
    telegram_bot_token: str | None = None
    telegram_allowed_user_id: int | None = None  # the bot's owner; approves everyone else
    telegram_users_path: Path = PROJECT_DIR / "data" / "telegram_users.json"


def _resolve(path_str: str) -> Path:
    """Resolve relative paths against the project dir, not the caller's CWD.

    LM Studio launches stdio servers from an arbitrary working directory.
    """
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def _int_or_none(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    return Settings(
        comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/"),
        workflow_path=_resolve(os.getenv("WORKFLOW_PATH", "workflow_api.json")),
        output_dir=_resolve(os.getenv("OUTPUT_DIR", "outputs")),
        timeout=float(os.getenv("TIMEOUT", "180")),
        comfyui_output_dir=_resolve(raw) if (raw := os.getenv("COMFYUI_OUTPUT_DIR", "").strip()) else None,
        cache_dir=_resolve(os.getenv("CACHE_DIR", "cache")),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
        telegram_allowed_user_id=_int_or_none(os.getenv("TELEGRAM_ALLOWED_USER_ID", "")),
    )
