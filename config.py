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


def _resolve(path_str: str) -> Path:
    """Resolve relative paths against the project dir, not the caller's CWD.

    LM Studio launches stdio servers from an arbitrary working directory.
    """
    path = Path(path_str).expanduser()
    return path if path.is_absolute() else PROJECT_DIR / path


def load_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env")
    return Settings(
        comfyui_url=os.getenv("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/"),
        workflow_path=_resolve(os.getenv("WORKFLOW_PATH", "workflow_api.json")),
        output_dir=_resolve(os.getenv("OUTPUT_DIR", "outputs")),
        timeout=float(os.getenv("TIMEOUT", "180")),
    )
