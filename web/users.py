"""Who may use the Telegram bot besides its owner: allowed, pending and denied users, saved to a JSON file."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

KINDS = ("allowed", "pending", "denied")


class UserStore:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: dict[str, dict[str, str]] = self._load()

    def is_allowed(self, user_id: int) -> bool:
        return str(user_id) in self._data["allowed"]

    def is_pending(self, user_id: int) -> bool:
        return str(user_id) in self._data["pending"]

    def is_denied(self, user_id: int) -> bool:
        return str(user_id) in self._data["denied"]

    def name_of(self, user_id: int) -> str:
        key = str(user_id)
        return next((self._data[kind][key] for kind in KINDS if key in self._data[kind]), f"user {user_id}")

    def allowed_users(self) -> list[tuple[int, str]]:
        return [(int(key), name) for key, name in self._data["allowed"].items()]

    def add_pending(self, user_id: int, name: str) -> None:
        self._move(user_id, "pending", name)

    def allow(self, user_id: int, name: str | None = None) -> None:
        self._move(user_id, "allowed", name)

    def deny(self, user_id: int) -> None:
        self._move(user_id, "denied", None)

    def remove(self, user_id: int) -> None:
        """Forget a user completely; they can ask again."""
        for kind in KINDS:
            self._data[kind].pop(str(user_id), None)
        self._save()

    def _move(self, user_id: int, kind: str, name: str | None) -> None:
        key = str(user_id)
        name = name or self.name_of(user_id)
        for other in KINDS:
            self._data[other].pop(key, None)
        self._data[kind][key] = name
        self._save()

    def _load(self) -> dict[str, dict[str, str]]:
        empty = {kind: {} for kind in KINDS}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return empty
        except (OSError, ValueError) as exc:
            log.warning("Ignoring unreadable %s (%s); starting with no approved users", self._path, exc)
            return empty
        return {kind: {str(k): str(v) for k, v in dict(raw.get(kind, {})).items()} for kind in KINDS}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic: a crash never leaves a half-written file
