"""Telegram bot behaviour without the Telegram library: access control, commands, generation, Vary."""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

from comfy_client import GenerationParams, InvalidParamsError
from web.jobs import BusyError, JobManager

log = logging.getLogger(__name__)

SHAPES = {"square": (512, 512), "portrait": (512, 768), "landscape": (768, 512)}
EDIT_INTERVAL = 1.5  # seconds between progress edits (Telegram rate-limits edits)
CAPTION_PROMPT_MAX = 900
VARY_KEEP = 200
HELP = (
    "Send me a prompt and I'll paint it on your PC.\n\n"
    "• a beach at sunset → square image\n"
    "• /portrait an old fisherman → tall 512×768\n"
    "• /landscape mountains at dawn → wide 768×512\n\n"
    "Tap 🔁 Vary under an image for a new version."
)
BUSY = "⏳ Busy with another image. Try again in a moment."
EXPIRED = "This button expired. Send the prompt again."
NOT_FOUND = "⚠️ The image was generated but couldn't be found in ComfyUI's output folder."


class Chat(Protocol):
    async def send_text(self, text: str) -> int: ...
    async def edit_text(self, message_id: int, text: str) -> None: ...
    async def delete(self, message_id: int) -> None: ...
    async def send_photo(self, path: Path, caption: str, vary_token: str) -> None: ...


class GalleryLookup(Protocol):
    async def get(self) -> Any: ...


def parse_request(text: str) -> tuple[str | None, str]:
    """Return (shape, prompt). Shape is None for /start, /help and unknown commands."""
    text = text.strip()
    if not text.startswith("/"):
        return "square", text
    parts = text.split(None, 1)
    command = parts[0][1:].split("@", 1)[0].lower()
    rest = parts[1].strip() if len(parts) > 1 else ""
    return (command if command in SHAPES else None), rest


def caption_for(params: GenerationParams, seed: int, elapsed: float) -> str:
    prompt = params.prompt
    if len(prompt) > CAPTION_PROMPT_MAX:
        prompt = prompt[: CAPTION_PROMPT_MAX - 1] + "…"
    return f"{prompt}\n\nseed {seed} · {params.width}×{params.height} · {elapsed}s"


class VaryStore:
    """Short tokens for 🔁 Vary buttons (Telegram callback data is limited to 64 bytes)."""

    def __init__(self, keep: int = VARY_KEEP) -> None:
        self._items: OrderedDict[str, GenerationParams] = OrderedDict()
        self._keep = keep

    def put(self, params: GenerationParams) -> str:
        token = uuid.uuid4().hex[:16]
        self._items[token] = replace(params, seed=None)
        while len(self._items) > self._keep:
            self._items.popitem(last=False)
        return token

    def get(self, token: str) -> GenerationParams | None:
        return self._items.get(token)


class StudioBot:
    def __init__(self, jobs: JobManager, galleries: GalleryLookup, allowed_user_id: int | None) -> None:
        self.jobs = jobs
        self.galleries = galleries
        self.allowed_user_id = allowed_user_id
        self.vary = VaryStore()

    async def handle_text(self, user_id: int, text: str, chat: Chat) -> None:
        if not await self._authorized(user_id, chat):
            return
        shape, prompt = parse_request(text)
        if shape is None:
            await chat.send_text(HELP)
        elif not prompt:
            await chat.send_text(f"Add a prompt after the command, e.g. /{shape} an old fisherman")
        else:
            width, height = SHAPES[shape]
            await self._generate(GenerationParams(prompt=prompt, width=width, height=height), chat)

    async def handle_vary(self, user_id: int, token: str, chat: Chat) -> None:
        if not await self._authorized(user_id, chat):
            return
        params = self.vary.get(token)
        if params is None:
            await chat.send_text(EXPIRED)
            return
        await self._generate(params, chat)

    async def _authorized(self, user_id: int, chat: Chat) -> bool:
        if self.allowed_user_id is None:
            await chat.send_text(
                f"👋 Your Telegram user ID is {user_id}.\n"
                f"Add TELEGRAM_ALLOWED_USER_ID={user_id} to .env and restart ComfyUI Studio."
            )
            return False
        if user_id != self.allowed_user_id:
            log.warning("Ignoring Telegram message from user %s", user_id)
            return False
        return True

    async def _generate(self, params: GenerationParams, chat: Chat) -> None:
        try:
            params.validate()
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        try:
            job = self.jobs.start(params)
        except BusyError:
            await chat.send_text(BUSY)
            return
        status = await chat.send_text("🎨 Generating…")
        last_edit = float("-inf")
        async for event in job.stream():
            if event["type"] == "progress":
                now = time.monotonic()
                if now - last_edit >= EDIT_INTERVAL:
                    last_edit = now
                    await chat.edit_text(status, f"🎨 Generating… step {event['step']}/{event['total']}")
            elif event["type"] == "error":
                await chat.edit_text(status, f"⚠️ {event['message']}")
            elif event["type"] == "done":
                await self._deliver(params, event, status, chat)

    async def _deliver(self, params: GenerationParams, event: dict[str, Any], status: int, chat: Chat) -> None:
        image = event.get("image")
        gallery = await self.galleries.get()
        if not image or gallery is None:
            await chat.edit_text(status, NOT_FOUND)
            return
        caption = caption_for(params, event["seed"], event["elapsed"])
        await chat.send_photo(gallery.path(image["name"]), caption, self.vary.put(params))
        await chat.delete(status)
