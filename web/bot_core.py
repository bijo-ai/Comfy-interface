"""Telegram bot behaviour without the Telegram library: access control, commands, models, styles, ×4 and upscale."""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError
from models import SD15_SHAPES, ModelProfile, check_limits, profile_for_ckpt
from styles import STYLES, style_by_key
from web.builders import FALLBACK_LIMITS, UPSCALE_MAX_SIDE, build_generation, build_upscale, resolve_profile
from web.jobs import BusyError, JobManager, JobRequest, UpscaleRequest

log = logging.getLogger(__name__)

Buttons = list[list[tuple[str, str]]]  # rows of (label, callback data)

SHAPE_COMMANDS = ("square", "portrait", "landscape")
EDIT_INTERVAL = 1.5  # seconds between progress edits (Telegram rate-limits edits)
CAPTION_PROMPT_MAX = 850
ACTIONS_KEEP = 300
HELP = (
    "Send me a prompt and I'll paint it on your PC.\n\n"
    "• a beach at sunset → square image\n"
    "• /portrait an old fisherman → tall image\n"
    "• /landscape mountains at dawn → wide image\n"
    "• /model → choose the model\n"
    "• /style → choose a style\n\n"
    "Under each image: 🔁 Vary · 🖼️ ×4 variations · 🔍 Upscale ×2."
)
BUSY = "⏳ Busy with another image. Try again in a moment."
EXPIRED = "This button expired. Send the prompt again."
NOT_FOUND = "⚠️ The image was generated but couldn't be found in ComfyUI's output folder."


class Chat(Protocol):
    async def send_text(self, text: str, buttons: Buttons | None = None) -> int: ...
    async def edit_text(self, message_id: int, text: str) -> None: ...
    async def delete(self, message_id: int) -> None: ...
    async def send_photo(self, path: Path, caption: str, buttons: Buttons) -> None: ...
    async def send_album(self, paths: list[Path], caption: str) -> None: ...
    async def send_document(self, path: Path, caption: str) -> None: ...


class GalleryLookup(Protocol):
    async def get(self) -> Any: ...


class Catalog(Protocol):
    async def available(self) -> list[ModelProfile]: ...


def command_name(text: str) -> str | None:
    text = text.strip()
    if not text.startswith("/"):
        return None
    return text.split(None, 1)[0][1:].split("@", 1)[0].lower()


def parse_request(text: str) -> tuple[str | None, str]:
    """Return (shape, prompt). Shape is None for every command that isn't a shape."""
    command = command_name(text)
    if command is None:
        return "square", text.strip()
    parts = text.strip().split(None, 1)
    rest = parts[1].strip() if len(parts) > 1 else ""
    return (command if command in SHAPE_COMMANDS else None), rest


def caption_for(params: GenerationParams, seed: int, elapsed: float, model_label: str) -> str:
    prompt = params.prompt
    if len(prompt) > CAPTION_PROMPT_MAX:
        prompt = prompt[: CAPTION_PROMPT_MAX - 1] + "…"
    return f"{prompt}\n\nseed {seed} · {params.width}×{params.height} · {model_label} · {elapsed}s"


@dataclass(frozen=True)
class Action:
    params: GenerationParams  # settings to reuse (seed is replaced on use)
    image_name: str | None  # gallery image for 🔍 Upscale


class ActionStore:
    """Short tokens for inline buttons (Telegram callback data is limited to 64 bytes)."""

    def __init__(self, keep: int = ACTIONS_KEEP) -> None:
        self._items: OrderedDict[str, Action] = OrderedDict()
        self._keep = keep

    def put(self, params: GenerationParams, image_name: str | None) -> str:
        token = uuid.uuid4().hex[:16]
        self._items[token] = Action(params, image_name)
        while len(self._items) > self._keep:
            self._items.popitem(last=False)
        return token

    def get(self, token: str) -> Action | None:
        return self._items.get(token)


class StudioBot:
    def __init__(self, jobs: JobManager, galleries: GalleryLookup, allowed_user_id: int | None, catalog: Catalog) -> None:
        self.jobs = jobs
        self.galleries = galleries
        self.allowed_user_id = allowed_user_id
        self.catalog = catalog
        self.model_key: str | None = None  # None = Studio's default model
        self.style_key = "none"
        self.actions = ActionStore()

    # --- entry points ---------------------------------------------------------

    async def handle_text(self, user_id: int, text: str, chat: Chat) -> None:
        if not await self._authorized(user_id, chat):
            return
        command = command_name(text)
        if command == "model":
            await self._model_menu(chat)
            return
        if command == "style":
            await self._style_menu(chat)
            return
        shape, prompt = parse_request(text)
        if shape is None:
            await chat.send_text(f"{HELP}\n\nNow: {await self._settings_line()}")
        elif not prompt:
            await chat.send_text(f"Add a prompt after the command, e.g. /{shape} an old fisherman")
        else:
            await self._generate(shape, prompt, chat)

    async def handle_button(self, user_id: int, data: str, chat: Chat) -> None:
        if not await self._authorized(user_id, chat):
            return
        kind, _, value = data.partition(":")
        if kind == "model":
            await self._choose_model(value, chat)
        elif kind == "style":
            await self._choose_style(value, chat)
        elif kind in ("vary", "x4", "up"):
            action = self.actions.get(value)
            if action is None:
                await chat.send_text(EXPIRED)
            elif kind == "vary":
                await self._run(replace(action.params, seed=None, batch=1), chat)
            elif kind == "x4":
                await self._four_more(action.params, chat)
            else:
                await self._upscale(action, chat)

    # --- settings ------------------------------------------------------------

    async def _profile(self) -> ModelProfile | None:
        return resolve_profile(self.model_key, await self.catalog.available())

    async def _settings_line(self) -> str:
        try:
            profile = await self._profile()
        except InvalidParamsError:
            profile = None
        style = style_by_key(self.style_key)
        return f"{profile.label if profile else 'default model'} · {style.emoji} {style.label}"

    async def _model_menu(self, chat: Chat) -> None:
        available = await self.catalog.available()
        current = resolve_profile(self.model_key, available) if available else None
        rows = [[(("✓ " if p == current else "") + p.label, f"model:{p.key}")] for p in available]
        await chat.send_text("Choose a model:" if rows else "ComfyUI isn't reachable right now.", rows or None)

    async def _choose_model(self, key: str, chat: Chat) -> None:
        try:
            profile = resolve_profile(key, await self.catalog.available())
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        self.model_key = key
        note = " It's heavy: about 30–60 s per image, 1 at a time, no ×4 or upscale." if profile.heavy else ""
        await chat.send_text(f"Model: {profile.label}.{note}")

    async def _style_menu(self, chat: Chat) -> None:
        buttons = [(("✓ " if s.key == self.style_key else "") + f"{s.emoji} {s.label}", f"style:{s.key}") for s in STYLES]
        await chat.send_text("Choose a style:", [buttons[i : i + 2] for i in range(0, len(buttons), 2)])

    async def _choose_style(self, key: str, chat: Chat) -> None:
        try:
            style = style_by_key(key)
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        self.style_key = key
        await chat.send_text(f"Style: {style.emoji} {style.label}.")

    # --- jobs ----------------------------------------------------------------

    async def _generate(self, shape: str, prompt: str, chat: Chat) -> None:
        try:
            profile = await self._profile()
            width, height = (profile.shapes if profile else SD15_SHAPES)[shape]
            params = build_generation(
                prompt=prompt, negative_prompt=DEFAULT_NEGATIVE, width=width, height=height,
                steps=profile.steps if profile else 20, cfg=profile.cfg if profile else 8.0,
                seed=None, batch=1, style=self.style_key, profile=profile,
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(params, chat)

    async def _four_more(self, params: GenerationParams, chat: Chat) -> None:
        params = replace(params, seed=None, batch=4)
        try:
            check_limits(profile_for_ckpt(params.model or "") or FALLBACK_LIMITS, params.width, params.height, 4)
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(params, chat)

    async def _upscale(self, action: Action, chat: Chat) -> None:
        gallery = await self.galleries.get()
        try:
            if gallery is None or action.image_name is None:
                raise FileNotFoundError(action.image_name)
            image = gallery.get(action.image_name)
            request = build_upscale(image, gallery.path(action.image_name), await self.catalog.available())
        except (FileNotFoundError, ValueError):
            await chat.send_text("⚠️ That image isn't in the gallery any more.")
            return
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat)

    async def _run(self, request: JobRequest, chat: Chat) -> None:
        try:
            job = self.jobs.start(request)
        except BusyError:
            await chat.send_text(BUSY)
            return
        label = "🔍 Upscaling ×2" if isinstance(request, UpscaleRequest) else f"🎨 {self._model_label(request)}"
        status = await chat.send_text(f"{label}…")
        last_edit = float("-inf")
        async for event in job.stream():
            if event["type"] == "progress":
                now = time.monotonic()
                if now - last_edit >= EDIT_INTERVAL:
                    last_edit = now
                    await chat.edit_text(status, f"{label} · step {event['step']}/{event['total']}")
            elif event["type"] == "error":
                await chat.edit_text(status, f"⚠️ {event['message']}")
            elif event["type"] == "done":
                await self._deliver(request, event, status, chat)

    async def _deliver(self, request: JobRequest, event: dict[str, Any], status: int, chat: Chat) -> None:
        images = event.get("images") or ([event["image"]] if event.get("image") else [])
        gallery = await self.galleries.get()
        if not images or gallery is None:
            await chat.edit_text(status, NOT_FOUND)
            return
        paths = [gallery.path(image["name"]) for image in images]
        if isinstance(request, UpscaleRequest):
            p = request.params
            await chat.send_document(paths[0], f"🔍 Upscaled ×2 · {p.width * 2}×{p.height * 2} · {event['elapsed']}s")
        else:
            caption = caption_for(request, event["seed"], event["elapsed"], self._model_label(request))
            if len(paths) > 1:
                await chat.send_album(paths, caption)
                await chat.send_text("Upscale your favourite, or make 4 more:", self._picker_buttons(request, images))
            else:
                await chat.send_photo(paths[0], caption, self._photo_buttons(request, images[0]["name"]))
        await chat.delete(status)

    def _photo_buttons(self, params: GenerationParams, image_name: str) -> Buttons:
        profile = profile_for_ckpt(params.model or "") or FALLBACK_LIMITS
        small = max(params.width, params.height) <= UPSCALE_MAX_SIDE
        reuse = self.actions.put(params, None)
        row = [("🔁 Vary", f"vary:{reuse}")]
        if profile.max_batch >= 4 and small:
            row.append(("🖼️ ×4", f"x4:{reuse}"))
        if profile.upscale and small:
            row.append(("🔍 Upscale", f"up:{self.actions.put(params, image_name)}"))
        return [row]

    def _picker_buttons(self, params: GenerationParams, images: list[dict[str, Any]]) -> Buttons:
        upscale_row = [(f"🔍 {n}", f"up:{self.actions.put(params, image['name'])}") for n, image in enumerate(images, 1)]
        return [upscale_row, [("🖼️ ×4 again", f"x4:{self.actions.put(params, None)}")]]

    @staticmethod
    def _model_label(params: GenerationParams) -> str:
        profile = profile_for_ckpt(params.model or "")
        return profile.label if profile else "Stable Diffusion"

    async def _authorized(self, user_id: int, chat: Chat) -> bool:
        if self.allowed_user_id is None:
            log.info("Telegram setup: add TELEGRAM_ALLOWED_USER_ID=%s to .env to allow this user", user_id)
            await chat.send_text(
                f"👋 Your Telegram user ID is {user_id}.\n"
                f"Add TELEGRAM_ALLOWED_USER_ID={user_id} to .env and restart ComfyUI Studio."
            )
            return False
        if user_id != self.allowed_user_id:
            log.warning("Ignoring Telegram message from user %s", user_id)
            return False
        return True
