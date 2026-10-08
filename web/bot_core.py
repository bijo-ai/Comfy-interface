"""Telegram bot behaviour without the Telegram library: access control, commands, models, styles, ×4 and upscale.

The owner (TELEGRAM_ALLOWED_USER_ID) approves anyone else who wants to use the bot.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError
from models import SD15_SHAPES, ModelProfile, check_limits, default_profile, profile_for_ckpt
from styles import STYLES, style_by_key
from web.builders import (
    DEFAULT_STRENGTH,
    FALLBACK_LIMITS,
    UPSCALE_MAX_SIDE,
    build_generation,
    build_img2img,
    build_upscale,
    resolve_profile,
)
from web.jobs import BusyError, Img2ImgRequest, JobManager, JobRequest, UpscaleRequest
from web.sources import SourceStore
from web.users import UserStore

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
    "• /style → choose a style\n"
    "• send a photo with a caption (e.g. \"make it winter\") → edit that photo\n\n"
    "Under each image: 🔁 Vary · 🖼️ ×4 variations · 🔍 Upscale ×2."
)
OWNER_HELP = "\n• /users → see and remove the people you've let in"
BUSY = "⏳ Busy with another image. Try again in a moment."
EXPIRED = "This button expired. Send the prompt again."
NOT_FOUND = "⚠️ The image was generated but couldn't be found in ComfyUI's output folder."
REQUEST_SENT = "📨 This is a private bot. I've asked its owner to let you in; you'll get a message here when they decide."
PENDING = "⏳ Still waiting for the owner to approve you."
WELCOME = (
    "✅ You're in! The owner approved you.\n\n"
    "Images are made on the owner's PC and saved on it too.\n\n" + HELP.replace(" on your PC", "")
)
DENIED_NOTICE = "Sorry, the owner didn't approve access to this bot."
NEED_CAPTION = 'Add a caption to the photo describing the change, e.g. "make it winter" or "as an oil painting".'


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


@dataclass
class Prefs:
    model_key: str | None = None  # None = Studio's default model
    style_key: str = "none"


@dataclass(frozen=True)
class Action:
    params: GenerationParams  # settings to reuse (seed is replaced on use)
    image_name: str | None  # gallery image for 🔍 Upscale
    user_id: int  # buttons only work for the person they were sent to
    source: Path | None = None  # start image when the result came from image-to-image
    strength: float = 0.0


class ActionStore:
    """Short tokens for inline buttons (Telegram callback data is limited to 64 bytes)."""

    def __init__(self, keep: int = ACTIONS_KEEP) -> None:
        self._items: OrderedDict[str, Action] = OrderedDict()
        self._keep = keep

    def put(
        self, params: GenerationParams, image_name: str | None, user_id: int,
        source: Path | None = None, strength: float = 0.0,
    ) -> str:
        token = uuid.uuid4().hex[:16]
        self._items[token] = Action(params, image_name, user_id, source, strength)
        while len(self._items) > self._keep:
            self._items.popitem(last=False)
        return token

    def get(self, token: str, user_id: int) -> Action | None:
        action = self._items.get(token)
        return action if action is not None and action.user_id == user_id else None


class StudioBot:
    def __init__(
        self,
        jobs: JobManager,
        galleries: GalleryLookup,
        owner_id: int | None,
        catalog: Catalog,
        users: UserStore,
        chat_for: Callable[[int], Chat] | None = None,
        sources: SourceStore | None = None,
    ) -> None:
        self.sources = sources  # where Telegram photos are kept for image-to-image
        self.jobs = jobs
        self.galleries = galleries
        self.owner_id = owner_id
        self.catalog = catalog
        self.users = users
        self.chat_for = chat_for  # opens a chat with any user (set by the Telegram adapter)
        self.actions = ActionStore()
        self._prefs: dict[int, Prefs] = {}

    def prefs_for(self, user_id: int) -> Prefs:
        return self._prefs.setdefault(user_id, Prefs())

    # --- entry points ---------------------------------------------------------

    async def handle_text(self, user_id: int, text: str, chat: Chat, who: str = "") -> None:
        if not await self._admitted(user_id, who, chat):
            return
        prefs = self.prefs_for(user_id)
        command = command_name(text)
        if command == "model":
            await self._model_menu(prefs, chat)
        elif command == "style":
            await self._style_menu(prefs, chat)
        elif command == "users" and user_id == self.owner_id:
            await self._users_menu(chat)
        else:
            shape, prompt = parse_request(text)
            if shape is None:
                extra = OWNER_HELP if user_id == self.owner_id else ""
                await chat.send_text(f"{HELP}{extra}\n\nNow: {await self._settings_line(prefs)}")
            elif not prompt:
                await chat.send_text(f"Add a prompt after the command, e.g. /{shape} an old fisherman")
            else:
                await self._generate(shape, prompt, prefs, user_id, chat)

    async def handle_photo(self, user_id: int, data: bytes, caption: str, chat: Chat, who: str = "") -> None:
        """A photo with a caption: repaint the photo following the caption (image-to-image)."""
        if not await self._admitted(user_id, who, chat):
            return
        if not caption.strip():
            await chat.send_text(NEED_CAPTION)
            return
        if self.sources is None:
            await chat.send_text("⚠️ Photo editing isn't set up on this bot.")
            return
        prefs = self.prefs_for(user_id)
        try:
            source = self.sources.path(self.sources.save(data))
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        found, profile = await self._current_profile(prefs, chat)
        if not found:
            return
        try:
            request = build_img2img(
                source=source, prompt=caption.strip(), negative_prompt=DEFAULT_NEGATIVE,
                steps=profile.steps if profile else 20, cfg=profile.cfg if profile else 8.0,
                seed=None, batch=1, style=prefs.style_key, strength=DEFAULT_STRENGTH, profile=profile,
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat, user_id)

    async def handle_button(self, user_id: int, data: str, chat: Chat, who: str = "") -> None:
        if not await self._admitted(user_id, who, chat):
            return
        kind, _, value = data.partition(":")
        if kind in ("allow", "deny", "remove"):
            if user_id == self.owner_id and value.isdigit():
                await self._decide(kind, int(value), chat)
        elif kind == "model":
            await self._choose_model(value, self.prefs_for(user_id), chat)
        elif kind == "style":
            await self._choose_style(value, self.prefs_for(user_id), chat)
        elif kind in ("vary", "x4", "up"):
            action = self.actions.get(value, user_id)
            if action is None:
                await chat.send_text(EXPIRED)
            elif kind == "vary":
                await self._run(self._again(action, batch=1), chat, user_id)
            elif kind == "x4":
                await self._four_more(action, chat, user_id)
            else:
                await self._upscale(action, chat, user_id)

    # --- access --------------------------------------------------------------

    async def _admitted(self, user_id: int, who: str, chat: Chat) -> bool:
        if self.owner_id is None:
            log.info("Telegram setup: add TELEGRAM_ALLOWED_USER_ID=%s to .env to allow this user", user_id)
            await chat.send_text(
                f"👋 Your Telegram user ID is {user_id}.\n"
                f"Add TELEGRAM_ALLOWED_USER_ID={user_id} to .env and restart ComfyUI Studio."
            )
            return False
        if user_id == self.owner_id or self.users.is_allowed(user_id):
            return True
        if self.users.is_denied(user_id):
            log.info("Ignoring Telegram message from denied user %s", user_id)
        elif self.users.is_pending(user_id):
            await chat.send_text(PENDING)
        else:
            await self._ask_owner(user_id, who or f"user {user_id}", chat)
        return False

    async def _ask_owner(self, user_id: int, who: str, chat: Chat) -> None:
        self.users.add_pending(user_id, who)
        await chat.send_text(REQUEST_SENT)
        if self.chat_for is not None:
            buttons = [[("✅ Allow", f"allow:{user_id}"), ("🚫 Deny", f"deny:{user_id}")]]
            await self.chat_for(self.owner_id).send_text(f"👤 {who} (id {user_id}) wants to use your bot.", buttons)

    async def _decide(self, kind: str, target: int, chat: Chat) -> None:
        name = self.users.name_of(target)
        if kind == "allow":
            self.users.allow(target)
            await chat.send_text(f"✅ {name} can use the bot now. Remove them any time with /users.")
            await self._tell(target, WELCOME)
        elif kind == "deny":
            self.users.deny(target)
            await chat.send_text(f"🚫 {name} won't be able to use the bot.")
            await self._tell(target, DENIED_NOTICE)
        else:
            self.users.remove(target)
            await chat.send_text(f"Removed {name}. They'd have to ask again to use the bot.")

    async def _tell(self, user_id: int, text: str) -> None:
        if self.chat_for is not None:
            await self.chat_for(user_id).send_text(text)

    async def _users_menu(self, chat: Chat) -> None:
        people = self.users.allowed_users()
        if not people:
            await chat.send_text("Nobody else can use the bot yet. When someone messages it, you'll get Allow / Deny buttons.")
            return
        lines = "\n".join(f"• {name}" for _, name in people)
        buttons = [[(f"Remove {name}", f"remove:{uid}")] for uid, name in people]
        await chat.send_text(f"People who can use your bot:\n{lines}", buttons)

    # --- settings ------------------------------------------------------------

    async def _profile(self, prefs: Prefs) -> ModelProfile | None:
        return resolve_profile(prefs.model_key, await self.catalog.available())

    async def _settings_line(self, prefs: Prefs) -> str:
        try:
            profile = await self._profile(prefs)
        except InvalidParamsError:
            profile = None
        style = style_by_key(prefs.style_key)
        return f"{profile.label if profile else 'default model'} · {style.emoji} {style.label}"

    async def _model_menu(self, prefs: Prefs, chat: Chat) -> None:
        available = await self.catalog.available()
        try:
            current = resolve_profile(prefs.model_key, available) if available else None
        except InvalidParamsError:  # the chosen model was removed from ComfyUI
            current = default_profile(available)
        rows = [[(("✓ " if p == current else "") + p.label, f"model:{p.key}")] for p in available]
        await chat.send_text("Choose a model:" if rows else "ComfyUI isn't reachable right now.", rows or None)

    async def _choose_model(self, key: str, prefs: Prefs, chat: Chat) -> None:
        try:
            profile = resolve_profile(key, await self.catalog.available())
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        prefs.model_key = key
        note = " It's heavy: about 30–60 s per image, 1 at a time, no ×4 or upscale." if profile.heavy else ""
        await chat.send_text(f"Model: {profile.label}.{note}")

    async def _style_menu(self, prefs: Prefs, chat: Chat) -> None:
        buttons = [(("✓ " if s.key == prefs.style_key else "") + f"{s.emoji} {s.label}", f"style:{s.key}") for s in STYLES]
        await chat.send_text("Choose a style:", [buttons[i : i + 2] for i in range(0, len(buttons), 2)])

    async def _choose_style(self, key: str, prefs: Prefs, chat: Chat) -> None:
        try:
            style = style_by_key(key)
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        prefs.style_key = key
        await chat.send_text(f"Style: {style.emoji} {style.label}.")

    # --- jobs ----------------------------------------------------------------

    async def _current_profile(self, prefs: Prefs, chat: Chat) -> tuple[bool, ModelProfile | None]:
        """The person's model; if it was removed from ComfyUI, fall back to the default and say so."""
        try:
            return True, await self._profile(prefs)
        except InvalidParamsError as exc:
            if prefs.model_key is None:
                await chat.send_text(f"⚠️ {exc}")
                return False, None
            prefs.model_key = None  # fall back instead of getting stuck
            try:
                profile = await self._profile(prefs)
            except InvalidParamsError as again:
                await chat.send_text(f"⚠️ {again}")
                return False, None
            await chat.send_text(f"⚠️ {exc} I switched to {profile.label if profile else 'the default model'}.")
            return True, profile

    async def _generate(self, shape: str, prompt: str, prefs: Prefs, user_id: int, chat: Chat) -> None:
        found, profile = await self._current_profile(prefs, chat)
        if not found:
            return
        try:
            width, height = (profile.shapes if profile else SD15_SHAPES)[shape]
            params = build_generation(
                prompt=prompt, negative_prompt=DEFAULT_NEGATIVE, width=width, height=height,
                steps=profile.steps if profile else 20, cfg=profile.cfg if profile else 8.0,
                seed=None, batch=1, style=prefs.style_key, profile=profile,
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(params, chat, user_id)

    @staticmethod
    def _again(action: Action, batch: int) -> GenerationParams | Img2ImgRequest:
        """The same settings with a new seed: from the same start image if there was one."""
        params = replace(action.params, seed=None, batch=batch)
        return Img2ImgRequest(action.source, params, action.strength) if action.source else params

    async def _four_more(self, action: Action, chat: Chat, user_id: int) -> None:
        params = action.params
        try:
            check_limits(profile_for_ckpt(params.model or "") or FALLBACK_LIMITS, params.width, params.height, 4)
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(self._again(action, batch=4), chat, user_id)

    async def _upscale(self, action: Action, chat: Chat, user_id: int) -> None:
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
        await self._run(request, chat, user_id)

    async def _run(self, request: JobRequest, chat: Chat, user_id: int) -> None:
        try:
            job = self.jobs.start(request)
        except BusyError:
            await chat.send_text(BUSY)
            return
        if isinstance(request, UpscaleRequest):
            label = "🔍 Upscaling ×2"
        elif isinstance(request, Img2ImgRequest):
            label = f"🎨 {self._model_label(request.params)} · editing your photo"
        else:
            label = f"🎨 {self._model_label(request)}"
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
                await self._deliver(request, event, status, chat, user_id)

    async def _deliver(self, request: JobRequest, event: dict[str, Any], status: int, chat: Chat, user_id: int) -> None:
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
            params = request.params if isinstance(request, Img2ImgRequest) else request
            caption = caption_for(params, event["seed"], event["elapsed"], self._model_label(params))
            if len(paths) > 1:
                await chat.send_album(paths, caption)
                await chat.send_text("Upscale your favourite, or make 4 more:", self._picker_buttons(request, images, user_id))
            else:
                await chat.send_photo(paths[0], caption, self._photo_buttons(request, images[0]["name"], user_id))
        await chat.delete(status)

    def _reuse(self, request: GenerationParams | Img2ImgRequest, image_name: str | None, user_id: int) -> str:
        if isinstance(request, Img2ImgRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, request.strength)
        return self.actions.put(request, image_name, user_id)

    def _photo_buttons(self, request: GenerationParams | Img2ImgRequest, image_name: str, user_id: int) -> Buttons:
        params = request.params if isinstance(request, Img2ImgRequest) else request
        profile = profile_for_ckpt(params.model or "") or FALLBACK_LIMITS
        small = max(params.width, params.height) <= UPSCALE_MAX_SIDE
        reuse = self._reuse(request, None, user_id)
        row = [("🔁 Vary", f"vary:{reuse}")]
        if profile.max_batch >= 4 and small:
            row.append(("🖼️ ×4", f"x4:{reuse}"))
        if profile.upscale and small:
            row.append(("🔍 Upscale", f"up:{self.actions.put(params, image_name, user_id)}"))
        return [row]

    def _picker_buttons(self, request: GenerationParams | Img2ImgRequest, images: list[dict[str, Any]], user_id: int) -> Buttons:
        upscale_row = [(f"🔍 {n}", f"up:{self._reuse(request, image['name'], user_id)}") for n, image in enumerate(images, 1)]
        return [upscale_row, [("🖼️ ×4 again", f"x4:{self._reuse(request, None, user_id)}")]]

    @staticmethod
    def _model_label(params: GenerationParams) -> str:
        profile = profile_for_ckpt(params.model or "")
        return profile.label if profile else "Stable Diffusion"
