"""Telegram bot behaviour without the Telegram library: access control, commands, models, styles, ×4, upscale,
/fix, /extend, /remove, /faces, /nobg and /bg.

The owner (TELEGRAM_ALLOWED_USER_ID) approves anyone else who wants to use the bot.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from io import BytesIO
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError
from models import INPAINT_KEY, SD15_SHAPES, ModelProfile, check_limits, default_profile, profile_by_key, profile_for_ckpt
from styles import STYLES, style_by_key
from ai_edits import ExtendLayout
from web.builders import (
    DEFAULT_EXTEND_AMOUNT,
    DEFAULT_STRENGTH,
    FALLBACK_LIMITS,
    SHARP_MAX_SIDE,
    UPSCALE_MAX_SIDE,
    build_background,
    build_extend,
    build_faces,
    build_generation,
    build_img2img,
    build_inpaint,
    build_remove,
    build_sharp_upscale,
    build_upscale,
    resolve_profile,
)
from web.jobs import (
    BackgroundRequest,
    BusyError,
    ExtendRequest,
    FacesRequest,
    Img2ImgRequest,
    InpaintRequest,
    JobManager,
    JobRequest,
    RemoveRequest,
    SharpUpscaleRequest,
    UpscaleRequest,
)
from web.scribble import REGIONS, find_scribble, region_mask
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
    "• send a photo with a caption (e.g. \"make it winter\") → edit that photo\n"
    "• draw on a photo with Telegram's pink pen, caption it /fix red beanie hat → change only that area\n"
    "• draw over something with the pink pen, caption it /remove → it disappears\n"
    "• send a photo captioned /extend → make it wider, taller or bigger all round\n"
    "• send a photo captioned /faces → clearer, more detailed faces\n"
    "• send a photo captioned /nobg → the subject cut out (a PNG with a see-through background)\n"
    "• send a photo captioned /bg white, /bg blur or /bg a sunny beach → a new background\n\n"
    "Under each image: 🔁 Vary · 🖼️ ×4 variations · 🔍 Upscale ×2 · ↔️ Extend · 🔎 Sharp ×4 · 😊 Faces · 🌄 Background."
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
FIX_HINT = "Add what should be there after /fix, e.g. /fix a red beanie hat"
REMOVE_HINT = (
    "I didn't find a pink scribble on the photo. Draw over what should go with Telegram's pink pen, "
    "then send it again with the caption /remove"
)
EXTEND_QUESTION = "Which way should I extend the picture?"
BACKGROUND_QUESTION = "What should be behind the subject? (For a new scene, send the photo captioned /bg a sunny beach.)"
BACKGROUND_CHOICES = {"transparent": "✂️ Nothing (cut out)", "white": "⬜ White", "black": "⬛ Black", "blur": "🌫️ Blur"}
EXTEND_CHOICES = {
    "w": ({"left", "right"}, "↔️ Wider"),
    "t": ({"top", "bottom"}, "↕️ Taller"),
    "a": ({"left", "top", "right", "bottom"}, "⛶ All sides"),
}
FIX_CANCELLED = "Cancelled. Nothing was changed."
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
    async def upscalers(self) -> list[str]: ...
    async def files(self, folder: str) -> list[str]: ...


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
    mask: Path | None = None  # painted area when the result came from /fix (inpainting) or /remove
    layout: ExtendLayout | None = None  # canvas when the result came from /extend
    tool: str = ""  # "extend" / "remove" for those results (🔁 Vary repeats the same edit)


class ActionStore:
    """Short tokens for inline buttons (Telegram callback data is limited to 64 bytes)."""

    def __init__(self, keep: int = ACTIONS_KEEP) -> None:
        self._items: OrderedDict[str, Action] = OrderedDict()
        self._keep = keep

    def put(
        self, params: GenerationParams, image_name: str | None, user_id: int,
        source: Path | None = None, strength: float = 0.0, mask: Path | None = None,
        layout: ExtendLayout | None = None, tool: str = "",
    ) -> str:
        token = uuid.uuid4().hex[:16]
        self._items[token] = Action(params, image_name, user_id, source, strength, mask, layout, tool)
        while len(self._items) > self._keep:
            self._items.popitem(last=False)
        return token

    def get(self, token: str, user_id: int) -> Action | None:
        action = self._items.get(token)
        return action if action is not None and action.user_id == user_id else None


@dataclass(frozen=True)
class PendingFix:
    """A /fix, /remove or /extend photo waiting for the person to confirm the marked area (or pick an option)."""

    source: Path
    mask: Path | None  # None: no scribble found (/fix: waiting for an area button), or /extend
    prompt: str
    user_id: int
    tool: str = "fix"


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
        self._fixes: OrderedDict[str, PendingFix] = OrderedDict()

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
        command = command_name(caption)
        if command in ("fix", "remove", "extend", "faces", "nobg", "bg"):
            source = self._keep_photo(data)
            if source is None:
                await chat.send_text("⚠️ That photo can't be read. Send it as a normal photo (JPEG or PNG).")
            elif command == "fix":
                await self._start_fix(parse_request(caption)[1], source, chat, user_id)
            elif command == "remove":
                await self._start_remove(source, chat, user_id)
            elif command == "faces":
                await self._faces(source, None, chat, user_id)
            elif command == "nobg":
                await self._background(source, "transparent", "", chat, user_id)
            elif command == "bg":
                choice = parse_request(caption)[1]
                mode = choice.lower() if choice.lower() in BACKGROUND_CHOICES else "prompt"
                await self._background(source, mode, "" if mode != "prompt" else choice, chat, user_id)
            else:
                await self._ask_extend(source, parse_request(caption)[1], chat, user_id)
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
        if kind in ("fixok", "fixno", "area", "rmok", "exd", "bgx"):
            await self._continue_fix(kind, value, chat, user_id)
        elif kind in ("allow", "deny", "remove"):
            if user_id == self.owner_id and value.isdigit():
                await self._decide(kind, int(value), chat)
        elif kind == "model":
            await self._choose_model(value, self.prefs_for(user_id), chat)
        elif kind == "style":
            await self._choose_style(value, self.prefs_for(user_id), chat)
        elif kind in ("vary", "x4", "up", "sharp", "ext", "faces", "bgm"):
            action = self.actions.get(value, user_id)
            if action is None:
                await chat.send_text(EXPIRED)
            elif kind == "vary":
                await self._run(self._again(action, batch=1), chat, user_id)
            elif kind == "x4":
                await self._four_more(action, chat, user_id)
            elif kind == "sharp":
                await self._sharp(action, chat, user_id)
            elif kind == "ext":
                await self._extend_result(action, chat, user_id)
            elif kind in ("faces", "bgm"):
                await self._result_tool(kind, action, chat, user_id)
            else:
                await self._upscale(action, chat, user_id)

    # --- /fix (inpainting from a photo) -----------------------------------------

    def _keep_photo(self, data: bytes) -> Path | None:
        try:
            return self.sources.path(self.sources.save(data))
        except InvalidParamsError:
            return None

    async def _has_inpainter(self, command: str, chat: Chat) -> bool:
        inpainter = profile_by_key(INPAINT_KEY)
        if inpainter in await self.catalog.available():
            return True
        await chat.send_text(f"⚠️ /{command} needs the {inpainter.label} model, which isn't installed in ComfyUI.")
        return False

    async def _start_fix(self, prompt: str, source: Path, chat: Chat, user_id: int) -> None:
        if not prompt:
            await chat.send_text(FIX_HINT)
            return
        if not await self._has_inpainter("fix", chat):
            return
        marked = find_scribble(source)
        if marked is None:
            token = self._put_fix(PendingFix(source, None, prompt, user_id))
            buttons = [(label, f"area:{token}:{key}") for key, label in REGIONS]
            await chat.send_text(
                "I didn't find a pink scribble on the photo. Which area should change?",
                [buttons[:3], buttons[3:] + [("❌ Cancel", f"fixno:{token}")]],
            )
            return
        mask = self.sources.path(self.sources.save(_png(marked)))
        preview = self.sources.path(self.sources.save(_highlight(source, marked)))
        token = self._put_fix(PendingFix(source, mask, prompt, user_id))
        await chat.send_photo(
            preview,
            f"I'll repaint the highlighted area to show: {prompt}\nEverything else stays exactly the same.",
            [[("✅ Fix this area", f"fixok:{token}"), ("❌ Cancel", f"fixno:{token}")]],
        )

    async def _start_remove(self, source: Path, chat: Chat, user_id: int) -> None:
        if not await self._has_inpainter("remove", chat):
            return
        marked = find_scribble(source)
        if marked is None:
            await chat.send_text(REMOVE_HINT)
            return
        mask = self.sources.path(self.sources.save(_png(marked)))
        preview = self.sources.path(self.sources.save(_highlight(source, marked)))
        token = self._put_fix(PendingFix(source, mask, "", user_id, tool="remove"))
        await chat.send_photo(
            preview,
            "I'll remove what's in the highlighted area and fill it in to match its surroundings.",
            [[("✅ Remove it", f"rmok:{token}"), ("❌ Cancel", f"fixno:{token}")]],
        )

    async def _ask_extend(self, source: Path, prompt: str, chat: Chat, user_id: int) -> None:
        if not await self._has_inpainter("extend", chat):
            return
        token = self._put_fix(PendingFix(source, None, prompt, user_id, tool="extend"))
        buttons = [(label, f"exd:{token}:{key}") for key, (_, label) in EXTEND_CHOICES.items()]
        await chat.send_text(EXTEND_QUESTION, [buttons, [("❌ Cancel", f"fixno:{token}")]])

    async def _gallery_file(self, action: Action, chat: Chat) -> Path | None:
        gallery = await self.galleries.get()
        path = gallery.path(action.image_name) if gallery is not None and action.image_name else None
        if path is None or not path.is_file():
            await chat.send_text("⚠️ That image isn't in the gallery any more.")
            return None
        return path

    async def _extend_result(self, action: Action, chat: Chat, user_id: int) -> None:
        path = await self._gallery_file(action, chat)
        if path is not None:
            await self._ask_extend(path, "", chat, user_id)

    async def _result_tool(self, kind: str, action: Action, chat: Chat, user_id: int) -> None:
        """😊 Faces runs straight away; 🌄 Background asks what should go behind the subject."""
        path = await self._gallery_file(action, chat)
        if path is None:
            return
        if kind == "faces":
            await self._faces(path, action.params.prompt if action.image_name else None, chat, user_id)
            return
        token = self._put_fix(PendingFix(path, None, "", user_id, tool="background"))
        buttons = [(label, f"bgx:{token}:{mode}") for mode, label in BACKGROUND_CHOICES.items()]
        await chat.send_text(BACKGROUND_QUESTION, [buttons[:2], buttons[2:], [("❌ Cancel", f"fixno:{token}")]])

    async def _faces(self, source: Path, prompt: str | None, chat: Chat, user_id: int) -> None:
        try:
            request = build_faces(
                source=source, parent_params={"prompt": prompt} if prompt else None,
                available=await self.catalog.available(), detectors=await self.catalog.files("detection"),
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat, user_id)

    async def _background(self, source: Path, mode: str, prompt: str, chat: Chat, user_id: int) -> None:
        try:
            request = build_background(
                source=source, mode=mode, prompt=prompt, available=await self.catalog.available(),
                removers=await self.catalog.files("background_removal"),
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat, user_id)

    async def _continue_fix(self, kind: str, value: str, chat: Chat, user_id: int) -> None:
        token, _, region = value.partition(":")
        pending = self._fixes.get(token)
        if pending is None or pending.user_id != user_id:
            await chat.send_text(EXPIRED)
            return
        del self._fixes[token]
        if kind == "fixno":
            await chat.send_text(FIX_CANCELLED)
            return
        if kind == "bgx":
            if region in BACKGROUND_CHOICES:
                await self._background(pending.source, region, "", chat, user_id)
            else:
                await chat.send_text(EXPIRED)
            return
        if kind in ("rmok", "exd"):
            await self._run_edit(kind, region, pending, chat, user_id)
            return
        mask = pending.mask
        if kind == "area":
            if region not in dict(REGIONS):
                await chat.send_text(EXPIRED)
                return
            with Image.open(pending.source) as img:
                mask = self.sources.path(self.sources.save(_png(region_mask(img.size, region))))
        if mask is None:
            await chat.send_text(EXPIRED)
            return
        try:
            request = build_inpaint(
                source=pending.source, mask=mask, prompt=pending.prompt, negative_prompt=DEFAULT_NEGATIVE,
                style=self.prefs_for(user_id).style_key, seed=None, available=await self.catalog.available(),
            )
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat, user_id)

    async def _run_edit(self, kind: str, choice: str, pending: PendingFix, chat: Chat, user_id: int) -> None:
        """Start a confirmed /remove (rmok) or the chosen /extend direction (exd)."""
        available = await self.catalog.available()
        try:
            if kind == "rmok" and pending.mask is not None:
                request: JobRequest = build_remove(source=pending.source, mask=pending.mask, available=available)
            elif kind == "exd" and choice in EXTEND_CHOICES:
                request = build_extend(
                    source=pending.source, sides=EXTEND_CHOICES[choice][0], amount=DEFAULT_EXTEND_AMOUNT,
                    prompt=pending.prompt, available=available,
                )
            else:
                await chat.send_text(EXPIRED)
                return
        except InvalidParamsError as exc:
            await chat.send_text(f"⚠️ {exc}")
            return
        await self._run(request, chat, user_id)

    def _put_fix(self, pending: PendingFix) -> str:
        token = uuid.uuid4().hex[:16]
        self._fixes[token] = pending
        while len(self._fixes) > ACTIONS_KEEP:
            self._fixes.popitem(last=False)
        return token

    # --- access --------------------------------------------------------------

    async def _admitted(self, user_id: int, who: str, chat: Chat) -> bool:
        if self.owner_id is None:
            log.info("Telegram setup: add TELEGRAM_ALLOWED_USER_ID=%s to .env to allow this user", user_id)
            await chat.send_text(
                f"👋 Your Telegram user ID is {user_id}.\n"
                f"Add TELEGRAM_ALLOWED_USER_ID={user_id} to .env and restart LUMOS Studios."
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
        rows = [[(("✓ " if p == current else "") + p.label, f"model:{p.key}")] for p in available if p.selectable]
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
    def _again(action: Action, batch: int) -> JobRequest:
        """The same settings with a new seed: from the same start image (and painted area) if there was one."""
        params = replace(action.params, seed=None, batch=batch)
        if action.source and action.tool == "extend" and action.layout:
            return ExtendRequest(action.source, action.layout, replace(params, batch=1))
        if action.source and action.tool == "remove" and action.mask:
            return RemoveRequest(action.source, action.mask, replace(params, batch=1))
        if action.source and action.tool == "faces":
            return FacesRequest(action.source, replace(params, batch=1))
        if action.source and action.tool.startswith("background:"):
            return BackgroundRequest(action.source, action.tool.split(":", 1)[1], replace(params, batch=1))
        if action.source and action.mask:
            return InpaintRequest(action.source, action.mask, replace(params, batch=1))
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

    async def _sharp(self, action: Action, chat: Chat, user_id: int) -> None:
        gallery = await self.galleries.get()
        try:
            if gallery is None or action.image_name is None:
                raise FileNotFoundError(action.image_name)
            image = gallery.get(action.image_name)
            request = build_sharp_upscale(gallery.path(action.image_name), image.params, await self.catalog.upscalers())
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
        elif isinstance(request, SharpUpscaleRequest):
            label = "🔎 Upscaling ×4"
        elif isinstance(request, ExtendRequest):
            label = "↔️ Extending the picture"
        elif isinstance(request, RemoveRequest):
            label = "🧽 Removing the marked area"
        elif isinstance(request, FacesRequest):
            label = "😊 Fixing faces"
        elif isinstance(request, BackgroundRequest):
            label = "🌄 Changing the background"
        elif isinstance(request, Img2ImgRequest):
            label = f"🎨 {self._model_label(request.params)} · editing your photo"
        elif isinstance(request, InpaintRequest):
            label = "🖌️ Fixing the marked area"
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
        if isinstance(request, UpscaleRequest | SharpUpscaleRequest):
            p, scale = request.params, 2 if isinstance(request, UpscaleRequest) else 4
            icon = "🔍" if scale == 2 else "🔎"
            await chat.send_document(paths[0], f"{icon} Upscaled ×{scale} · {p.width * scale}×{p.height * scale} · {event['elapsed']}s")
        elif isinstance(request, BackgroundRequest) and request.mode == "transparent":
            # as a file: Telegram photos lose the see-through background
            await chat.send_document(paths[0], f"✂️ Cut out · {images[0]['width']}×{images[0]['height']} · {event['elapsed']}s")
        elif isinstance(request, ExtendRequest | RemoveRequest | FacesRequest | BackgroundRequest):
            what = {
                ExtendRequest: "↔️ Extended", RemoveRequest: "🧽 Removed", FacesRequest: "😊 Faces fixed",
                BackgroundRequest: "🌄 New background",
            }[type(request)]
            caption = f"{what} · {images[0]['width']}×{images[0]['height']} · {event['elapsed']}s"
            await chat.send_photo(paths[0], caption, self._photo_buttons(request, images[0]["name"], user_id))
        else:
            params = request.params if isinstance(request, Img2ImgRequest | InpaintRequest) else request
            caption = caption_for(params, event["seed"], event["elapsed"], self._model_label(params))
            if len(paths) > 1:
                await chat.send_album(paths, caption)
                await chat.send_text("Upscale your favourite, or make 4 more:", self._picker_buttons(request, images, user_id))
            else:
                await chat.send_photo(paths[0], caption, self._photo_buttons(request, images[0]["name"], user_id))
        await chat.delete(status)

    def _reuse(self, request: JobRequest, image_name: str | None, user_id: int) -> str:
        if isinstance(request, ExtendRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, layout=request.layout, tool="extend")
        if isinstance(request, RemoveRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, mask=request.mask, tool="remove")
        if isinstance(request, FacesRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, tool="faces")
        if isinstance(request, BackgroundRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, tool=f"background:{request.mode}")
        if isinstance(request, InpaintRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, mask=request.mask)
        if isinstance(request, Img2ImgRequest):
            return self.actions.put(request.params, image_name, user_id, request.source, request.strength)
        return self.actions.put(request, image_name, user_id)

    def _photo_buttons(self, request: JobRequest, image_name: str, user_id: int) -> Buttons:
        edit = isinstance(request, InpaintRequest | ExtendRequest | RemoveRequest | FacesRequest | BackgroundRequest)
        params = request.params if isinstance(request, Img2ImgRequest) or edit else request
        profile = profile_for_ckpt(params.model or "") or FALLBACK_LIMITS
        small = max(params.width, params.height) <= UPSCALE_MAX_SIDE
        reuse = self._reuse(request, None, user_id)
        image = self.actions.put(params, image_name, user_id)
        row = [("🔁 Vary", f"vary:{reuse}")]
        if profile.max_batch >= 4 and small and not edit:
            row.append(("🖼️ ×4", f"x4:{reuse}"))
        if profile.upscale and small:
            row.append(("🔍 Upscale", f"up:{image}"))
        more = [("↔️ Extend", f"ext:{image}")]
        if max(params.width, params.height) <= SHARP_MAX_SIDE:
            more.append(("🔎 Sharp ×4", f"sharp:{image}"))
        looks = [("😊 Faces", f"faces:{image}"), ("🌄 Background", f"bgm:{image}")]
        return [row, more, looks]

    def _picker_buttons(self, request: GenerationParams | Img2ImgRequest, images: list[dict[str, Any]], user_id: int) -> Buttons:
        upscale_row = [(f"🔍 {n}", f"up:{self._reuse(request, image['name'], user_id)}") for n, image in enumerate(images, 1)]
        return [upscale_row, [("🖼️ ×4 again", f"x4:{self._reuse(request, None, user_id)}")]]

    @staticmethod
    def _model_label(params: GenerationParams) -> str:
        profile = profile_for_ckpt(params.model or "")
        return profile.label if profile else "Stable Diffusion"


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def _highlight(source: Path, mask: Image.Image) -> bytes:
    """The photo with the area that will change tinted amber, for the confirmation message."""
    with Image.open(source) as img:
        photo = img.convert("RGB")
    tint = Image.new("RGB", photo.size, (242, 181, 68))
    photo.paste(tint, mask=mask.convert("L").point(lambda v: 140 if v else 0))
    photo.thumbnail((1280, 1280))
    buffer = BytesIO()
    photo.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()
