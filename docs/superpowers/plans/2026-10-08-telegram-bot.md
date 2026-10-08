# Telegram Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** @lumoscomfybot generates images through the local ComfyUI from Telegram (plain/`/portrait`/`/landscape`, live progress, 🔁 Vary), locked to one user.

**Architecture:**
- **Core:** library-free bot behaviour in `web/bot_core.py`, driving the existing `JobManager` through a small `Chat` protocol.
- **Adapter:** the python-telegram-bot adapter and a resilient `BotRunner` live in `web/bot.py`.
- **Startup:** the FastAPI lifespan in `web/app.py` starts the runner when `TELEGRAM_BOT_TOKEN` is set.

**Tech Stack:** Python 3.12, python-telegram-bot 22.8, truststore 0.10, FastAPI lifespan, pytest.

**Spec:** `docs/superpowers/specs/2026-10-08-telegram-bot-design.md`

## Global Constraints
- Shapes: square 512×512, portrait 512×768, landscape 768×512.
- Progress edits at most every 1.5 s. Caption prompt is truncated to 900 chars (Telegram's caption limit is 1024).
- Only `TELEGRAM_ALLOWED_USER_ID` is served. Unset → reply with the sender's ID only.
- The token is never logged: the `httpx` logger is set to WARNING. The token lives only in `.env`.
- Retry startup every 30 s while offline. `InvalidToken` is not retried. Pending updates are dropped on start.
- Website behaviour is unchanged when no token is set. All 42 existing tests stay green.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. **Telegram unreachable at startup** (laptop boots before Wi-Fi): the website must work, and the bot connects later by itself. → Task 2 `test_runner_retries_after_network_error`.
2. **Website busy when a Telegram prompt arrives:** clear "Busy" reply, no second generation. → Task 1 `test_busy_reply`.
3. **A stranger finds the bot:** gets nothing, and no generation. → Task 1 `test_other_user_ignored`.
4. **Command with the bot's @username** (`/portrait@lumoscomfybot cat`) or a newline after the command → parsed correctly. → Task 1 `test_parse_request`.
5. **Token leakage via logs:** httpx INFO lines include the token URL; the log level must be raised. → Task 2 Step 5 (check console output contains no token).

---

### Task 1: Settings + bot core

**Files:** Modify `config.py`, `.env.example`. Create `web/bot_core.py`. Test `tests/test_bot_core.py`.

**Interfaces:**
- Produces:
  - `Settings.telegram_bot_token: str | None = None` and `Settings.telegram_allowed_user_id: int | None = None`.
  - `parse_request(text) -> tuple[str | None, str]`, `caption_for(params, seed, elapsed) -> str`.
  - `VaryStore.put(params) -> str` and `VaryStore.get(token) -> GenerationParams | None`.
  - `Chat` protocol: `send_text(text) -> int`, `edit_text(message_id, text)`, `delete(message_id)`, `send_photo(path, caption, vary_token)`.
  - `StudioBot(jobs: JobManager, galleries, allowed_user_id)` with `.handle_text(user_id, text, chat)` and `.handle_vary(user_id, token, chat)`.
  - Constants: `HELP`, `BUSY`, `EXPIRED`, `SHAPES`, `EDIT_INTERVAL`.
- Consumes: `JobManager`, `BusyError` (web/jobs.py); `GenerationParams`, `InvalidParamsError` (comfy_client); `GalleryProvider.get()` → `Gallery.path(name)` (web/app.py, web/gallery.py).

- [ ] **Step 1: Failing tests** — `tests/test_bot_core.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from comfy_client import ComfyUIUnavailableError, GenerationParams
from config import _int_or_none
from web import bot_core
from web.bot_core import BUSY, EXPIRED, HELP, StudioBot, caption_for, parse_request
from web.gallery import Gallery
from web.jobs import JobManager

USER = 42


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._next_id = 100

    async def send_text(self, text: str) -> int:
        self._next_id += 1
        self.calls.append(("send_text", text, self._next_id))
        return self._next_id

    async def edit_text(self, message_id: int, text: str) -> None:
        self.calls.append(("edit_text", message_id, text))

    async def delete(self, message_id: int) -> None:
        self.calls.append(("delete", message_id))

    async def send_photo(self, path: Path, caption: str, vary_token: str) -> None:
        self.calls.append(("send_photo", path, caption, vary_token))

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def texts(self) -> list[str]:
        return [call[1] for call in self.calls if call[0] == "send_text"]


class FakeGalleries:
    def __init__(self, gallery: Gallery) -> None:
        self.gallery = gallery

    async def get(self) -> Gallery:
        return self.gallery


def make_bot(tmp_path: Path, runner, allowed: int | None = USER) -> StudioBot:
    return StudioBot(JobManager(runner), FakeGalleries(Gallery(tmp_path, tmp_path / "cache")), allowed)


def done_runner(seen: list[GenerationParams]):
    async def runner(params, on_progress):
        seen.append(params)
        for step in range(1, 21):
            on_progress(step, 20)
        return {"image": {"name": "a.png"}, "seed": 1234, "elapsed": 5.5}

    return runner


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a beach", ("square", "a beach")),
        ("  a beach  ", ("square", "a beach")),
        ("/portrait an old man", ("portrait", "an old man")),
        ("/portrait@lumoscomfybot an old man", ("portrait", "an old man")),
        ("/landscape\nmountains", ("landscape", "mountains")),
        ("/LANDSCAPE hills", ("landscape", "hills")),
        ("/portrait", ("portrait", "")),
        ("/start", (None, "")),
        ("/help me", (None, "me")),
    ],
)
def test_parse_request(text, expected) -> None:
    assert parse_request(text) == expected


def test_int_or_none() -> None:
    assert _int_or_none(" 123 ") == 123
    assert _int_or_none("") is None
    assert _int_or_none("abc") is None


def test_caption_truncates_long_prompts() -> None:
    caption = caption_for(GenerationParams(prompt="x" * 2000, width=512, height=768), 7, 3.2)
    assert len(caption) < 1024
    assert caption.endswith("seed 7 · 512×768 · 3.2s")


def test_unset_allowed_user_replies_with_id(tmp_path) -> None:
    seen: list = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen), allowed=None).handle_text(777, "a cat", chat))
    assert seen == [] and "777" in chat.texts()[0] and "TELEGRAM_ALLOWED_USER_ID=777" in chat.texts()[0]


def test_other_user_ignored(tmp_path) -> None:
    seen: list = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen)).handle_text(999, "a cat", chat))
    assert seen == [] and chat.calls == []


def test_help_commands(tmp_path) -> None:
    for text in ("/start", "/help"):
        chat = FakeChat()
        asyncio.run(make_bot(tmp_path, done_runner([])).handle_text(USER, text, chat))
        assert chat.texts() == [HELP]


def test_command_without_prompt_gets_hint(tmp_path) -> None:
    seen: list = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen)).handle_text(USER, "/portrait", chat))
    assert seen == [] and "/portrait an old fisherman" in chat.texts()[0]


def test_plain_text_generates_square_and_sends_photo(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(bot_core, "EDIT_INTERVAL", 60)  # only the first progress edit may happen
    seen: list[GenerationParams] = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen)).handle_text(USER, "a beach", chat))
    assert [(p.prompt, p.width, p.height) for p in seen] == [("a beach", 512, 512)]
    assert chat.kinds() == ["send_text", "edit_text", "send_photo", "delete"]
    status_id = chat.calls[0][2]
    assert chat.calls[1] == ("edit_text", status_id, "🎨 Generating… step 1/20")
    _, path, caption, token = chat.calls[2]
    assert path == (tmp_path / "a.png").resolve()
    assert caption == "a beach\n\nseed 1234 · 512×512 · 5.5s"
    assert token
    assert chat.calls[3] == ("delete", status_id)


def test_portrait_size(tmp_path) -> None:
    seen: list[GenerationParams] = []
    asyncio.run(make_bot(tmp_path, done_runner(seen)).handle_text(USER, "/portrait a man", FakeChat()))
    assert (seen[0].width, seen[0].height) == (512, 768)


def test_vary_reuses_prompt_with_new_seed(tmp_path) -> None:
    seen: list[GenerationParams] = []
    bot = make_bot(tmp_path, done_runner(seen))
    chat = FakeChat()

    async def scenario():
        await bot.handle_text(USER, "/landscape hills", chat)
        token = chat.calls[[c[0] for c in chat.calls].index("send_photo")][3]
        await bot.handle_vary(USER, token, chat)

    asyncio.run(scenario())
    assert [(p.prompt, p.width, p.height, p.seed) for p in seen] == [("hills", 768, 512, None)] * 2


def test_expired_vary(tmp_path) -> None:
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner([])).handle_vary(USER, "nope", chat))
    assert chat.texts() == [EXPIRED]


def test_busy_reply(tmp_path) -> None:
    async def slow_runner(params, on_progress):
        await asyncio.sleep(0.2)
        return {"image": None, "seed": 1, "elapsed": 0.2}

    async def scenario():
        jobs = JobManager(slow_runner)
        jobs.start(GenerationParams(prompt="from the website"))
        bot = StudioBot(jobs, FakeGalleries(Gallery(tmp_path, tmp_path / "c")), USER)
        chat = FakeChat()
        await bot.handle_text(USER, "a cat", chat)
        return chat

    assert asyncio.run(scenario()).texts() == [BUSY]


def test_error_event_edits_status(tmp_path) -> None:
    async def failing_runner(params, on_progress):
        raise ComfyUIUnavailableError("Cannot connect to ComfyUI. Is ComfyUI running?")

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, failing_runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"]
    assert chat.calls[1][2] == "⚠️ Cannot connect to ComfyUI. Is ComfyUI running?"


def test_missing_image_reported(tmp_path) -> None:
    async def runner(params, on_progress):
        return {"image": None, "seed": 1, "elapsed": 1.0}

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"] and "couldn't be found" in chat.calls[1][2]
```

- [ ] **Step 2: Run** `uv run pytest tests/test_bot_core.py -q` → FAIL (`cannot import name '_int_or_none'` / no `web.bot_core`).

- [ ] **Step 3: Implement.**

`config.py`: add the two fields with defaults after `cache_dir`:
```python
    telegram_bot_token: str | None = None
    telegram_allowed_user_id: int | None = None
```
Add a helper:
```python
def _int_or_none(raw: str) -> int | None:
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None
```
Add to `load_settings()`:
```python
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip() or None,
        telegram_allowed_user_id=_int_or_none(os.getenv("TELEGRAM_ALLOWED_USER_ID", "")),
```
Append to `.env.example`:
```
# Optional: Telegram bot (token from @BotFather; your numeric user ID - the bot tells you it on first message)
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_USER_ID=
```

`web/bot_core.py`:
```python
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
```

- [ ] **Step 4: Run** `uv run pytest -q` → all PASS (42 + new).
- [ ] **Step 5: Commit** `feat: Telegram bot core — commands, access control, progress, Vary`.

---

### Task 2: python-telegram-bot adapter, lifecycle, startup wiring

**Files:** Create `web/bot.py`. Modify `web/app.py`, `web/__main__.py`, `pyproject.toml`/`uv.lock`. Test `tests/test_bot_runner.py`.

**Interfaces:**
- Consumes: `StudioBot`, `Chat` (Task 1); `Settings.telegram_*` (Task 1).
- Produces:
  - `TelegramChat(bot, chat_id)`.
  - `build_application(token, studio) -> Application`.
  - `BotRunner(factory, retry_seconds=30)` with `.start()` and `async .stop()`.
  - `create_app` starts and stops the runner in its lifespan when `settings.telegram_bot_token` is set (module-level name `BotRunner` in `web.app`, so tests can patch it).

- [ ] **Step 1: Deps** — `UV_SYSTEM_CERTS=1 uv add python-telegram-bot truststore`.

- [ ] **Step 2: Failing tests** — `tests/test_bot_runner.py`:

```python
from __future__ import annotations

import asyncio
from dataclasses import replace

from fastapi.testclient import TestClient
from telegram.error import InvalidToken, NetworkError

from web import app as app_module
from web.bot import BotRunner


class FakeUpdater:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def start_polling(self, **kwargs) -> None:
        self.log.append(f"poll drop={kwargs.get('drop_pending_updates')}")

    async def stop(self) -> None:
        self.log.append("updater.stop")


class FakeApp:
    def __init__(self, log: list[str], fail_with: Exception | None = None) -> None:
        self.log = log
        self.fail_with = fail_with
        self.updater = FakeUpdater(log)

    async def initialize(self) -> None:
        self.log.append("initialize")
        if self.fail_with:
            raise self.fail_with

    async def start(self) -> None:
        self.log.append("start")

    async def stop(self) -> None:
        self.log.append("stop")

    async def shutdown(self) -> None:
        self.log.append("shutdown")


def test_runner_retries_after_network_error() -> None:
    log: list[str] = []
    apps = iter([FakeApp(log, NetworkError("offline")), FakeApp(log)])

    async def scenario():
        runner = BotRunner(lambda: next(apps), retry_seconds=0)
        runner.start()
        await asyncio.wait_for(runner.wait_started(), 2)
        await runner.stop()

    asyncio.run(scenario())
    assert log == ["initialize", "shutdown", "initialize", "start", "poll drop=True",
                   "updater.stop", "stop", "shutdown"]


def test_runner_gives_up_on_invalid_token() -> None:
    log: list[str] = []
    calls = []

    def factory():
        calls.append(1)
        return FakeApp(log, InvalidToken("bad"))

    async def scenario():
        runner = BotRunner(factory, retry_seconds=0)
        runner.start()
        await asyncio.sleep(0.05)
        await runner.stop()

    asyncio.run(scenario())
    assert len(calls) == 1 and log == ["initialize", "shutdown"]


def test_app_starts_bot_only_with_token(settings, monkeypatch) -> None:
    events: list[str] = []

    class FakeRunner:
        def __init__(self, factory, retry_seconds=30) -> None:
            events.append("created")

        def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr(app_module, "BotRunner", FakeRunner)
    with TestClient(app_module.create_app(settings), base_url="http://127.0.0.1"):
        pass
    assert events == []
    with TestClient(app_module.create_app(replace(settings, telegram_bot_token="1:x")), base_url="http://127.0.0.1"):
        assert events == ["created", "start"]
    assert events == ["created", "start", "stop"]
```

- [ ] **Step 3: Run** `uv run pytest tests/test_bot_runner.py -q` → FAIL (`No module named 'web.bot'`).

- [ ] **Step 4: Implement** — `web/bot.py`:

```python
"""python-telegram-bot wiring for StudioBot, plus a start/stop lifecycle that survives being offline."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import InvalidToken, TelegramError
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from web.bot_core import StudioBot

log = logging.getLogger(__name__)

RETRY_SECONDS = 30
VARY_PREFIX = "vary:"


class TelegramChat:
    """`Chat` implementation that talks to one Telegram chat."""

    def __init__(self, bot: Any, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send_text(self, text: str) -> int:
        return (await self._bot.send_message(self._chat_id, text)).message_id

    async def edit_text(self, message_id: int, text: str) -> None:
        try:
            await self._bot.edit_message_text(text, chat_id=self._chat_id, message_id=message_id)
        except TelegramError as exc:  # e.g. "message is not modified"
            log.debug("Telegram edit failed: %s", exc)

    async def delete(self, message_id: int) -> None:
        try:
            await self._bot.delete_message(self._chat_id, message_id)
        except TelegramError as exc:
            log.debug("Telegram delete failed: %s", exc)

    async def send_photo(self, path: Path, caption: str, vary_token: str) -> None:
        markup = InlineKeyboardMarkup([[InlineKeyboardButton("🔁 Vary", callback_data=VARY_PREFIX + vary_token)]])
        with path.open("rb") as photo:
            await self._bot.send_photo(self._chat_id, photo, caption=caption, reply_markup=markup)


def build_application(token: str, studio: StudioBot) -> Application:
    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None or not message.text:
            return
        await studio.handle_text(user.id, message.text, TelegramChat(context.bot, message.chat_id))

    async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        if query is None or not query.data or query.message is None:
            return
        await query.answer()
        if query.data.startswith(VARY_PREFIX):
            token = query.data[len(VARY_PREFIX):]
            await studio.handle_vary(query.from_user.id, token, TelegramChat(context.bot, query.message.chat_id))

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Telegram handler failed", exc_info=context.error)

    application = Application.builder().token(token).build()
    application.add_handler(MessageHandler(filters.TEXT, on_text))
    application.add_handler(CallbackQueryHandler(on_button))
    application.add_error_handler(on_error)
    return application


class BotRunner:
    """Starts polling in the background, retrying while offline. Never takes the website down."""

    def __init__(self, factory: Callable[[], Any], retry_seconds: float = RETRY_SECONDS) -> None:
        self._factory = factory
        self._retry_seconds = retry_seconds
        self._app: Any = None
        self._task: asyncio.Task[None] | None = None
        self._started = asyncio.Event()

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def wait_started(self) -> None:
        await self._started.wait()

    async def _run(self) -> None:
        while True:
            app = self._factory()
            try:
                await app.initialize()
                await app.start()
                await app.updater.start_polling(drop_pending_updates=True)
            except InvalidToken:
                log.error("Telegram rejected TELEGRAM_BOT_TOKEN; the bot is off. Check the token in .env.")
                await _quiet_shutdown(app)
                return
            except Exception as exc:  # offline, Telegram down, ...
                log.warning("Telegram bot could not connect (%s); retrying in %ss", exc, self._retry_seconds)
                await _quiet_shutdown(app)
                await asyncio.sleep(self._retry_seconds)
                continue
            self._app = app
            self._started.set()
            log.info("Telegram bot connected")
            return

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._app is not None:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None


async def _quiet_shutdown(app: Any) -> None:
    with contextlib.suppress(Exception):
        await app.shutdown()
```

`web/app.py`:
- **Imports:** add `from contextlib import asynccontextmanager` (beside `contextmanager`), `from collections.abc import AsyncIterator` (already imported), `from web.bot import BotRunner, build_application`, and `from web.bot_core import StudioBot`.
- **In `create_app`, after `jobs = ...`,** add:
```python
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        runner = None
        if settings.telegram_bot_token:
            studio = StudioBot(jobs, galleries, settings.telegram_allowed_user_id)
            runner = BotRunner(lambda: build_application(settings.telegram_bot_token, studio))
            runner.start()
        yield
        if runner is not None:
            await runner.stop()
```
- **Then** pass `lifespan=lifespan` to `FastAPI(...)`.

`web/__main__.py`:
- **TLS:** first lines of `main()` are `import truststore` and `truststore.inject_into_ssl()`, so AVG's HTTPS scanning doesn't break Telegram.
- **Logging:** after `basicConfig`, add `logging.getLogger("httpx").setLevel(logging.WARNING)` (its INFO lines contain the bot token).
- **Console line:** load the settings once with `settings = load_settings()`, pass them to `create_app(settings)`, and print `Telegram bot: on (@ID locked)`, `on (waiting for your user ID)` or `off (no TELEGRAM_BOT_TOKEN)`.

- [ ] **Step 5: Run** `uv run pytest -q` → all PASS.
- [ ] **Step 6: Live check.** Start `uv run python -m web --no-browser` with the real `.env` token in the background, wait 10 s, and confirm the console contains "Telegram bot connected" and does **not** contain the token. Stop it.
- [ ] **Step 7: Commit** `feat: run the Telegram bot inside Studio with resilient startup`.

---

### Task 3: Docs, live end-to-end, release
- [ ] README: a "Telegram bot" section (create the bot with BotFather, `.env` keys, first-message ID flow, commands, laptop must be on).
- [ ] Restart Studio from `start_studio.bat`. Bijo sends `/start`, the ID gets written to `.env`, and Studio restarts. Bijo sends a prompt, and the image arrives and appears in the gallery.
- [ ] Full suite green; final review; push.
