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
    async def runner(params, on_progress, on_preview=None):
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


def test_unset_allowed_user_replies_with_id(tmp_path, caplog) -> None:
    seen: list = []
    chat = FakeChat()
    with caplog.at_level("INFO", logger="web.bot_core"):
        asyncio.run(make_bot(tmp_path, done_runner(seen), allowed=None).handle_text(777, "a cat", chat))
    assert seen == [] and "777" in chat.texts()[0] and "TELEGRAM_ALLOWED_USER_ID=777" in chat.texts()[0]
    assert "TELEGRAM_ALLOWED_USER_ID=777" in caplog.text  # visible in the Studio window too


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
    async def slow_runner(params, on_progress, on_preview=None):
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
    async def failing_runner(params, on_progress, on_preview=None):
        raise ComfyUIUnavailableError("Cannot connect to ComfyUI. Is ComfyUI running?")

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, failing_runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"]
    assert chat.calls[1][2] == "⚠️ Cannot connect to ComfyUI. Is ComfyUI running?"


def test_missing_image_reported(tmp_path) -> None:
    async def runner(params, on_progress, on_preview=None):
        return {"image": None, "seed": 1, "elapsed": 1.0}

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"] and "couldn't be found" in chat.calls[1][2]
