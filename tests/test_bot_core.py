from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from comfy_client import ComfyUIUnavailableError, GenerationParams
from config import _int_or_none
from models import available_profiles
from tests.conftest import graph_for
from web import bot_core
from web.bot_core import BUSY, EXPIRED, HELP, PENDING, REQUEST_SENT, WELCOME, StudioBot, caption_for, parse_request
from web.users import UserStore
from web.gallery import Gallery
from web.jobs import JobManager, UpscaleRequest

USER = 42
DREAM = "DreamShaper_8_pruned.safetensors"
SDXL = "sd_xl_base_1.0.safetensors"
SD15 = "v1-5-pruned-emaonly.safetensors"


class FakeChat:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self._next_id = 100

    async def send_text(self, text: str, buttons=None) -> int:
        self._next_id += 1
        self.calls.append(("send_text", text, self._next_id, buttons))
        return self._next_id

    async def edit_text(self, message_id: int, text: str) -> None:
        self.calls.append(("edit_text", message_id, text))

    async def delete(self, message_id: int) -> None:
        self.calls.append(("delete", message_id))

    async def send_photo(self, path: Path, caption: str, buttons) -> None:
        self.calls.append(("send_photo", path, caption, buttons))

    async def send_album(self, paths: list[Path], caption: str) -> None:
        self.calls.append(("send_album", paths, caption))

    async def send_document(self, path: Path, caption: str) -> None:
        self.calls.append(("send_document", path, caption))

    def kinds(self) -> list[str]:
        return [call[0] for call in self.calls]

    def texts(self) -> list[str]:
        return [call[1] for call in self.calls if call[0] == "send_text"]

    def last(self, kind: str) -> tuple:
        return next(call for call in reversed(self.calls) if call[0] == kind)


class FakeGalleries:
    def __init__(self, gallery: Gallery) -> None:
        self.gallery = gallery

    async def get(self) -> Gallery:
        return self.gallery


class FakeCatalog:
    def __init__(self, ckpts=(DREAM, SD15, SDXL)) -> None:
        self.ckpts = list(ckpts)

    async def available(self):
        return available_profiles(self.ckpts)


def button_data(buttons) -> list[str]:
    return [data for row in buttons for _, data in row]


def make_bot(tmp_path: Path, runner, allowed: int | None = USER, ckpts=(DREAM, SD15, SDXL), jobs=None) -> StudioBot:
    gallery = Gallery(tmp_path, tmp_path / "cache")
    chats: dict[int, FakeChat] = {}
    bot = StudioBot(
        jobs or JobManager(runner), FakeGalleries(gallery), allowed, FakeCatalog(ckpts),
        users=UserStore(tmp_path / "users.json"), chat_for=lambda uid: chats.setdefault(uid, FakeChat()),
    )
    bot.test_chats = chats
    return bot


def done_runner(seen: list, make_png=None, tmp_path: Path | None = None):
    """Runner that records requests and returns one gallery image per requested batch slot."""

    async def runner(request, on_progress, on_preview=None):
        seen.append(request)
        for step in range(1, 21):
            on_progress(step, 20)
        count = 1 if isinstance(request, UpscaleRequest) else request.batch
        names = [f"img{len(seen)}_{i}.png" for i in range(count)]
        if make_png is not None:
            for name in names:
                make_png(tmp_path / name, size=(512, 512))
        images = [{"name": name, "width": 512, "height": 512, "params": None} for name in names]
        return {"image": images[0], "images": images, "seed": 1234, "elapsed": 5.5}

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
    caption = caption_for(GenerationParams(prompt="x" * 2000, width=512, height=768), 7, 3.2, "DreamShaper 8")
    assert len(caption) < 1024
    assert caption.endswith("seed 7 · 512×768 · DreamShaper 8 · 3.2s")


def test_unset_allowed_user_replies_with_id(tmp_path, caplog) -> None:
    seen: list = []
    chat = FakeChat()
    with caplog.at_level("INFO", logger="web.bot_core"):
        asyncio.run(make_bot(tmp_path, done_runner(seen), allowed=None).handle_text(777, "a cat", chat))
    assert seen == [] and "TELEGRAM_ALLOWED_USER_ID=777" in chat.texts()[0]
    assert "TELEGRAM_ALLOWED_USER_ID=777" in caplog.text


def test_help_commands(tmp_path) -> None:
    for text in ("/start", "/help"):
        chat = FakeChat()
        asyncio.run(make_bot(tmp_path, done_runner([])).handle_text(USER, text, chat))
        assert chat.texts()[0].startswith(HELP) and "DreamShaper 8" in chat.texts()[0]


def test_command_without_prompt_gets_hint(tmp_path) -> None:
    seen: list = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen)).handle_text(USER, "/portrait", chat))
    assert seen == [] and "/portrait an old fisherman" in chat.texts()[0]


def test_plain_text_uses_default_model_and_sends_photo_with_buttons(tmp_path, monkeypatch, make_png) -> None:
    monkeypatch.setattr(bot_core, "EDIT_INTERVAL", 60)  # only the first progress edit may happen
    seen: list = []
    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, done_runner(seen, make_png, tmp_path)).handle_text(USER, "a beach", chat))
    params = seen[0]
    assert (params.prompt, params.width, params.height, params.model, params.steps, params.cfg) == (
        "a beach", 512, 512, DREAM, 25, 7.0)
    assert chat.kinds() == ["send_text", "edit_text", "send_photo", "delete"]
    status_id = chat.calls[0][2]
    assert chat.calls[1] == ("edit_text", status_id, "🎨 DreamShaper 8 · step 1/20")
    _, path, caption, buttons = chat.calls[2]
    assert path == (tmp_path / "img1_0.png").resolve()
    assert caption == "a beach\n\nseed 1234 · 512×512 · DreamShaper 8 · 5.5s"
    assert [data.split(":")[0] for data in button_data(buttons)] == ["vary", "x4", "up"]
    assert chat.calls[3] == ("delete", status_id)


def test_model_and_style_choice_persist(tmp_path, make_png) -> None:
    seen: list = []
    chat = FakeChat()
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))

    async def scenario():
        await bot.handle_text(USER, "/model", chat)
        await bot.handle_button(USER, "model:sdxl", chat)
        await bot.handle_text(USER, "/style", chat)
        await bot.handle_button(USER, "style:anime", chat)
        await bot.handle_text(USER, "/portrait a cat", chat)

    asyncio.run(scenario())
    model_menu, style_menu = chat.calls[0][3], chat.calls[2][3]
    assert {"model:sdxl", "model:dreamshaper", "model:sd15"} <= set(button_data(model_menu))
    assert "style:anime" in button_data(style_menu)
    params = seen[0]
    assert (params.model, params.width, params.height) == (SDXL, 832, 1216)
    assert params.prompt.startswith("a cat, ") and "anime" in params.prompt
    photo_buttons = chat.last("send_photo")[3]
    assert [data.split(":")[0] for data in button_data(photo_buttons)] == ["vary"]  # SDXL: no ×4, no upscale


def test_choosing_missing_model_is_refused(tmp_path) -> None:
    chat = FakeChat()
    bot = make_bot(tmp_path, done_runner([]), ckpts=(DREAM,))
    asyncio.run(bot.handle_button(USER, "model:sdxl", chat))
    assert bot.prefs_for(USER).model_key is None and "isn't installed" in chat.texts()[0]


def test_vary_reuses_prompt_with_new_seed(tmp_path, make_png) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    chat = FakeChat()

    async def scenario():
        await bot.handle_text(USER, "/landscape hills", chat)
        vary = button_data(chat.last("send_photo")[3])[0]
        await bot.handle_button(USER, vary, chat)

    asyncio.run(scenario())
    assert [(p.prompt, p.width, p.height, p.seed, p.batch) for p in seen] == [("hills", 768, 512, None, 1)] * 2


def test_x4_sends_album_and_picker(tmp_path, make_png) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    chat = FakeChat()

    async def scenario():
        await bot.handle_text(USER, "a fox", chat)
        x4 = button_data(chat.last("send_photo")[3])[1]
        await bot.handle_button(USER, x4, chat)

    asyncio.run(scenario())
    assert (seen[1].prompt, seen[1].batch, seen[1].seed) == ("a fox", 4, None)
    _, paths, caption = chat.last("send_album")
    assert [p.name for p in paths] == [f"img2_{i}.png" for i in range(4)] and "a fox" in caption
    picker = next(c for c in reversed(chat.calls) if c[0] == "send_text" and c[3])
    assert [data.split(":")[0] for data in button_data(picker[3])] == ["up", "up", "up", "up", "x4"]


def test_upscale_sends_document(tmp_path, make_png) -> None:
    source_params = GenerationParams(prompt="a fox", negative_prompt="blurry", width=512, height=512, seed=9)
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    chat = FakeChat()

    async def scenario():
        await bot.handle_text(USER, "a fox", chat)
        make_png(tmp_path / "img1_0.png", graph_for(source_params), size=(512, 512))  # real settings in the PNG
        up = button_data(chat.last("send_photo")[3])[2]
        await bot.handle_button(USER, up, chat)

    asyncio.run(scenario())
    request = seen[1]
    assert isinstance(request, UpscaleRequest) and request.params.prompt == "a fox" and request.params.seed == 9
    _, path, caption = chat.last("send_document")
    assert path.name == "img2_0.png" and caption.startswith("🔍 Upscaled ×2")


def test_sdxl_refuses_x4_and_upscale(tmp_path, make_png) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    chat = FakeChat()
    params = GenerationParams(prompt="a", width=1024, height=1024, model=SDXL, seed=1)
    graph = graph_for(params)
    make_png(tmp_path / "xl.png", graph, size=(512, 512))

    async def scenario():
        x4_token = bot.actions.put(params, None, USER)
        up_token = bot.actions.put(params, "xl.png", USER)
        await bot.handle_button(USER, f"x4:{x4_token}", chat)
        await bot.handle_button(USER, f"up:{up_token}", chat)

    asyncio.run(scenario())
    assert seen == []
    assert "1 image at a time" in chat.texts()[0] and "can't be upscaled" in chat.texts()[1]


def test_expired_and_unknown_buttons(tmp_path) -> None:
    chat = FakeChat()
    bot = make_bot(tmp_path, done_runner([]))

    async def scenario():
        await bot.handle_button(USER, "vary:nope", chat)
        await bot.handle_button(USER, "up:nope", chat)
        await bot.handle_button(USER, "garbage", chat)

    asyncio.run(scenario())
    assert chat.texts() == [EXPIRED, EXPIRED]


def test_busy_reply(tmp_path) -> None:
    async def slow_runner(request, on_progress, on_preview=None):
        await asyncio.sleep(0.2)
        return {"image": None, "images": [], "seed": 1, "elapsed": 0.2}

    async def scenario():
        jobs = JobManager(slow_runner)
        jobs.start(GenerationParams(prompt="from the website"))
        bot = make_bot(tmp_path, slow_runner, jobs=jobs)
        chat = FakeChat()
        await bot.handle_text(USER, "a cat", chat)
        return chat

    assert asyncio.run(scenario()).texts() == [BUSY]


def test_error_event_edits_status(tmp_path) -> None:
    async def failing_runner(request, on_progress, on_preview=None):
        raise ComfyUIUnavailableError("Cannot connect to ComfyUI. Is ComfyUI running?")

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, failing_runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"]
    assert chat.calls[1][2] == "⚠️ Cannot connect to ComfyUI. Is ComfyUI running?"


def test_missing_image_reported(tmp_path) -> None:
    async def runner(request, on_progress, on_preview=None):
        return {"image": None, "images": [], "seed": 1, "elapsed": 1.0}

    chat = FakeChat()
    asyncio.run(make_bot(tmp_path, runner).handle_text(USER, "a cat", chat))
    assert chat.kinds() == ["send_text", "edit_text"] and "couldn't be found" in chat.calls[1][2]


def test_deleted_model_falls_back_to_default(tmp_path, make_png) -> None:
    seen: list = []
    chat = FakeChat()
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path), ckpts=(DREAM, SD15))  # SDXL file deleted
    bot.prefs_for(USER).model_key = "sdxl"

    async def scenario():
        await bot.handle_text(USER, "/model", chat)
        await bot.handle_text(USER, "a cat", chat)

    asyncio.run(scenario())
    menu = chat.calls[0][3]
    assert button_data(menu) == ["model:dreamshaper", "model:sd15"]  # menu still works
    assert bot.prefs_for(USER).model_key is None and "switched to" in chat.texts()[1]
    assert seen[0].model == DREAM  # the prompt still ran, on the default model


# --- friends: owner approves who may use the bot ---------------------------------------------

FRIEND = 999


def test_stranger_request_goes_to_owner_once(tmp_path) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen))
    stranger = FakeChat()

    async def scenario():
        await bot.handle_text(FRIEND, "a cat", stranger, who="Ann (@ann)")
        await bot.handle_text(FRIEND, "a dog", stranger, who="Ann (@ann)")

    asyncio.run(scenario())
    assert seen == [] and stranger.texts() == [REQUEST_SENT, PENDING]
    owner = bot.test_chats[USER]
    assert len(owner.calls) == 1  # pinged once, not on every message
    _, text, _, buttons = owner.calls[0]
    assert "Ann (@ann)" in text and button_data(buttons) == [f"allow:{FRIEND}", f"deny:{FRIEND}"]


def test_owner_approves_friend_with_own_settings(tmp_path, make_png) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    friend, owner = FakeChat(), FakeChat()

    async def scenario():
        await bot.handle_text(FRIEND, "hi", friend, who="Ann")
        await bot.handle_button(USER, f"allow:{FRIEND}", owner)
        await bot.handle_button(FRIEND, "model:sdxl", friend)
        await bot.handle_text(FRIEND, "a castle", friend, who="Ann")

    asyncio.run(scenario())
    assert "Ann can use the bot" in owner.texts()[0]
    welcome = bot.test_chats[FRIEND].texts()
    assert welcome and welcome[0] == WELCOME and "saved on" in WELCOME
    assert seen[0].model == SDXL and seen[0].prompt == "a castle"
    assert bot.prefs_for(USER).model_key is None  # the owner's choice is untouched
    assert UserStore(tmp_path / "users.json").is_allowed(FRIEND)  # survives a restart


def test_denied_user_is_ignored_afterwards(tmp_path) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen))
    friend, owner = FakeChat(), FakeChat()

    async def scenario():
        await bot.handle_text(FRIEND, "hi", friend, who="Bob")
        await bot.handle_button(USER, f"deny:{FRIEND}", owner)
        await bot.handle_text(FRIEND, "please", friend, who="Bob")

    asyncio.run(scenario())
    assert seen == [] and friend.texts() == [REQUEST_SENT]  # nothing after the denial
    assert len(bot.test_chats[FRIEND].calls) == 1  # told once that the owner said no
    assert len(bot.test_chats[USER].calls) == 1  # no new request ping


def test_only_owner_can_approve(tmp_path) -> None:
    bot = make_bot(tmp_path, done_runner([]))
    friend = FakeChat()

    async def scenario():
        bot.users.allow(FRIEND, "Ann")
        await bot.handle_button(FRIEND, "allow:555", friend)
        await bot.handle_text(FRIEND, "/users", friend)

    asyncio.run(scenario())
    assert not bot.users.is_allowed(555)
    assert "/users" not in friend.texts()[0]  # friends just get the normal help


def test_users_command_lists_and_removes(tmp_path) -> None:
    bot = make_bot(tmp_path, done_runner([]))
    owner, friend = FakeChat(), FakeChat()

    async def scenario():
        bot.users.allow(FRIEND, "Ann")
        await bot.handle_text(USER, "/users", owner)
        remove = button_data(owner.calls[0][3])[0]
        await bot.handle_button(USER, remove, owner)
        await bot.handle_text(FRIEND, "a cat", friend, who="Ann")

    asyncio.run(scenario())
    assert "Ann" in owner.calls[0][1] and button_data(owner.calls[0][3]) == [f"remove:{FRIEND}"]
    assert not bot.users.is_allowed(FRIEND) and friend.texts() == [REQUEST_SENT]


def test_buttons_are_personal(tmp_path, make_png) -> None:
    seen: list = []
    bot = make_bot(tmp_path, done_runner(seen, make_png, tmp_path))
    owner, friend = FakeChat(), FakeChat()

    async def scenario():
        bot.users.allow(FRIEND, "Ann")
        await bot.handle_text(USER, "a cat", owner)
        owners_vary = button_data(owner.last("send_photo")[3])[0]
        await bot.handle_button(FRIEND, owners_vary, friend)

    asyncio.run(scenario())
    assert len(seen) == 1 and friend.texts() == [EXPIRED]
