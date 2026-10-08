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
        def __init__(self, factory, retry_seconds=30, on_started=None) -> None:
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


class FakeStudio:
    def __init__(self) -> None:
        self.button_calls: list[tuple] = []

    async def handle_text(self, user_id, text, chat, who="") -> None: ...

    async def handle_button(self, user_id, data, chat, who="") -> None:
        self.button_calls.append((user_id, data, chat._chat_id))


def test_updates_are_handled_concurrently() -> None:
    from web.bot import build_application

    # A long generation must not hold up other updates (Busy replies, Vary button answers).
    assert build_application("123:abc", FakeStudio()).concurrent_updates > 1


def test_vary_on_inaccessible_message_uses_effective_chat() -> None:
    from types import SimpleNamespace

    from telegram.ext import CallbackQueryHandler

    from web.bot import build_application

    studio = FakeStudio()
    application = build_application("123:abc", studio)
    handler = next(h for h in application.handlers[0] if isinstance(h, CallbackQueryHandler))

    async def answer() -> None: ...

    old_message = SimpleNamespace()  # like telegram.InaccessibleMessage: no .chat_id attribute
    query = SimpleNamespace(data="vary:tok123", message=old_message, from_user=SimpleNamespace(id=42), answer=answer)
    update = SimpleNamespace(callback_query=query, effective_chat=SimpleNamespace(id=555, type="private"))
    asyncio.run(handler.callback(update, SimpleNamespace(bot=object())))
    assert studio.button_calls == [(42, "vary:tok123", 555)]


def test_stop_finishes_cleanup_even_if_a_step_fails() -> None:
    log: list[str] = []
    app = FakeApp(log)

    async def broken_stop() -> None:
        log.append("updater.stop failed")
        raise RuntimeError("network")

    app.updater.stop = broken_stop

    async def scenario():
        runner = BotRunner(lambda: app, retry_seconds=0)
        runner.start()
        await asyncio.wait_for(runner.wait_started(), 2)
        await runner.stop()

    asyncio.run(scenario())
    assert log[-3:] == ["updater.stop failed", "stop", "shutdown"]


class RecordingBot:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def send_photo(self, chat_id, photo, caption=None, reply_markup=None):
        self.calls.append(("photo", chat_id, caption, reply_markup))

    async def send_media_group(self, chat_id, media):
        self.calls.append(("album", chat_id, media))

    async def send_document(self, chat_id, document, caption=None):
        self.calls.append(("document", chat_id, caption))

    async def send_message(self, chat_id, text, reply_markup=None):
        self.calls.append(("message", chat_id, text, reply_markup))
        from types import SimpleNamespace

        return SimpleNamespace(message_id=7)


def test_telegram_chat_maps_buttons_album_and_document(tmp_path) -> None:
    from telegram import InputMediaPhoto

    from web.bot import TelegramChat

    paths = []
    for i in range(4):
        path = tmp_path / f"{i}.png"
        path.write_bytes(b"png")
        paths.append(path)
    bot = RecordingBot()
    chat = TelegramChat(bot, 555)

    async def scenario():
        await chat.send_photo(paths[0], "cap", [[("🔁 Vary", "vary:t1"), ("🔍 Upscale", "up:t2")]])
        await chat.send_album(paths, "album cap")
        await chat.send_document(paths[1], "doc cap")
        return await chat.send_text("pick", [[("🔍 1", "up:a")], [("🖼️ ×4 again", "x4:b")]])

    assert asyncio.run(scenario()) == 7
    photo, album, document, message = bot.calls
    keyboard = photo[3].inline_keyboard
    assert [[(b.text, b.callback_data) for b in row] for row in keyboard] == [[("🔁 Vary", "vary:t1"), ("🔍 Upscale", "up:t2")]]
    assert len(album[2]) == 4 and all(isinstance(m, InputMediaPhoto) for m in album[2])
    assert album[2][0].caption == "album cap" and album[2][1].caption is None
    assert document == ("document", 555, "doc cap")
    assert [[b.callback_data for b in row] for row in message[3].inline_keyboard] == [["up:a"], ["x4:b"]]


def test_runner_calls_on_started_and_survives_its_failure() -> None:
    log: list[str] = []

    async def on_started(app) -> None:
        log.append("on_started")
        raise RuntimeError("set_my_commands failed")

    async def scenario():
        runner = BotRunner(lambda: FakeApp(log), retry_seconds=0, on_started=on_started)
        runner.start()
        await asyncio.wait_for(runner.wait_started(), 2)
        await runner.stop()

    asyncio.run(scenario())
    assert log[:4] == ["initialize", "start", "poll drop=True", "on_started"] and log[-1] == "shutdown"


def test_group_chats_are_ignored() -> None:
    from types import SimpleNamespace

    from telegram.ext import MessageHandler

    from web.bot import build_application

    calls: list = []

    class Studio(FakeStudio):
        async def handle_text(self, user_id, text, chat, who="") -> None:
            calls.append((user_id, text, who))

    application = build_application("123:abc", Studio())
    handler = next(h for h in application.handlers[0] if isinstance(h, MessageHandler))
    user = SimpleNamespace(id=42, full_name="Ann Lee", username="ann")

    def update(chat_type: str):
        message = SimpleNamespace(text="a cat", chat_id=7)
        return SimpleNamespace(effective_message=message, effective_user=user, effective_chat=SimpleNamespace(id=7, type=chat_type))

    asyncio.run(handler.callback(update("group"), SimpleNamespace(bot=object())))
    asyncio.run(handler.callback(update("private"), SimpleNamespace(bot=object())))
    assert calls == [(42, "a cat", "Ann Lee (@ann)")]
