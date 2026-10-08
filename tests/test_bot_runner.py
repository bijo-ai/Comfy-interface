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


class FakeStudio:
    def __init__(self) -> None:
        self.vary_calls: list[tuple] = []

    async def handle_text(self, user_id, text, chat) -> None: ...

    async def handle_vary(self, user_id, token, chat) -> None:
        self.vary_calls.append((user_id, token, chat._chat_id))


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
    update = SimpleNamespace(callback_query=query, effective_chat=SimpleNamespace(id=555))
    asyncio.run(handler.callback(update, SimpleNamespace(bot=object())))
    assert studio.vary_calls == [(42, "tok123", 555)]


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
