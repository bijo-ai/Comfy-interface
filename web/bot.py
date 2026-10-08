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
