"""python-telegram-bot wiring for StudioBot, plus a start/stop lifecycle that survives being offline."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, InputFile, InputMediaPhoto, Update
from telegram.error import InvalidToken, TelegramError
from telegram.ext import Application, CallbackQueryHandler, ContextTypes, MessageHandler, filters

from web.bot_core import Buttons, StudioBot

log = logging.getLogger(__name__)

RETRY_SECONDS = 30
COMMANDS = [
    BotCommand("portrait", "Tall image: /portrait <prompt>"),
    BotCommand("landscape", "Wide image: /landscape <prompt>"),
    BotCommand("model", "Choose the model"),
    BotCommand("style", "Choose a style"),
    BotCommand("fix", "Photo caption: /fix <what should be there> (draw with the pink pen first)"),
    BotCommand("remove", "Photo caption: /remove (draw over it with the pink pen first)"),
    BotCommand("extend", "Photo caption: /extend [what the new area shows]"),
    BotCommand("faces", "Photo caption: /faces (clearer, more detailed faces)"),
    BotCommand("nobg", "Photo caption: /nobg (cut the subject out)"),
    BotCommand("bg", "Photo caption: /bg white | black | blur | <a new scene>"),
    BotCommand("help", "How to use this bot"),
]


class TelegramChat:
    """`Chat` implementation that talks to one Telegram chat."""

    def __init__(self, bot: Any, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = chat_id

    async def send_text(self, text: str, buttons: Buttons | None = None) -> int:
        markup = _keyboard(buttons) if buttons else None
        return (await self._bot.send_message(self._chat_id, text, reply_markup=markup)).message_id

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

    async def send_photo(self, path: Path, caption: str, buttons: Buttons) -> None:
        await self._bot.send_photo(self._chat_id, path.read_bytes(), caption=caption, reply_markup=_keyboard(buttons))

    async def send_album(self, paths: list[Path], caption: str) -> None:
        media = [InputMediaPhoto(path.read_bytes(), caption=caption if i == 0 else None) for i, path in enumerate(paths)]
        await self._bot.send_media_group(self._chat_id, media)

    async def send_document(self, path: Path, caption: str) -> None:
        await self._bot.send_document(self._chat_id, InputFile(path.read_bytes(), filename=path.name), caption=caption)


def _private(chat: Any) -> bool:
    """Only one-to-one chats: in groups, everyone there would see (and could tap) the owner's images."""
    return chat is not None and getattr(chat, "type", None) == "private"


def _who(user: Any) -> str:
    name = getattr(user, "full_name", "") or f"user {user.id}"
    username = getattr(user, "username", None)
    return f"{name} (@{username})" if username else name


def _keyboard(buttons: Buttons) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in buttons])


def build_application(token: str, studio: StudioBot) -> Application:
    async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user, chat = update.effective_message, update.effective_user, update.effective_chat
        if message is None or user is None or not message.text or not _private(chat):
            return
        await studio.handle_text(user.id, message.text, TelegramChat(context.bot, message.chat_id), who=_who(user))

    async def on_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user, chat = update.effective_message, update.effective_user, update.effective_chat
        if message is None or user is None or not _private(chat):
            return
        attachment = message.photo[-1] if message.photo else message.document  # largest photo size, or an image file
        reply = TelegramChat(context.bot, message.chat_id)
        try:
            data = bytes(await (await attachment.get_file()).download_as_bytearray())
        except TelegramError as exc:  # e.g. files over Telegram's 20 MB bot download limit
            await reply.send_text(f"⚠️ Couldn't download that image ({exc}).")
            return
        await studio.handle_photo(user.id, data, message.caption or "", reply, who=_who(user))

    async def on_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query, chat = update.callback_query, update.effective_chat
        if query is None or not query.data or not _private(chat):
            return
        await query.answer()
        # effective_chat, not query.message.chat_id: old buttons arrive with an InaccessibleMessage
        await studio.handle_button(query.from_user.id, query.data, TelegramChat(context.bot, chat.id), who=_who(query.from_user))

    async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("Telegram handler failed", exc_info=context.error)

    # Concurrent updates: a long generation must not hold up Busy replies or Vary button answers.
    application = Application.builder().token(token).concurrent_updates(True).build()
    studio.chat_for = lambda user_id: TelegramChat(application.bot, user_id)  # private chat id == user id
    application.add_handler(MessageHandler(filters.TEXT, on_text))
    application.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, on_photo))
    application.add_handler(CallbackQueryHandler(on_button))
    application.add_error_handler(on_error)
    return application


async def register_commands(app: Application) -> None:
    await app.bot.set_my_commands(COMMANDS)  # shows them in Telegram's "/" menu


class BotRunner:
    """Starts polling in the background, retrying while offline. Never takes the website down."""

    def __init__(
        self,
        factory: Callable[[], Any],
        retry_seconds: float = RETRY_SECONDS,
        on_started: Callable[[Any], Awaitable[None]] | None = None,
    ) -> None:
        self._factory = factory
        self._on_started = on_started
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
            if self._on_started is not None:
                try:
                    await self._on_started(app)
                except Exception as exc:  # cosmetic extras must not take the bot down
                    log.warning("Telegram bot started, but setup step failed: %s", exc)
            return

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if self._app is not None:
            for step in (self._app.updater.stop, self._app.stop, self._app.shutdown):
                try:
                    await step()
                except Exception:  # keep shutting down even if one step fails
                    log.warning("Telegram bot shutdown step failed", exc_info=True)
            self._app = None


async def _quiet_shutdown(app: Any) -> None:
    with contextlib.suppress(Exception):
        await app.shutdown()
