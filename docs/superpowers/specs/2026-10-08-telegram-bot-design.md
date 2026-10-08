# Telegram bot — design

Date: 2026-10-08 · Status: approved in chat

## Goal
Bijo messages **@lumoscomfybot** from his phone, anywhere. The laptop generates the image with ComfyUI and the
bot sends it back. Images land in ComfyUI's output folder, so they also appear in the Studio gallery.

## Decisions
| Decision | Choice |
|---|---|
| Process | The bot runs **inside the ComfyUI Studio web process**: started from the FastAPI lifespan when `TELEGRAM_BOT_TOKEN` is set. One window, one click (`start_studio.bat`). |
| GPU sharing | The bot uses the web app's `JobManager`, so web and bot never run two generations at once. If the web app is busy, the bot replies "Busy". |
| Library | `python-telegram-bot` 22.x, long polling (outbound only; no ports opened). |
| TLS | `truststore.inject_into_ssl()` at startup so AVG's HTTPS scanning doesn't break connections. |
| Commands | Plain text → square 512×512. `/portrait <prompt>` → 512×768. `/landscape <prompt>` → 768×512. `/start`, `/help` → help text. 🔁 **Vary** inline button → same prompt and size, new seed. |
| Progress | One status message "🎨 Generating… step N/T", edited at most every 1.5 s, then deleted once the photo is sent. Caption: prompt, seed, size, time. |
| Access | Only `TELEGRAM_ALLOWED_USER_ID` is served. If it's unset, the bot replies with the sender's ID and setup instructions, and generates nothing. Other users are ignored and logged. |
| Offline at startup | Retry connecting every 30 s in the background; the website runs regardless. An invalid token is logged once and not retried. |
| Pending updates | Dropped on start, so prompts sent while the laptop was off don't all fire at once. |
| Secrets | The token lives only in `.env` (git-ignored). The `httpx` logger is set to WARNING, because its INFO lines include the token in the URL. |

## Units
- `config.py`: `telegram_bot_token: str | None`, `telegram_allowed_user_id: int | None` (both default None).
- `web/bot_core.py`: library-free behaviour. `parse_request`, `caption_for`, `VaryStore`, and `StudioBot`
  (auth, generation via `JobManager`, progress throttling, delivery), talking to a `Chat` protocol.
- `web/bot.py`: python-telegram-bot adapter. `TelegramChat` implements `Chat`, `build_application`, and
  `BotRunner` (resilient start/stop).
- `web/app.py`: the lifespan starts and stops a `BotRunner` when a token is configured.
- `web/__main__.py`: truststore, the httpx log level, and a console line saying whether the bot is on or off.

## Errors
- **ComfyUI errors:** offline, timeout and workflow problems edit the status message to "⚠️ <message>".
- **Invalid input:** a command with no prompt gets a usage hint.
- **Expired button:** a Vary token missing after a restart gets "This button expired. Send the prompt again."
- **Telegram failures:** API errors on edit/delete are swallowed. Handler exceptions are logged by the error handler, and the bot keeps polling.

## Testing
- Unit tests with a fake `Chat` and a fake runner cover: parsing, auth (unset / other user / allowed), square and
  portrait sizes, missing prompt, busy, error event, throttled progress, photo plus caption plus Vary, Vary reuse,
  and expired Vary.
- `BotRunner`: retries after a network failure, stops on an invalid token, and `stop()` shuts the app down.
- `create_app` starts and stops the runner only when a token is set.
- Live: the bot connects (polling starts), Bijo sends `/start`, his ID is set, and a real prompt makes a real image that appears in the gallery.
