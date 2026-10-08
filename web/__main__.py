"""Start ComfyUI Studio:  python -m web [--port 7860] [--no-browser]"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from config import load_settings
from web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI Studio web app")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    import truststore

    truststore.inject_into_ssl()  # trust the Windows certificate store (AVG and co. re-sign HTTPS)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its INFO lines contain the bot token
    settings = load_settings()
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"ComfyUI Studio running at {url}  (press Ctrl+C to stop)")
    print(f"Telegram bot: {telegram_status(settings.telegram_bot_token, settings.telegram_allowed_user_id)}")
    uvicorn.run(create_app(settings), host="127.0.0.1", port=args.port, log_level="warning")


def telegram_status(token: str | None, allowed_user_id: int | None) -> str:
    if not token:
        return "off (no TELEGRAM_BOT_TOKEN in .env)"
    if allowed_user_id is None:
        return "on, waiting for your user ID (send it any message)"
    return f"on, answering only user {allowed_user_id}"


if __name__ == "__main__":
    main()
