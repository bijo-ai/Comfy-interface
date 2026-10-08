"""Start ComfyUI Studio:  python -m web [--port 7860] [--no-browser]"""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from web.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ComfyUI Studio web app")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    url = f"http://127.0.0.1:{args.port}"
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    print(f"ComfyUI Studio running at {url}  (press Ctrl+C to stop)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
