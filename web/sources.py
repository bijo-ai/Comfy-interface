"""Start images for image-to-image (uploads and Telegram photos), stored as PNGs under the cache folder."""

from __future__ import annotations

import re
import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from comfy_client import InvalidParamsError

STORE_MAX_SIDE = 2048  # enough for any model size; keeps stored photos small
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_ID = re.compile(r"[0-9a-f]{32}")


class SourceStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def save(self, data: bytes) -> str:
        """Validate and store an uploaded image; returns its id."""
        if len(data) > MAX_UPLOAD_BYTES:
            raise InvalidParamsError("That image is larger than 20 MB.")
        try:
            with Image.open(BytesIO(data)) as img:
                img.load()
                picture = img.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidParamsError("That file isn't an image Studio can read.") from exc
        picture.thumbnail((STORE_MAX_SIDE, STORE_MAX_SIDE), Image.Resampling.LANCZOS)
        source_id = uuid.uuid4().hex
        self.folder.mkdir(parents=True, exist_ok=True)
        picture.save(self.folder / f"{source_id}.png", format="PNG")
        return source_id

    def path(self, source_id: str) -> Path:
        if not _ID.fullmatch(source_id):
            raise ValueError(f"Invalid source id: {source_id!r}")
        path = self.folder / f"{source_id}.png"
        if not path.is_file():
            raise FileNotFoundError(source_id)
        return path
