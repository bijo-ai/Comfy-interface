"""Gallery over ComfyUI's output folder: listing, embedded settings, thumbnails, deletion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from comfy_client import WorkflowError, identify_nodes

THUMB_WIDTH = 320
NUMERIC_KEYS = ("width", "height", "steps", "cfg", "seed")


@dataclass(frozen=True)
class GalleryImage:
    name: str  # POSIX path relative to the output folder
    created: float  # file mtime, seconds since epoch
    width: int
    height: int
    params: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def resolve_safe(output_dir: Path, name: str) -> Path:
    """Map a gallery name to a file inside `output_dir`, rejecting anything that could escape it."""
    pure = PurePosixPath(name)
    unsafe = (
        not name
        or "\\" in name
        or ":" in name
        or pure.is_absolute()
        or ".." in pure.parts
        or pure.suffix.lower() != ".png"
    )
    root = output_dir.resolve()
    path = (root / pure).resolve()
    if unsafe or not path.is_relative_to(root):
        raise ValueError(f"Invalid image name: {name!r}")
    return path


def params_from_graph(graph: Any) -> dict[str, Any] | None:
    """Extract generation settings from an API-format graph, or None if it isn't our shape."""
    if not isinstance(graph, dict):
        return None
    try:
        roles = identify_nodes(graph)
        sampler = graph[roles.sampler]["inputs"]
        latent = graph[roles.latent]["inputs"]
        params = {
            "prompt": graph[roles.positive]["inputs"]["text"],
            "negative_prompt": graph[roles.negative]["inputs"]["text"],
            "width": latent["width"],
            "height": latent["height"],
            "steps": sampler["steps"],
            "cfg": sampler["cfg"],
            "seed": sampler["seed"],
        }
    except (WorkflowError, KeyError, TypeError, AttributeError):
        return None
    texts_ok = isinstance(params["prompt"], str) and isinstance(params["negative_prompt"], str)
    numbers_ok = all(
        isinstance(params[key], int | float) and not isinstance(params[key], bool) for key in NUMERIC_KEYS
    )
    return params if texts_ok and numbers_ok else None


def read_png(path: Path) -> tuple[int, int, dict[str, Any] | None]:
    """Return (width, height, params) for a PNG. Raises OSError if it can't be opened."""
    with Image.open(path) as img:
        width, height = img.size
        raw = img.info.get("prompt")
    try:
        graph = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        graph = None
    return width, height, params_from_graph(graph)


class Gallery:
    def __init__(self, output_dir: Path, cache_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.thumb_dir = cache_dir / "thumbs"
        self._meta: dict[Path, tuple[float, GalleryImage]] = {}

    def path(self, name: str) -> Path:
        return resolve_safe(self.output_dir, name)

    def page(self, offset: int, limit: int) -> tuple[int, list[GalleryImage]]:
        files = self._scan()
        return len(files), [self._describe(path, mtime) for path, mtime in files[offset : offset + limit]]

    def get(self, name: str) -> GalleryImage:
        path = self._existing(name)
        return self._describe(path, path.stat().st_mtime)

    def thumbnail(self, name: str) -> Path:
        path = self._existing(name)
        thumb = self._thumb_path(name)
        if not thumb.exists() or thumb.stat().st_mtime < path.stat().st_mtime:
            self.thumb_dir.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as img:
                img.thumbnail((THUMB_WIDTH, THUMB_WIDTH * 4))
                img.convert("RGB").save(thumb, "WEBP", quality=80)
        return thumb

    def delete(self, name: str) -> None:
        path = self._existing(name)
        path.unlink()
        self._thumb_path(name).unlink(missing_ok=True)
        self._meta.pop(path, None)

    def _existing(self, name: str) -> Path:
        path = self.path(name)
        if not path.is_file():
            raise FileNotFoundError(name)
        return path

    def _scan(self) -> list[tuple[Path, float]]:
        if not self.output_dir.is_dir():
            return []
        entries: list[tuple[Path, float]] = []
        for path in self.output_dir.rglob("*"):
            if path.suffix.lower() != ".png":
                continue
            try:
                if path.is_file():
                    entries.append((path, path.stat().st_mtime))
            except OSError:  # vanished mid-scan
                continue
        entries.sort(key=lambda entry: entry[1], reverse=True)
        return entries

    def _describe(self, path: Path, mtime: float) -> GalleryImage:
        cached = self._meta.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            width, height, params = read_png(path)
        except OSError:
            width, height, params = 0, 0, None
        image = GalleryImage(path.relative_to(self.output_dir).as_posix(), mtime, width, height, params)
        self._meta[path] = (mtime, image)
        return image

    def _thumb_path(self, name: str) -> Path:
        return self.thumb_dir / f"{hashlib.sha1(name.encode()).hexdigest()}.webp"
