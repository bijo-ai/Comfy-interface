"""Gallery over ComfyUI's output folder: listing, embedded settings, thumbnails, deletion."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_client import WorkflowError, find_single_node, identify_nodes, linked_node

log = logging.getLogger(__name__)

THUMB_WIDTH = 320
KINDS = ("generated", "edited", "fixed", "upscaled")
_KIND_PREFIXES = (
    ("ComfyUI_img2img", "edited"), ("LUMOS_edit", "edited"), ("ComfyUI_outpaint", "edited"), ("ComfyUI_background", "edited"),
    ("ComfyUI_inpaint", "fixed"), ("ComfyUI_remove", "fixed"), ("ComfyUI_faces", "fixed"), ("ComfyUI_upscaled", "upscaled"),
)
EDIT_PREFIX = "LUMOS_edit_"


def image_kind(name: str) -> str:
    """How an image was made, from the filename prefix each Studio workflow saves with."""
    filename = name.rsplit("/", 1)[-1]
    return next((kind for prefix, kind in _KIND_PREFIXES if filename.startswith(prefix)), "generated")
NUMERIC_KEYS = ("width", "height", "steps", "cfg", "seed")


@dataclass(frozen=True)
class GalleryImage:
    name: str  # POSIX path relative to the output folder
    created: float  # file mtime, seconds since epoch
    width: int
    height: int
    params: dict[str, Any] | None

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "kind": image_kind(self.name)}


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


def params_from_graph(graph: Any, size: tuple[int, int] | None = None) -> dict[str, Any] | None:
    """Extract generation settings from an API-format graph, or None if it isn't our shape.

    Graphs without an EmptyLatentImage (upscale, image-to-image) take their size from the image itself.
    """
    if not isinstance(graph, dict):
        return None
    try:
        sampler_id, positive, negative, width, height = _graph_layout(graph, size)
        sampler = graph[sampler_id]["inputs"]
        params = {
            "prompt": graph[positive]["inputs"]["text"],
            "negative_prompt": graph[negative]["inputs"]["text"],
            "width": width,
            "height": height,
            "steps": sampler["steps"],
            "cfg": sampler["cfg"],
            "seed": sampler["seed"],
        }
    except (WorkflowError, KeyError, TypeError, AttributeError):
        return None
    loaders = [node for node in graph.values() if node.get("class_type") == "CheckpointLoaderSimple"]
    if len(loaders) == 1 and isinstance(ckpt := loaders[0].get("inputs", {}).get("ckpt_name"), str):
        params["model"] = ckpt
    texts_ok = isinstance(params["prompt"], str) and isinstance(params["negative_prompt"], str)
    numbers_ok = all(
        isinstance(params[key], int | float) and not isinstance(params[key], bool) for key in NUMERIC_KEYS
    )
    return params if texts_ok and numbers_ok else None


def _graph_layout(graph: dict[str, Any], size: tuple[int, int] | None) -> tuple[str, str, str, Any, Any]:
    try:
        roles = identify_nodes(graph)
    except WorkflowError:
        if size is None:
            raise
        sampler = find_single_node(graph, "KSampler")
        positive = _prompt_node(graph, sampler, "positive")
        negative = _prompt_node(graph, sampler, "negative")
        return sampler, positive, negative, size[0], size[1]
    latent = graph[roles.latent]["inputs"]
    return roles.sampler, roles.positive, roles.negative, latent["width"], latent["height"]


def _prompt_node(graph: dict[str, Any], sampler: str, name: str) -> str:
    """The CLIPTextEncode feeding the sampler, looking through an InpaintModelConditioning (Extend, Remove)."""
    link = graph[sampler]["inputs"][name]
    if isinstance(link, list) and graph.get(link[0], {}).get("class_type") == "InpaintModelConditioning":
        return linked_node(graph, link[0], name, "CLIPTextEncode")
    return linked_node(graph, sampler, name, "CLIPTextEncode")


def _save_mode(img: Image.Image) -> str:
    """Keep see-through pixels (cut-outs); everything else is stored as RGB."""
    return "RGBA" if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info else "RGB"


def read_png(path: Path) -> tuple[int, int, dict[str, Any] | None]:
    """Return (width, height, params) for a PNG. Raises OSError if it can't be opened."""
    with Image.open(path) as img:
        width, height = img.size
        raw = img.info.get("prompt")
    try:
        graph = json.loads(raw) if isinstance(raw, str) else None
    except json.JSONDecodeError:
        graph = None
    return width, height, params_from_graph(graph, size=(width, height))


class Gallery:
    def __init__(self, output_dir: Path, cache_dir: Path) -> None:
        self.output_dir = output_dir.resolve()
        self.thumb_dir = cache_dir / "thumbs"
        self._meta: dict[Path, tuple[float, GalleryImage]] = {}

    def path(self, name: str) -> Path:
        return resolve_safe(self.output_dir, name)

    def page(self, offset: int, limit: int, query: str = "", kind: str | None = None) -> tuple[int, list[GalleryImage]]:
        """Newest first; optionally only one kind, and/or images whose prompt contains `query`."""
        files = self._scan()
        if kind:
            files = [(path, mtime) for path, mtime in files if image_kind(path.name) == kind]
        if not query.strip():
            return len(files), [self._describe(path, mtime) for path, mtime in files[offset : offset + limit]]
        needle = query.strip().lower()
        matches = [
            image for image in (self._describe(path, mtime) for path, mtime in files)
            if image.params and needle in image.params["prompt"].lower()
        ]
        return len(matches), matches[offset : offset + limit]

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(("all", *KINDS), 0)
        for path, _ in self._scan():
            counts["all"] += 1
            counts[image_kind(path.name)] += 1
        return counts

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
                img.convert(_save_mode(img)).save(thumb, "WEBP", quality=80)
        return thumb

    def save_edit(self, source: Path, parent: str | None = None) -> GalleryImage:
        """Store an Edit-studio draft as the next LUMOS_edit_NNNNN_.png, carrying the parent's settings."""
        info = PngInfo()
        if parent is not None:
            with Image.open(self._existing(parent)) as original:
                prompt = original.info.get("prompt")
            if isinstance(prompt, str):
                info.add_text("prompt", prompt)
        numbers = [int(m.group(1)) for p in self.output_dir.glob(f"{EDIT_PREFIX}*.png") if (m := re.search(r"_(\d+)_\.png$", p.name))]
        name = f"{EDIT_PREFIX}{max(numbers, default=0) + 1:05d}_.png"
        with Image.open(source) as img:
            img.convert(_save_mode(img)).save(self.output_dir / name, format="PNG", pnginfo=info)
        return self.get(name)

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
        except Exception:  # corrupt files, oversized text chunks, decompression bombs: list without settings
            log.debug("Could not read %s", path, exc_info=True)
            width, height, params = 0, 0, None
        image = GalleryImage(path.relative_to(self.output_dir).as_posix(), mtime, width, height, params)
        self._meta[path] = (mtime, image)
        return image

    def _thumb_path(self, name: str) -> Path:
        return self.thumb_dir / f"{hashlib.sha1(name.encode()).hexdigest()}.webp"
