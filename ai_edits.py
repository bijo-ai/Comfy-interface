"""Phase 2 AI edits: Extend (outpainting), Remove object, and the sharp ×4 upscaler.

Pixels are prepared here with Pillow/numpy (canvases, masks, fills), then run through ComfyUI with the core nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageChops, ImageFilter

from comfy_client import (
    TILED,
    ComfyClient,
    GenerationParams,
    GenerationResult,
    InvalidParamsError,
    PreviewCallback,
    ProgressCallback,
    Workflow,
    _run,
    paste_at_full_size,
)
from config import Settings

SIDES = ("left", "top", "right", "bottom")
EXTEND_OVERLAP_PX = 16  # the model repaints a thin strip of the original too, so the seam disappears
EXTEND_FEATHER_PX = 8
EXTEND_FILL_BLUR = 24  # the new area starts as a blurred stretch of the picture: from flat grey the model may leave it grey
EXTEND_DENOISE = 1.0
EXTEND_FULL_MAX_SIDE = 2048  # the finished picture: only the new area is scaled up from the GPU-sized canvas
REMOVE_GROW_PX = 24  # brush strokes rarely cover an object's soft edge and shadow; leftovers turn into new objects
REMOVE_FEATHER_PX = 6
REMOVE_DENOISE = 0.9  # strong enough to repaint big holes cleanly; the smooth fill still steers colours
REMOVE_PROMPT = "empty background, seamless natural texture, nothing there, high quality"
REMOVE_NEGATIVE = (
    "person, people, animal, object, face, figure, text, watermark, logo, frame, border, blurry, smudge, artifacts"
)
EXTEND_PROMPT = "the same scene continuing naturally, background scenery, high quality, detailed"  # when no prompt is given
EXTEND_NEGATIVE = (
    "person, people, animal, extra objects, duplicate, border, frame, seam, split image, collage, text, watermark, "
    "blurry, low quality"
)
SHARP_SCALE = 4
SAVE_NODE = "9"


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _binary(mask: Image.Image) -> Image.Image:
    return mask.convert("L").point(lambda v: 255 if v > 127 else 0)


# --- Extend ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendLayout:
    canvas: tuple[int, int]  # picture size (the GPU canvas: multiples of 8)
    box: tuple[int, int, int, int]  # where the original lands: x, y, width, height


def extend_layout(size: tuple[int, int], sides: set[str], amount: float, max_side: int, min_side: int = 512) -> ExtendLayout:
    """Grow each chosen side by `amount` × the picture's width (left/right) or height (top/bottom), GPU-safe size."""
    unknown = sides - set(SIDES)
    if not sides or unknown:
        raise InvalidParamsError("Choose which sides to extend: left, top, right or bottom.")
    if not 0.05 <= amount <= 0.5:
        raise InvalidParamsError(f"amount must be between 0.05 and 0.5 (got {amount}).")
    width, height = size
    grow = {side: amount * (width if side in ("left", "right") else height) if side in sides else 0.0 for side in SIDES}
    new_width = width + grow["left"] + grow["right"]
    new_height = height + grow["top"] + grow["bottom"]
    scale = min(max(max(new_width, new_height), min_side), max_side) / max(new_width, new_height)
    canvas = (round(new_width * scale) // 8 * 8, round(new_height * scale) // 8 * 8)
    # an axis that isn't extended spans the whole canvas, so the picture never gets a sliver of new area there
    box_width = canvas[0] if not grow["left"] and not grow["right"] else min(round(width * scale), canvas[0])
    box_height = canvas[1] if not grow["top"] and not grow["bottom"] else min(round(height * scale), canvas[1])
    x = min(round(grow["left"] * scale), canvas[0] - box_width)
    y = min(round(grow["top"] * scale), canvas[1] - box_height)
    return ExtendLayout(canvas, (x, y, box_width, box_height))


def full_size_layout(work: ExtendLayout, size: tuple[int, int], max_side: int = EXTEND_FULL_MAX_SIDE) -> ExtendLayout | None:
    """The same layout with the original at its own resolution (capped at max_side), or None if that's no bigger."""
    x, y, box_width, box_height = work.box
    scale_x, scale_y = size[0] / box_width, size[1] / box_height
    width, height = round(work.canvas[0] * scale_x), round(work.canvas[1] * scale_y)
    fit = min(1.0, max_side / max(width, height))
    canvas = (round(width * fit), round(height * fit))
    if canvas[0] * canvas[1] <= work.canvas[0] * work.canvas[1]:
        return None
    full_width, full_height = min(round(size[0] * fit), canvas[0]), min(round(size[1] * fit), canvas[1])
    full_x = min(round(x * scale_x * fit), canvas[0] - full_width)
    full_y = min(round(y * scale_y * fit), canvas[1] - full_height)
    return ExtendLayout(canvas, (full_x, full_y, full_width, full_height))


def _extend_masks(layout: ExtendLayout, overlap: int, feather: float) -> tuple[Image.Image, Image.Image]:
    """(hard, soft) masks over the canvas: the new area plus an overlap strip; soft fades over that strip."""
    x, y, box_width, box_height = layout.box
    outside = Image.new("L", layout.canvas, 255)
    outside.paste(0, (x, y, x + box_width, y + box_height))
    keep = Image.new("L", layout.canvas, 255)
    keep.paste(0, _inset((x, y, x + box_width, y + box_height), overlap, layout.canvas))
    hard = ImageChops.lighter(outside, keep)
    soft = ImageChops.lighter(hard.filter(ImageFilter.GaussianBlur(feather)), outside)
    return hard, soft


def extend_images(source: Path, layout: ExtendLayout) -> tuple[bytes, bytes, bytes]:
    """(canvas, hard mask, soft mask) PNGs: the picture placed on its bigger canvas, and what may be painted."""
    x, y, box_width, box_height = layout.box
    with Image.open(source) as img:
        picture = img.convert("RGB").resize((box_width, box_height), Image.Resampling.LANCZOS)
    canvas = picture.resize(layout.canvas, Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(EXTEND_FILL_BLUR))
    canvas.paste(picture, (x, y))
    hard, soft = _extend_masks(layout, EXTEND_OVERLAP_PX, EXTEND_FEATHER_PX)
    return _png(canvas), _png(hard.convert("RGB")), _png(soft.convert("RGB"))


def full_size_images(source: Path, work: ExtendLayout, full: ExtendLayout) -> tuple[bytes, bytes]:
    """(original, opacity) PNGs for laying the original over the scaled-up result, fading out over the overlap."""
    x, y, box_width, box_height = full.box
    with Image.open(source) as img:
        picture = img.convert("RGB").resize((box_width, box_height), Image.Resampling.LANCZOS)
    ratio = full.canvas[0] / work.canvas[0]
    _, soft = _extend_masks(full, round(EXTEND_OVERLAP_PX * ratio), EXTEND_FEATHER_PX * ratio)
    opacity = ImageChops.invert(soft.crop((x, y, x + box_width, y + box_height)))
    return _png(picture), _png(opacity.convert("RGB"))


def add_full_size_overlay(workflow: Workflow, full: ExtendLayout, original: str, opacity: str) -> None:
    """Scale the extended picture up to `full` and lay the original over it at its own resolution."""
    x, y, _, _ = full.box
    workflow["12"] = {"class_type": "LoadImage", "inputs": {"image": original}}
    workflow["13"] = {"class_type": "LoadImageMask", "inputs": {"image": opacity, "channel": "red"}}
    workflow["14"] = {
        "class_type": "ImageScale",
        "inputs": {"image": ["11", 0], "upscale_method": "lanczos", "width": full.canvas[0], "height": full.canvas[1], "crop": "disabled"},
    }
    workflow["15"] = {
        "class_type": "ImageCompositeMasked",
        "inputs": {"destination": ["14", 0], "source": ["12", 0], "x": x, "y": y, "resize_source": False, "mask": ["13", 0]},
    }
    workflow[SAVE_NODE]["inputs"]["images"] = ["15", 0]


def _inset(rect: tuple[int, int, int, int], by: int, canvas: tuple[int, int]) -> tuple[int, int, int, int]:
    """Shrink the box by `by` on the sides that touch new area (edges of the canvas stay put)."""
    left, top, right, bottom = rect
    return (
        left + by if left > 0 else left,
        top + by if top > 0 else top,
        right - by if right < canvas[0] else right,
        bottom - by if bottom < canvas[1] else bottom,
    )


async def extend(
    source: Path,
    layout: ExtendLayout,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Outpaint: keep the picture and paint new surroundings on the chosen sides.

    The model paints on a GPU-sized canvas (`layout`); the original then goes back on top at its own resolution.
    """
    if not params.model:
        raise InvalidParamsError("Extending needs the inpainting model.")
    params.validate()
    params = params.with_seed()
    canvas, hard, soft = extend_images(source, layout)
    with Image.open(source) as img:
        full = full_size_layout(layout, img.size)
    async with httpx.AsyncClient(timeout=30) as http:
        client = ComfyClient(settings.comfyui_url, http)
        image_name = await client.upload_bytes(canvas, "studio_extend_src.png")
        hard_name = await client.upload_bytes(hard, "studio_extend_mask.png")
        soft_name = await client.upload_bytes(soft, "studio_extend_mask_soft.png")
        workflow = build_fill_workflow(params.model, image_name, hard_name, soft_name, params, EXTEND_DENOISE, "ComfyUI_outpaint")
        if full is not None:
            original, opacity = full_size_images(source, layout, full)
            add_full_size_overlay(
                workflow, full,
                await client.upload_bytes(original, "studio_extend_full.png"),
                await client.upload_bytes(opacity, "studio_extend_full_mask.png"),
            )
    result = await _run(workflow, SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)
    if full is not None:
        result.params = replace(params, width=full.canvas[0], height=full.canvas[1])
    return result


# --- Remove object ----------------------------------------------------------------------------------


def smooth_fill(image: Image.Image, hole: Image.Image) -> Image.Image:
    """Fill the white area of `hole` with colours flowing in from its surroundings (no AI, no new objects)."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    known = (np.asarray(hole.convert("L")) < 128).astype(np.float32)
    if not known.any():
        return image.convert("RGB")
    filled = pixels.copy()
    missing = known == 0
    height, width = known.shape
    radius = max(width, height) // 4
    # large blurs first (they always reach), then smaller ones overwrite with closer, more detailed colour
    while radius >= 2:
        weight = _blur(known, radius)
        estimate = np.stack([_blur(pixels[..., c] * known, radius) for c in range(3)], axis=-1)
        usable = missing & (weight > 0.02)
        filled[usable] = estimate[usable] / weight[usable][:, None]
        radius //= 2
    return Image.fromarray(np.clip(filled, 0, 255).astype(np.uint8))


def _blur(channel: np.ndarray, radius: int) -> np.ndarray:
    """Box blur (window sum ÷ window size) via a summed-area table; edges use the part of the window inside."""
    height, width = channel.shape
    table = np.zeros((height + 1, width + 1), dtype=np.float64)
    table[1:, 1:] = channel.cumsum(0).cumsum(1)
    rows, cols = np.arange(height), np.arange(width)
    top, bottom = np.clip(rows - radius, 0, height), np.clip(rows + radius + 1, 0, height)
    left, right = np.clip(cols - radius, 0, width), np.clip(cols + radius + 1, 0, width)
    sums = table[bottom][:, right] - table[top][:, right] - table[bottom][:, left] + table[top][:, left]
    area = (bottom - top)[:, None] * (right - left)[None, :]
    return (sums / area).astype(np.float32)


def remove_images(source: Path, mask: Path, size: tuple[int, int]) -> tuple[bytes, bytes, bytes]:
    """(filled picture, hard mask, soft mask) PNGs at `size`; raises if nothing was painted."""
    with Image.open(mask) as img:
        painted = _binary(img.resize(size, Image.Resampling.NEAREST))
    if painted.getbbox() is None:
        raise InvalidParamsError("Paint over what you want to remove first.")
    with Image.open(source) as img:
        picture = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    hard = painted.filter(ImageFilter.MaxFilter(REMOVE_GROW_PX * 2 + 1))
    soft = hard.filter(ImageFilter.GaussianBlur(REMOVE_FEATHER_PX))
    return _png(smooth_fill(picture, hard)), _png(hard.convert("RGB")), _png(soft.convert("RGB"))


def build_fill_workflow(
    ckpt: str, image_name: str, hard_mask: str, soft_mask: str, params: GenerationParams, denoise: float, prefix: str,
) -> Workflow:
    """Repaint only the masked area, starting from its pre-filled pixels, then paste it back onto the original."""
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": hard_mask, "channel": "red"}},
        "4": {"class_type": "LoadImageMask", "inputs": {"image": soft_mask, "channel": "red"}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["1", 1]}},
        "5": {
            "class_type": "InpaintModelConditioning",
            "inputs": {
                "positive": ["6", 0], "negative": ["7", 0], "vae": ["1", 2], "pixels": ["2", 0], "mask": ["3", 0],
                "noise_mask": True,
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": params.seed, "steps": params.steps, "cfg": params.cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": denoise,
                "positive": ["5", 0], "negative": ["5", 1], "latent_image": ["5", 2],
            },
        },
        "10": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["8", 0], "vae": ["1", 2], **TILED}},
        "11": {
            "class_type": "ImageCompositeMasked",
            "inputs": {"destination": ["2", 0], "source": ["10", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["4", 0]},
        },
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": prefix}},
    }


async def remove_object(
    source: Path,
    mask: Path,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Erase the painted object: fill the hole from its surroundings, then let the model add matching texture."""
    if not params.model:
        raise InvalidParamsError("Removing objects needs the inpainting model.")
    params.validate()
    params = params.with_seed()
    work_size = (params.width, params.height)
    filled, hard, soft = remove_images(source, mask, work_size)
    async with httpx.AsyncClient(timeout=30) as http:
        client = ComfyClient(settings.comfyui_url, http)
        image_name = await client.upload_bytes(filled, "studio_remove_src.png")
        hard_name = await client.upload_bytes(hard, "studio_remove_mask.png")
        soft_name = await client.upload_bytes(soft, "studio_remove_mask_soft.png")
        workflow = build_fill_workflow(params.model, image_name, hard_name, soft_name, params, REMOVE_DENOISE, "ComfyUI_remove")
        full_size = await paste_at_full_size(client, workflow, source, mask, work_size)
    result = await _run(workflow, SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)
    result.params = replace(params, width=full_size[0], height=full_size[1])
    return result


# --- Sharp ×4 upscale ---------------------------------------------------------------------------------


def build_sharp_upscale_workflow(model_name: str, image_name: str) -> Workflow:
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_name}},
        "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
        SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "ComfyUI_upscaled_x4"}},
    }


async def sharp_upscale(
    source: Path,
    model_name: str,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """×4 with a dedicated upscaling model: crisp, keeps the picture exactly, no diffusion (light on the GPU)."""
    async with httpx.AsyncClient(timeout=30) as http:
        image_name = await ComfyClient(settings.comfyui_url, http).upload_image(source, "studio_sharp_src.png")
    result = await _run(
        build_sharp_upscale_workflow(model_name, image_name), SAVE_NODE, params, settings, on_progress, on_preview,
        save_copy=False,
    )
    result.params = replace(params, width=params.width * SHARP_SCALE, height=params.height * SHARP_SCALE)
    return result

