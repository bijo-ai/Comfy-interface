"""AI edits: Extend (outpainting), Remove object, the sharp ×4 upscaler, Background and Faces.

Pixels are prepared here with Pillow/numpy (canvases, masks, fills), then run through ComfyUI with the core nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

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



# --- Background: cut out, plain colour, blur, or a new painted background ---------------------------------

BACKGROUND_MODEL = "birefnet.safetensors"  # ComfyUI models/background_removal
BACKGROUND_MODES = ("transparent", "white", "black", "blur", "prompt")
BACKGROUND_COLORS = {"white": 0xFFFFFF, "black": 0x000000}
BACKGROUND_BLUR_SIDE = 384  # blurred at this size and scaled back: strong, even blur at any resolution
BACKGROUND_BLUR_RADIUS = 24
BACKGROUND_BLUR_SIGMA = 10.0  # ImageBlur's maximum
BACKGROUND_GROW_PX = 6  # the new background reaches a little under the subject's edge, so no old colour shows
BACKGROUND_NEGATIVE = "person, people, duplicate, border, frame, text, watermark, blurry, low quality"


def build_background_workflow(
    image_name: str, size: tuple[int, int], mode: str, params: GenerationParams | None = None,
) -> Workflow:
    """Find the subject with BiRefNet, then put it on a transparent, plain, blurred or newly painted background.

    For "prompt", `params` carries the inpainting model, the prompt and the GPU-safe work size.
    """
    graph: Workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadBackgroundRemovalModel", "inputs": {"bg_removal_name": BACKGROUND_MODEL}},
        "3": {"class_type": "RemoveBackground", "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]}},  # 1 = subject
    }
    if mode == "transparent":
        graph["4"] = {"class_type": "InvertMask", "inputs": {"mask": ["3", 0]}}  # JoinImageWithAlpha inverts its alpha
        graph["5"] = {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["1", 0], "alpha": ["4", 0]}}
    elif mode in BACKGROUND_COLORS:
        graph["4"] = {
            "class_type": "EmptyImage",
            "inputs": {"width": size[0], "height": size[1], "batch_size": 1, "color": BACKGROUND_COLORS[mode]},
        }
        graph["5"] = _composite(["4", 0], ["1", 0], ["3", 0])
    elif mode == "blur":
        graph["6"] = _scale(["1", 0], fit_within(size, BACKGROUND_BLUR_SIDE))
        graph["7"] = {
            "class_type": "ImageBlur",
            "inputs": {"image": ["6", 0], "blur_radius": BACKGROUND_BLUR_RADIUS, "sigma": BACKGROUND_BLUR_SIGMA},
        }
        graph["4"] = _scale(["7", 0], size)
        graph["5"] = _composite(["4", 0], ["1", 0], ["3", 0])
    elif mode == "prompt" and params is not None and params.model:
        work = (params.width, params.height)
        graph.update({
            "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": params.model}},
            "11": _scale(["1", 0], work),
            "12": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
            "13": _scale(["12", 0], work),
            "14": {"class_type": "ImageToMask", "inputs": {"image": ["13", 0], "channel": "red"}},
            "15": {"class_type": "InvertMask", "inputs": {"mask": ["14", 0]}},
            "16": {"class_type": "ThresholdMask", "inputs": {"mask": ["15", 0], "value": 0.5}},
            "17": {"class_type": "GrowMask", "inputs": {"mask": ["16", 0], "expand": BACKGROUND_GROW_PX, "tapered_corners": True}},
            "18": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["10", 1]}},
            "19": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["10", 1]}},
            "20": {
                "class_type": "InpaintModelConditioning",
                "inputs": {
                    "positive": ["18", 0], "negative": ["19", 0], "vae": ["10", 2], "pixels": ["11", 0],
                    "mask": ["17", 0], "noise_mask": True,
                },
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["10", 0], "seed": params.seed, "steps": params.steps, "cfg": params.cfg,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                    "positive": ["20", 0], "negative": ["20", 1], "latent_image": ["20", 2],
                },
            },
            "21": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["8", 0], "vae": ["10", 2], **TILED}},
            "4": _scale(["21", 0], size),
            "5": _composite(["4", 0], ["1", 0], ["3", 0]),  # the original subject goes on top at full resolution
        })
    else:
        raise InvalidParamsError(f"Unknown background choice: {mode}.")
    graph[SAVE_NODE] = {"class_type": "SaveImage", "inputs": {"images": ["5", 0], "filename_prefix": "ComfyUI_background"}}
    return graph


def _scale(image: list[Any], size: tuple[int, int]) -> dict[str, Any]:
    return {
        "class_type": "ImageScale",
        "inputs": {"image": image, "upscale_method": "lanczos", "width": size[0], "height": size[1], "crop": "disabled"},
    }


def _composite(destination: list[Any], source: list[Any], mask: list[Any], x: int = 0, y: int = 0) -> dict[str, Any]:
    return {
        "class_type": "ImageCompositeMasked",
        "inputs": {"destination": destination, "source": source, "x": x, "y": y, "resize_source": False, "mask": mask},
    }


def fit_within(size: tuple[int, int], max_side: int) -> tuple[int, int]:
    """Scale down (never up) so the long side is at most max_side."""
    scale = min(1.0, max_side / max(size))
    return max(1, round(size[0] * scale)), max(1, round(size[1] * scale))


async def change_background(
    source: Path,
    mode: str,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Cut the subject out and put it on a new background; the subject keeps its full resolution."""
    with Image.open(source) as img:
        size = img.size
    painted = mode == "prompt"
    if painted:
        params.validate()
        params = params.with_seed()
    async with httpx.AsyncClient(timeout=30) as http:
        image_name = await ComfyClient(settings.comfyui_url, http).upload_bytes(_rgb_png(source), "studio_background_src.png")
    workflow = build_background_workflow(image_name, size, mode, params if painted else None)
    result = await _run(workflow, SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)
    result.params = replace(params, width=size[0], height=size[1])
    return result


def _rgb_png(source: Path) -> bytes:
    with Image.open(source) as img:
        return _png(img.convert("RGB"))


# --- Faces: find them, repaint each one larger, paste it back -------------------------------------------

FACE_MODEL = "mediapipe_face_fp32.safetensors"  # ComfyUI models/detection
MAX_FACES = 6
FACE_CONTEXT = 0.6  # the crop reaches this share of the face's size beyond it on every side
FACE_WORK_SIDE = 512  # faces are repainted at SD 1.5's natural size
FACE_MIN_PX = 12  # smaller blobs are noise
FACE_GROW = 0.12  # the repainted area includes a little hair, ears and jaw line
FACE_DENOISE = 0.35  # sharper eyes, mouth and skin without changing who it is (or their age)
FACE_PROMPT = "detailed face, natural skin texture, sharp eyes, high quality photo"
FACE_NEGATIVE = "deformed, distorted, disfigured, blurry, extra eyes, bad anatomy, low quality, cartoon"


def build_face_detect_workflow(image_name: str) -> Workflow:
    """A white-on-black mask of every face (face oval), up to MAX_FACES, as a temporary (non-gallery) image."""
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "LoadMediaPipeFaceLandmarker", "inputs": {"model_name": FACE_MODEL}},
        "3": {
            "class_type": "MediaPipeFaceLandmarker",
            "inputs": {
                "face_detection_model": ["2", 0], "image": ["1", 0], "detector_variant": "both", "num_faces": MAX_FACES,
                "min_confidence": 0.4, "missing_frame_fallback": "empty",
            },
        },
        "4": {"class_type": "MediaPipeFaceMask", "inputs": {"face_landmarks": ["3", 0], "regions": "all"}},
        "5": {"class_type": "MaskToImage", "inputs": {"mask": ["4", 0]}},
        SAVE_NODE: {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    }


def face_boxes(mask: Image.Image) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (left, top, right, bottom) of the separate white blobs in a face mask, largest first."""
    work = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
    boxes = []
    while (start := _first_white(work)) is not None:
        ImageDraw.floodfill(work, start, 128)
        blob = work.point(lambda v: 255 if v == 128 else 0)
        box = blob.getbbox()
        work.paste(0, mask=blob)
        if box and min(box[2] - box[0], box[3] - box[1]) >= FACE_MIN_PX:
            boxes.append(box)
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return boxes[:MAX_FACES]


def _first_white(image: Image.Image) -> tuple[int, int] | None:
    ys, xs = np.nonzero(np.asarray(image) == 255)
    return (int(xs[0]), int(ys[0])) if len(xs) else None


@dataclass(frozen=True)
class FaceCrop:
    box: tuple[int, int, int, int]  # square region of the picture: x, y, side, side
    work: int  # side it's repainted at


def face_crops(boxes: list[tuple[int, int, int, int]], size: tuple[int, int]) -> list[FaceCrop]:
    """A square crop with some context around each face, kept inside the picture."""
    crops = []
    for left, top, right, bottom in boxes:
        face = max(right - left, bottom - top)
        side = min(round(face * (1 + 2 * FACE_CONTEXT)), *size)
        center_x, center_y = (left + right) / 2, (top + bottom) / 2
        x = int(min(max(0, round(center_x - side / 2)), size[0] - side))
        y = int(min(max(0, round(center_y - side / 2)), size[1] - side))
        work = FACE_WORK_SIDE if side <= FACE_WORK_SIDE else min(768, side) // 8 * 8
        crops.append(FaceCrop((x, y, side, side), work))
    return crops


def face_masks(mask: Image.Image, crop: FaceCrop) -> tuple[bytes, bytes]:
    """(work-size mask for the sampler, crop-size soft mask for pasting) for one face."""
    x, y, side, _ = crop.box
    region = mask.convert("L").crop((x, y, x + side, y + side)).point(lambda v: 255 if v > 127 else 0)
    grow = max(3, round(side * FACE_GROW / (1 + 2 * FACE_CONTEXT)))
    region = region.filter(ImageFilter.MaxFilter(grow * 2 + 1))
    soft = region.filter(ImageFilter.GaussianBlur(max(2, grow // 2)))
    work = region.resize((crop.work, crop.work), Image.Resampling.BILINEAR)
    return _png(work.convert("RGB")), _png(soft.convert("RGB"))


def build_face_workflow(
    image_name: str, crops: list[FaceCrop], masks: list[tuple[str, str]], params: GenerationParams,
) -> Workflow:
    """Repaint each face larger (img2img on a crop, only inside the face) and paste it back with a soft edge."""
    graph: Workflow = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": params.model}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["2", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["2", 1]}},
    }
    picture: list[Any] = ["1", 0]
    steps = ("crop", "up", "encode", "work_mask", "noise", "sample", "decode", "down", "paste_mask", "paste")
    for index, (crop, (work_mask, paste_mask)) in enumerate(zip(crops, masks, strict=True)):
        x, y, side, _ = crop.box
        ids = {step: str(100 + index * 20 + offset) for offset, step in enumerate(steps)}
        graph.update({
            ids["crop"]: {"class_type": "ImageCrop", "inputs": {"image": picture, "width": side, "height": side, "x": x, "y": y}},
            ids["up"]: _scale([ids["crop"], 0], (crop.work, crop.work)),
            ids["encode"]: {"class_type": "VAEEncode", "inputs": {"pixels": [ids["up"], 0], "vae": ["2", 2]}},
            ids["work_mask"]: {"class_type": "LoadImageMask", "inputs": {"image": work_mask, "channel": "red"}},
            ids["noise"]: {"class_type": "SetLatentNoiseMask", "inputs": {"samples": [ids["encode"], 0], "mask": [ids["work_mask"], 0]}},
            ids["sample"]: {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0], "seed": params.seed + index, "steps": params.steps, "cfg": params.cfg,
                    "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": FACE_DENOISE,
                    "positive": ["3", 0], "negative": ["4", 0], "latent_image": [ids["noise"], 0],
                },
            },
            ids["decode"]: {"class_type": "VAEDecode", "inputs": {"samples": [ids["sample"], 0], "vae": ["2", 2]}},
            ids["down"]: _scale([ids["decode"], 0], (side, side)),
            ids["paste_mask"]: {"class_type": "LoadImageMask", "inputs": {"image": paste_mask, "channel": "red"}},
            ids["paste"]: _composite(picture, [ids["down"], 0], [ids["paste_mask"], 0], x, y),
        })
        picture = [ids["paste"], 0]
    graph[SAVE_NODE] = {"class_type": "SaveImage", "inputs": {"images": picture, "filename_prefix": "ComfyUI_faces"}}
    return graph


async def restore_faces(
    source: Path,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Find the faces (MediaPipe), then repaint each one in detail; everything else keeps its pixels."""
    if not params.model:
        raise InvalidParamsError("Fixing faces needs a model.")
    params = params.with_seed()
    with Image.open(source) as img:
        size = img.size
    async with httpx.AsyncClient(timeout=30) as http:
        client = ComfyClient(settings.comfyui_url, http)
        image_name = await client.upload_bytes(_rgb_png(source), "studio_faces_src.png")
        found = await _run(build_face_detect_workflow(image_name), SAVE_NODE, params, settings, None, None, save_copy=False)
        with Image.open(BytesIO(found.image)) as detected:
            mask = detected.convert("L").resize(size, Image.Resampling.NEAREST)
        crops = face_crops(face_boxes(mask), size)
        if not crops:
            raise InvalidParamsError("I couldn't find a face in this picture.")
        masks = []
        for index, crop in enumerate(crops):
            work_mask, paste_mask = face_masks(mask, crop)
            masks.append((
                await client.upload_bytes(work_mask, f"studio_face{index}_work.png"),
                await client.upload_bytes(paste_mask, f"studio_face{index}_paste.png"),
            ))
    workflow = build_face_workflow(image_name, crops, masks, params)
    result = await _run(workflow, SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)
    result.params = replace(params, width=size[0], height=size[1])
    return result
