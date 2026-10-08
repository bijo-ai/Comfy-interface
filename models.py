"""Checkpoints Studio and the bot offer, with per-model sizes and GPU-safety limits (6 GB laptop GPU)."""

from __future__ import annotations

from dataclasses import dataclass

from comfy_client import InvalidParamsError

MAX_BATCH_SIDE = 768  # 4-at-once only up to this size, to keep VRAM in check
DEFAULT_KEY = "dreamshaper"

SD15_SHAPES = {"square": (512, 512), "portrait": (512, 768), "landscape": (768, 512)}
SDXL_SHAPES = {"square": (1024, 1024), "portrait": (832, 1216), "landscape": (1216, 832)}


@dataclass(frozen=True)
class ModelProfile:
    key: str
    label: str
    ckpt: str
    family: str  # "sd15" or "sdxl"
    shapes: dict[str, tuple[int, int]]
    max_batch: int
    upscale: bool
    steps: int
    cfg: float
    heavy: bool = False
    selectable: bool = True  # False: only used behind the scenes (inpainting), never offered in pickers


PROFILES = (
    ModelProfile("dreamshaper", "DreamShaper 8", "DreamShaper_8_pruned.safetensors", "sd15", SD15_SHAPES, 4, True, 25, 7.0),
    ModelProfile("sd15", "SD 1.5 base", "v1-5-pruned-emaonly.safetensors", "sd15", SD15_SHAPES, 4, True, 20, 8.0),
    ModelProfile("sdxl", "SDXL (heavy, slow)", "sd_xl_base_1.0.safetensors", "sdxl", SDXL_SHAPES, 1, False, 25, 7.0, heavy=True),
    ModelProfile(
        "dreamshaper_inpaint", "DreamShaper 8 Inpainting", "DreamShaper_8_INPAINTING.inpainting.safetensors",
        "sd15", SD15_SHAPES, 1, True, 25, 7.0, selectable=False,  # its results can be upscaled (by DreamShaper)
    ),
)
INPAINT_KEY = "dreamshaper_inpaint"


def available_profiles(ckpt_names: list[str]) -> list[ModelProfile]:
    installed = set(ckpt_names)
    return [profile for profile in PROFILES if profile.ckpt in installed]


def default_profile(available: list[ModelProfile]) -> ModelProfile | None:
    selectable = [p for p in available if p.selectable]
    return next((p for p in selectable if p.key == DEFAULT_KEY), selectable[0] if selectable else None)


def profile_by_key(key: str) -> ModelProfile | None:
    return next((p for p in PROFILES if p.key == key), None)


def profile_for_ckpt(ckpt: str) -> ModelProfile | None:
    return next((p for p in PROFILES if p.ckpt == ckpt), None)


def check_limits(profile: ModelProfile, width: int, height: int, batch: int) -> None:
    """Refuse requests that would overload the GPU, with a reason a person understands."""
    if not 1 <= batch <= 4:
        raise InvalidParamsError(f"Studio makes 1 to 4 images at once (got {batch}).")
    if batch > profile.max_batch:
        raise InvalidParamsError(f"{profile.label} makes 1 image at a time to keep your GPU safe.")
    if batch > 1 and max(width, height) > MAX_BATCH_SIDE:
        raise InvalidParamsError(f"4 at once is limited to {MAX_BATCH_SIDE}×{MAX_BATCH_SIDE} to keep your GPU safe.")
