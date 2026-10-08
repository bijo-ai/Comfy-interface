"""Turn user choices (model, style, size, ×4, upscale) into GPU-safe job requests. Shared by Studio and the bot."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError, fit_size, inpaint_masks
from models import INPAINT_KEY, ModelProfile, check_limits, default_profile, profile_by_key, profile_for_ckpt
from styles import apply_style
from web.gallery import GalleryImage
from web.jobs import Img2ImgRequest, InpaintRequest, UpscaleRequest

UPSCALE_MAX_SIDE = 768  # ×2 of this is the largest image the 6 GB GPU refines comfortably
IMG2IMG_MAX_SIDE = {"sd15": 768, "sdxl": 1024}  # start images are resized to fit these
DEFAULT_STRENGTH = 0.55
FALLBACK_UPSCALE_PROMPT = "high quality, detailed"
FALLBACK_LIMITS = profile_by_key("sd15")


def resolve_profile(key: str | None, available: list[ModelProfile]) -> ModelProfile | None:
    """The chosen model, or the default one. None only when ComfyUI reported no known models."""
    if not key:
        return default_profile(available)
    profile = profile_by_key(key)
    if profile is not None and not profile.selectable:
        profile = None  # behind-the-scenes models can't be chosen directly
    if not available:
        raise InvalidParamsError("ComfyUI isn't reachable right now, or none of Studio's models are installed.")
    if profile is None or profile not in available:
        name = profile.label if profile else key
        raise InvalidParamsError(f"{name} isn't installed in ComfyUI.")
    return profile


def build_generation(
    *,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    cfg: float,
    seed: int | None,
    batch: int,
    style: str,
    profile: ModelProfile | None,
) -> GenerationParams:
    if not prompt.strip():
        raise InvalidParamsError("Invalid parameters: prompt must not be empty.")
    prompt, negative_prompt = apply_style(prompt, negative_prompt, style)
    params = GenerationParams(
        prompt, negative_prompt, width, height, steps, cfg, seed,
        model=profile.ckpt if profile else None, batch=batch,
    )
    params.validate()
    check_limits(profile or FALLBACK_LIMITS, width, height, batch)
    return params


def build_img2img(
    *,
    source: Path,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg: float,
    seed: int | None,
    batch: int,
    style: str,
    strength: float,
    profile: ModelProfile | None,
) -> Img2ImgRequest:
    """Plan an image-to-image job: the start image is fitted to a GPU-safe size for the chosen model."""
    if not 0.05 <= strength <= 1.0:
        raise InvalidParamsError(f"strength must be between 0.05 and 1.0 (got {strength}).")
    try:
        with Image.open(source) as img:
            source_width, source_height = img.size
    except OSError as exc:
        raise InvalidParamsError("The start image can't be read.") from exc
    max_side = IMG2IMG_MAX_SIDE[profile.family if profile else "sd15"]
    width, height = fit_size(source_width, source_height, max_side)
    params = build_generation(
        prompt=prompt, negative_prompt=negative_prompt, width=width, height=height,
        steps=steps, cfg=cfg, seed=seed, batch=batch, style=style, profile=profile,
    )
    if params.model is None:
        params = GenerationParams(**{**params.__dict__, "model": FALLBACK_LIMITS.ckpt})
    return Img2ImgRequest(source=source, params=params, strength=strength)


def build_inpaint(
    *, source: Path, mask: Path, prompt: str, negative_prompt: str, style: str, seed: int | None,
    available: list[ModelProfile],
) -> InpaintRequest:
    """Plan an inpainting job: only the painted area changes, using the dedicated inpainting model."""
    profile = profile_by_key(INPAINT_KEY)
    if profile not in available:
        raise InvalidParamsError(f"Inpainting needs the {profile.label} model, which isn't installed in ComfyUI.")
    if not prompt.strip():
        raise InvalidParamsError("Invalid parameters: prompt must not be empty.")
    try:
        with Image.open(source) as img:
            width, height = fit_size(*img.size, IMG2IMG_MAX_SIDE["sd15"])
    except OSError as exc:
        raise InvalidParamsError("The image can't be read.") from exc
    inpaint_masks(mask, width, height)  # raises if nothing was painted
    prompt, negative_prompt = apply_style(prompt, negative_prompt, style)
    params = GenerationParams(
        prompt, negative_prompt, width, height, profile.steps, profile.cfg, seed, model=profile.ckpt,
    )
    params.validate()
    return InpaintRequest(source=source, mask=mask, params=params)


def build_upscale(image: GalleryImage, source: Path, available: list[ModelProfile]) -> UpscaleRequest:
    """Plan a ×2 upscale of a gallery image, refusing anything that would overload the GPU."""
    if not image.width or not image.height:
        raise InvalidParamsError("This image can't be read, so it can't be upscaled.")
    if max(image.width, image.height) > UPSCALE_MAX_SIDE:
        raise InvalidParamsError(f"This image is already large ({image.width}×{image.height}); upscaling it would overload the GPU.")
    params = image.params or {}
    source_profile = profile_for_ckpt(params.get("model", ""))
    if source_profile is not None and not source_profile.upscale:
        raise InvalidParamsError(f"{source_profile.label} images can't be upscaled; it would overload the GPU.")
    usable = source_profile in available and source_profile.selectable  # never refine with the inpainting model
    profile = source_profile if usable else _upscale_fallback(available)
    return UpscaleRequest(
        source=source,
        params=GenerationParams(
            prompt=params.get("prompt") or FALLBACK_UPSCALE_PROMPT,
            negative_prompt=params.get("negative_prompt") or DEFAULT_NEGATIVE,
            width=image.width,
            height=image.height,
            cfg=profile.cfg,
            seed=params.get("seed"),
            model=profile.ckpt,
        ),
    )


def _upscale_fallback(available: list[ModelProfile]) -> ModelProfile:
    candidates = [p for p in available if p.upscale and p.selectable]
    profile = default_profile(candidates)
    if profile is None:
        raise InvalidParamsError("No SD 1.5 model is installed in ComfyUI for upscaling.")
    return profile
