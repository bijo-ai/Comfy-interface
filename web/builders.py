"""Turn user choices (model, style, size, ×4, upscale) into GPU-safe job requests. Shared by Studio and the bot."""

from __future__ import annotations

from pathlib import Path

from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError
from models import ModelProfile, check_limits, default_profile, profile_by_key, profile_for_ckpt
from styles import apply_style
from web.gallery import GalleryImage
from web.jobs import UpscaleRequest

UPSCALE_MAX_SIDE = 768  # ×2 of this is the largest image the 6 GB GPU refines comfortably
FALLBACK_UPSCALE_PROMPT = "high quality, detailed"
FALLBACK_LIMITS = profile_by_key("sd15")


def resolve_profile(key: str | None, available: list[ModelProfile]) -> ModelProfile | None:
    """The chosen model, or the default one. None only when ComfyUI reported no known models."""
    if not key:
        return default_profile(available)
    profile = profile_by_key(key)
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
    profile = source_profile if source_profile in available else _upscale_fallback(available)
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
    candidates = [p for p in available if p.upscale]
    profile = default_profile(candidates)
    if profile is None:
        raise InvalidParamsError("No SD 1.5 model is installed in ComfyUI for upscaling.")
    return profile
