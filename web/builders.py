"""Turn user choices (model, style, size, ×4, upscale, extend, remove) into GPU-safe job requests. Shared by Studio and the bot."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ai_edits import (
    BACKGROUND_MODEL,
    BACKGROUND_MODES,
    BACKGROUND_NEGATIVE,
    EXTEND_NEGATIVE,
    EXTEND_PROMPT,
    FACE_MODEL,
    FACE_NEGATIVE,
    FACE_PROMPT,
    REMOVE_NEGATIVE,
    REMOVE_PROMPT,
    extend_layout,
    remove_images,
)
from comfy_client import DEFAULT_NEGATIVE, GenerationParams, InvalidParamsError, fit_size, inpaint_masks
from models import INPAINT_KEY, ModelProfile, check_limits, default_profile, profile_by_key, profile_for_ckpt
from styles import apply_style
from web.gallery import GalleryImage
from web.jobs import (
    BackgroundRequest,
    ExtendRequest,
    FacesRequest,
    Img2ImgRequest,
    InpaintRequest,
    RemoveRequest,
    SharpUpscaleRequest,
    UpscaleRequest,
)

UPSCALE_MAX_SIDE = 768  # ×2 of this is the largest image the 6 GB GPU refines comfortably
IMG2IMG_MAX_SIDE = {"sd15": 768, "sdxl": 1024}  # start images are resized to fit these
EXTEND_MAX_SIDE = 1024  # the whole new canvas; SD 1.5 inpainting stays well within 6 GB at this size
SHARP_MAX_SIDE = 1024  # ×4 → 4096 px, the largest picture worth handling here
SHARP_MODEL = "RealESRGAN_x4plus.safetensors"
DEFAULT_EXTEND_AMOUNT = 0.25
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
    profile = _inpainter(available, "Inpainting")
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


def build_extend(
    *, source: Path, sides: set[str], amount: float, prompt: str, available: list[ModelProfile],
) -> ExtendRequest:
    """Plan an Extend: the picture is kept and new surroundings are painted on the chosen sides."""
    profile = _inpainter(available, "Extend")
    layout = extend_layout(_image_size(source), sides, amount, EXTEND_MAX_SIDE)
    params = GenerationParams(
        prompt.strip() or EXTEND_PROMPT, EXTEND_NEGATIVE, *layout.canvas, profile.steps, profile.cfg, model=profile.ckpt,
    )
    params.validate()
    return ExtendRequest(source=source, layout=layout, params=params)


def build_remove(*, source: Path, mask: Path, available: list[ModelProfile]) -> RemoveRequest:
    """Plan a Remove object: only the painted area changes, filled in to match its surroundings."""
    profile = _inpainter(available, "Remove object")
    width, height = fit_size(*_image_size(source), IMG2IMG_MAX_SIDE["sd15"])
    remove_images(source, mask, (width, height))  # raises if nothing was painted
    params = GenerationParams(REMOVE_PROMPT, REMOVE_NEGATIVE, width, height, profile.steps, profile.cfg, model=profile.ckpt)
    params.validate()
    return RemoveRequest(source=source, mask=mask, params=params)


def build_sharp_upscale(source: Path, parent_params: dict | None, upscalers: list[str]) -> SharpUpscaleRequest:
    """Plan a sharp ×4 upscale with the dedicated upscaling model (no diffusion, so any model's images work)."""
    if SHARP_MODEL not in upscalers:
        raise InvalidParamsError("The sharp ×4 upscaler (RealESRGAN x4plus) isn't installed in ComfyUI.")
    width, height = _image_size(source)
    if max(width, height) > SHARP_MAX_SIDE:
        raise InvalidParamsError(f"This image is already large ({width}×{height}); ×4 would make it over 4000 px.")
    parent = parent_params or {}
    params = GenerationParams(
        parent.get("prompt") or FALLBACK_UPSCALE_PROMPT, parent.get("negative_prompt") or DEFAULT_NEGATIVE,
        width, height, seed=parent.get("seed"), model=parent.get("model"),
    )
    return SharpUpscaleRequest(source=source, model_name=SHARP_MODEL, params=params)


def build_background(
    *, source: Path, mode: str, prompt: str, available: list[ModelProfile], removers: list[str],
) -> BackgroundRequest:
    """Plan a background change: cut out, plain colour, blur, or a newly painted background (`prompt`)."""
    if mode not in BACKGROUND_MODES:
        raise InvalidParamsError(f"Choose a background: {', '.join(BACKGROUND_MODES)}.")
    if BACKGROUND_MODEL not in removers:
        raise InvalidParamsError("Background removal needs the BiRefNet model, which isn't installed in ComfyUI.")
    size = _image_size(source)
    if mode != "prompt":
        return BackgroundRequest(source=source, mode=mode, params=GenerationParams("background", width=size[0], height=size[1]))
    if not prompt.strip():
        raise InvalidParamsError("Describe the new background, e.g. a sunny beach.")
    profile = _inpainter(available, "A new background")
    width, height = fit_size(*size, IMG2IMG_MAX_SIDE["sd15"])
    params = GenerationParams(prompt.strip(), BACKGROUND_NEGATIVE, width, height, profile.steps, profile.cfg, model=profile.ckpt)
    params.validate()
    return BackgroundRequest(source=source, mode=mode, params=params)


def build_faces(
    *, source: Path, parent_params: dict | None, available: list[ModelProfile], detectors: list[str],
) -> FacesRequest:
    """Plan a face fix: every face is found and repainted in detail with an SD 1.5 model."""
    if FACE_MODEL not in detectors:
        raise InvalidParamsError("Fixing faces needs the MediaPipe face model, which isn't installed in ComfyUI.")
    _image_size(source)  # raises if the picture can't be read
    profile = _upscale_fallback(available)  # a normal SD 1.5 model (DreamShaper first)
    prompt = _face_prompt((parent_params or {}).get("prompt"))
    params = GenerationParams(prompt, FACE_NEGATIVE, 512, 512, profile.steps, profile.cfg, model=profile.ckpt)
    return FacesRequest(source=source, params=params)


def _face_prompt(subject: str | None) -> str:
    """The picture's own prompt keeps age, look and style; prompts written by the edit tools themselves don't help."""
    subject = (subject or "").replace(f", {FACE_PROMPT}", "").strip()
    if not subject or subject in (FACE_PROMPT, EXTEND_PROMPT, REMOVE_PROMPT):
        return FACE_PROMPT
    return f"{subject}, {FACE_PROMPT}"


def _inpainter(available: list[ModelProfile], tool: str) -> ModelProfile:
    profile = profile_by_key(INPAINT_KEY)
    if profile not in available:
        raise InvalidParamsError(f"{tool} needs the {profile.label} model, which isn't installed in ComfyUI.")
    return profile


def _image_size(source: Path) -> tuple[int, int]:
    try:
        with Image.open(source) as img:
            return img.size
    except OSError as exc:
        raise InvalidParamsError("The image can't be read.") from exc
