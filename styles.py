"""Style presets: keywords appended to the prompt and negative prompt so short prompts look good."""

from __future__ import annotations

from dataclasses import dataclass

from comfy_client import InvalidParamsError


@dataclass(frozen=True)
class Style:
    key: str
    label: str
    emoji: str
    positive: str
    negative: str


STYLES = (
    Style("none", "None", "✨", "", ""),
    Style("photo", "Photo", "📷",
          "professional photograph, natural lighting, sharp focus, 35mm, highly detailed",
          "cartoon, illustration, painting, drawing, cgi"),
    Style("cinematic", "Cinematic", "🎬",
          "cinematic film still, dramatic lighting, shallow depth of field, color graded, epic composition",
          "cartoon, flat lighting, amateur"),
    Style("oil", "Oil painting", "🎨",
          "oil painting, visible brush strokes, rich colors, classical fine art, canvas texture",
          "photo, 3d render, smooth"),
    Style("anime", "Anime", "✏️",
          "anime style illustration, vibrant colors, clean line art, detailed background, studio quality",
          "photo, realistic, 3d render"),
    Style("render3d", "3D render", "🧱",
          "3d render, octane render, soft studio lighting, highly detailed, smooth materials",
          "photo, painting, sketch, flat"),
    Style("watercolor", "Watercolor", "💧",
          "watercolor painting, soft washes, paper texture, delicate details, pastel colors",
          "photo, 3d render, harsh lines"),
)


def style_by_key(key: str) -> Style:
    style = next((s for s in STYLES if s.key == key), None)
    if style is None:
        raise InvalidParamsError(f"Unknown style {key!r}. Choose one of: {', '.join(s.key for s in STYLES)}.")
    return style


def apply_style(prompt: str, negative: str, key: str) -> tuple[str, str]:
    style = style_by_key(key)
    join = lambda base, extra: f"{base}, {extra}" if extra else base  # noqa: E731
    return join(prompt, style.positive), join(negative, style.negative)
