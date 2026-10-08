"""Masks for Telegram inpainting: find a pink pen scribble drawn in Telegram's photo editor, or use a fixed area."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

WORK_SIDE = 768  # detect on a copy this size: fast, and it's the inpainting size anyway
HUE_RANGE = (205, 245)  # Pillow HSV hue (0-255) for pink / magenta (~290°-345°)
MIN_SATURATION = 140
MIN_VALUE = 140
MIN_COVERAGE = 0.001  # smaller than this is a speck of pink in the photo, not a scribble

REGIONS = [("top", "⬆️ Top"), ("bottom", "⬇️ Bottom"), ("left", "⬅️ Left"), ("right", "➡️ Right"), ("middle", "⏺️ Middle")]


def find_scribble(path: Path) -> Image.Image | None:
    """White-on-black mask (photo size) of the area marked with a pink pen, or None if there's no scribble."""
    with Image.open(path) as img:
        size = img.size
        work = img.convert("RGB")
    work.thumbnail((WORK_SIDE, WORK_SIDE))
    hue, saturation, value = work.convert("HSV").split()
    pink = ImageChops.multiply(
        ImageChops.multiply(
            hue.point(lambda h: 255 if HUE_RANGE[0] <= h <= HUE_RANGE[1] else 0),
            saturation.point(lambda s: 255 if s >= MIN_SATURATION else 0),
        ),
        value.point(lambda v: 255 if v >= MIN_VALUE else 0),
    )
    pink = pink.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))  # drop single-pixel noise
    if pink.histogram()[255] < MIN_COVERAGE * pink.width * pink.height:
        return None
    radius = max(6, round(min(work.size) * 0.02))  # cover the pen's soft edge and gaps between strokes
    marked = _fill_holes(pink.filter(ImageFilter.MaxFilter(radius * 2 + 1)))
    return marked.resize(size, Image.Resampling.NEAREST)


def region_mask(size: tuple[int, int], region: str) -> Image.Image:
    """A simple rectangular mask for the fallback area buttons."""
    width, height = size
    boxes = {
        "top": (0, 0, width, height * 0.45),
        "bottom": (0, height * 0.55, width, height),
        "left": (0, 0, width * 0.5, height),
        "right": (width * 0.5, 0, width, height),
        "middle": (width * 0.25, height * 0.2, width * 0.75, height * 0.8),
    }
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(boxes[region], fill=255)
    return mask


def _fill_holes(mask: Image.Image) -> Image.Image:
    """Treat areas enclosed by a scribble (e.g. a circle drawn around something) as marked too."""
    width, height = mask.size
    padded = Image.new("L", (width + 2, height + 2), 0)
    padded.paste(mask, (1, 1))
    ImageDraw.floodfill(padded, (0, 0), 128)  # everything reachable from outside the strokes
    return padded.point(lambda v: 0 if v == 128 else 255).crop((1, 1, width + 1, height + 1))
