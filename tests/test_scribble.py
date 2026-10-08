from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from web.scribble import REGIONS, find_scribble, region_mask

PINK = (255, 45, 170)  # Telegram-style bright pink pen


def photo(tmp_path: Path, draw=None, size=(800, 600)) -> Path:
    img = Image.new("RGB", size, (70, 120, 60))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 0, size[0], size[1] // 2), fill=(120, 170, 230))  # sky
    d.ellipse((300, 250, 500, 450), fill=(200, 160, 120))  # a "face"
    if draw:
        draw(d)
    path = tmp_path / "photo.jpg"
    img.save(path, quality=85)  # JPEG, like Telegram
    return path


def test_filled_scribble_becomes_mask(tmp_path) -> None:
    path = photo(tmp_path, lambda d: d.line([(320, 230), (480, 230), (470, 270), (330, 270)], fill=PINK, width=18))
    mask = find_scribble(path)
    assert mask is not None and mask.size == (800, 600)
    assert mask.getpixel((400, 250)) == 255  # on the strokes
    assert mask.getpixel((50, 50)) == 0 and mask.getpixel((400, 550)) == 0  # far away stays untouched


def test_outline_is_filled_in(tmp_path) -> None:
    path = photo(tmp_path, lambda d: d.ellipse((330, 120, 470, 260), outline=PINK, width=10))
    mask = find_scribble(path)
    assert mask is not None and mask.getpixel((400, 190)) == 255  # inside the circle counts
    assert mask.getpixel((200, 190)) == 0


def test_no_scribble_or_tiny_specks_are_ignored(tmp_path) -> None:
    assert find_scribble(photo(tmp_path)) is None
    assert find_scribble(photo(tmp_path, lambda d: d.rectangle((10, 10, 13, 13), fill=PINK))) is None


def test_region_masks() -> None:
    assert [key for key, _ in REGIONS] == ["top", "bottom", "left", "right", "middle"]
    top = region_mask((800, 600), "top")
    assert top.getpixel((400, 50)) == 255 and top.getpixel((400, 550)) == 0
    middle = region_mask((800, 600), "middle")
    assert middle.getpixel((400, 300)) == 255 and middle.getpixel((20, 20)) == 0
