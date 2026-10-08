from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from comfy_client import GenerationParams, InvalidParamsError
from models import available_profiles
from web.builders import build_img2img
from web.jobs import Img2ImgRequest
from web.sources import SourceStore

DREAM = "DreamShaper_8_pruned.safetensors"
SDXL = "sd_xl_base_1.0.safetensors"


def image_bytes(size=(640, 480), fmt="JPEG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 90, 160)).save(buffer, format=fmt)
    return buffer.getvalue()


def test_save_and_load_source(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    source_id = store.save(image_bytes())
    path = store.path(source_id)
    with Image.open(path) as img:
        assert img.format == "PNG" and img.size == (640, 480)


def test_huge_upload_is_downsized_for_storage(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    with Image.open(store.path(store.save(image_bytes((5000, 2500))))) as img:
        assert max(img.size) == 2048


def test_rejects_non_images_and_bad_ids(tmp_path: Path) -> None:
    store = SourceStore(tmp_path)
    with pytest.raises(InvalidParamsError, match="isn't an image"):
        store.save(b"definitely not a picture")
    with pytest.raises(ValueError):
        store.path("../../etc/passwd")
    with pytest.raises(FileNotFoundError):
        store.path("0" * 32)


def make_source(tmp_path: Path, size) -> Path:
    path = tmp_path / "src.png"
    Image.new("RGB", size).save(path)
    return path


def build(tmp_path, size=(4000, 3000), key="dreamshaper", batch=1, strength=0.55, style="none"):
    from models import profile_by_key

    return build_img2img(
        source=make_source(tmp_path, size), prompt="make it winter", negative_prompt="blurry",
        steps=25, cfg=7.0, seed=None, batch=batch, style=style, strength=strength,
        profile=profile_by_key(key),
    )


def test_build_img2img_fits_size_per_model(tmp_path) -> None:
    request = build(tmp_path)
    assert isinstance(request, Img2ImgRequest) and request.strength == 0.55
    assert (request.params.width, request.params.height, request.params.model) == (768, 576, DREAM)
    xl = build(tmp_path, key="sdxl")
    assert (xl.params.width, xl.params.height) == (1024, 768)


def test_build_img2img_limits(tmp_path) -> None:
    assert build(tmp_path, batch=4).params.batch == 4  # 768×576 is within the ×4 limit
    with pytest.raises(InvalidParamsError, match="1 image at a time"):
        build(tmp_path, key="sdxl", batch=4)
    with pytest.raises(InvalidParamsError, match="strength"):
        build(tmp_path, strength=1.5)
    styled = build(tmp_path, style="oil")
    assert styled.params.prompt.startswith("make it winter, ") and "oil painting" in styled.params.prompt
