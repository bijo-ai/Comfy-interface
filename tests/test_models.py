from __future__ import annotations

import re

import pytest

from comfy_client import InvalidParamsError
from models import PROFILES, available_profiles, check_limits, default_profile, profile_by_key, profile_for_ckpt

ALL = [p.ckpt for p in PROFILES]


def test_profiles_available_and_default() -> None:
    available = available_profiles(["v1-5-pruned-emaonly.safetensors", "sd_xl_base_1.0.safetensors", "other.ckpt"])
    assert [p.key for p in available] == ["sd15", "sdxl"]
    assert default_profile(available).key == "sd15"  # DreamShaper missing -> first available
    assert default_profile(available_profiles(ALL)).key == "dreamshaper"
    assert default_profile([]) is None


def test_profile_lookup() -> None:
    assert profile_by_key("sdxl").heavy is True
    assert profile_by_key("nope") is None
    assert profile_for_ckpt("DreamShaper_8_pruned.safetensors").key == "dreamshaper"
    assert profile_for_ckpt("unknown.safetensors") is None


def test_sdxl_shapes_are_larger() -> None:
    assert profile_by_key("sdxl").shapes["square"] == (1024, 1024)
    assert profile_by_key("dreamshaper").shapes["portrait"] == (512, 768)


@pytest.mark.parametrize(
    ("key", "width", "height", "batch", "message"),
    [
        ("sdxl", 1024, 1024, 4, "SDXL (heavy, slow) makes 1 image at a time"),
        ("dreamshaper", 1024, 1024, 4, "4 at once is limited to 768×768"),
        ("dreamshaper", 512, 512, 5, "makes 1 to 4 images"),
    ],
)
def test_check_limits_rejects(key, width, height, batch, message) -> None:
    with pytest.raises(InvalidParamsError, match=re.escape(message)):
        check_limits(profile_by_key(key), width, height, batch)


def test_check_limits_allows_normal_use() -> None:
    check_limits(profile_by_key("dreamshaper"), 512, 768, 4)
    check_limits(profile_by_key("sdxl"), 1024, 1024, 1)


def test_inpaint_model_is_hidden_from_pickers() -> None:
    from models import INPAINT_KEY

    names = ALL + ["DreamShaper_8_INPAINTING.inpainting.safetensors"]
    available = available_profiles(names)
    assert INPAINT_KEY in [p.key for p in available]
    assert default_profile(available).key == "dreamshaper"
    assert not profile_by_key(INPAINT_KEY).selectable
    only_inpaint = available_profiles(["DreamShaper_8_INPAINTING.inpainting.safetensors"])
    assert default_profile(only_inpaint) is None  # never used for normal generation
