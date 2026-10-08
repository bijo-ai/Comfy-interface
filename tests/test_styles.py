from __future__ import annotations

import pytest

from comfy_client import InvalidParamsError
from styles import STYLES, apply_style


def test_apply_style_appends_keywords() -> None:
    prompt, negative = apply_style("a cat", "blurry", "anime")
    assert prompt.startswith("a cat, ") and "anime" in prompt
    assert negative.startswith("blurry, ") and "photo" in negative


def test_none_style_is_identity() -> None:
    assert apply_style("a cat", "blurry", "none") == ("a cat", "blurry")


def test_unknown_style_rejected() -> None:
    with pytest.raises(InvalidParamsError, match="Unknown style"):
        apply_style("a cat", "blurry", "glitter")


def test_every_style_has_label_and_emoji() -> None:
    assert [s.key for s in STYLES][0] == "none"
    assert all(s.label and s.emoji for s in STYLES)
