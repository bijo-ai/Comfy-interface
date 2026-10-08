from __future__ import annotations

import asyncio

import pytest

from comfy_client import ComfyUIUnavailableError, InvalidParamsError
from web.builders import resolve_profile
from web.catalog import ModelCatalog


def test_failed_lookup_is_not_cached() -> None:
    answers = [ComfyUIUnavailableError("down"), ["DreamShaper_8_pruned.safetensors"]]
    calls = []

    async def fetch():
        calls.append(1)
        answer = answers.pop(0) if answers else ["DreamShaper_8_pruned.safetensors"]
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def scenario():
        catalog = ModelCatalog("http://comfy", fetch=fetch)
        first = await catalog.available()
        second = await catalog.available()  # ComfyUI is back: must ask again, not reuse the failure
        third = await catalog.available()  # success is cached
        return first, second, third

    first, second, third = asyncio.run(scenario())
    assert first == [] and [p.key for p in second] == ["dreamshaper"] and third == second
    assert len(calls) == 2


def test_empty_catalog_explains_unreachable() -> None:
    with pytest.raises(InvalidParamsError, match="ComfyUI isn't reachable"):
        resolve_profile("dreamshaper", [])
