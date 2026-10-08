from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from comfy_client import GenerationParams
from tests.conftest import graph_for
from web.gallery import Gallery, params_from_graph, read_png, resolve_safe

PARAMS = GenerationParams(prompt="a fox", negative_prompt="blurry", width=512, height=768, steps=12, cfg=6.5, seed=99)


@pytest.mark.parametrize("name", ["../x.png", "a/../../x.png", "/etc/x.png", "C:/x.png", r"a\x.png", "x.jpg", "", "dir/"])
def test_resolve_safe_rejects_unsafe(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        resolve_safe(tmp_path, name)


def test_resolve_safe_allows_subfolders_and_spaces(tmp_path: Path) -> None:
    assert resolve_safe(tmp_path, "my art/a b.PNG") == (tmp_path / "my art" / "a b.PNG").resolve()


def test_read_params_from_embedded_graph(tmp_path: Path, make_png) -> None:
    path = make_png(tmp_path / "a.png", graph_for(PARAMS), size=(512, 768))
    width, height, params = read_png(path)
    assert (width, height) == (512, 768)
    assert params == {"prompt": "a fox", "negative_prompt": "blurry", "width": 512, "height": 768,
                      "steps": 12, "cfg": 6.5, "seed": 99}


def test_read_params_without_chunk(tmp_path: Path, make_png) -> None:
    assert read_png(make_png(tmp_path / "a.png"))[2] is None


def test_read_params_with_invalid_json(tmp_path: Path, make_png) -> None:
    assert read_png(make_png(tmp_path / "a.png", graph="{not json"))[2] is None


def test_read_params_when_prompt_is_a_link() -> None:
    graph = graph_for(PARAMS)
    positive = graph["3"]["inputs"]["positive"][0]
    graph[positive]["inputs"]["text"] = ["42", 0]  # text fed by another node
    assert params_from_graph(graph) is None


def test_read_params_for_unrelated_graph() -> None:
    assert params_from_graph({"1": {"class_type": "LoadImage", "inputs": {}}}) is None
    assert params_from_graph(["not", "a", "dict"]) is None


def test_page_lists_newest_first_recursively(tmp_path: Path, make_png) -> None:
    old = make_png(tmp_path / "old.png")
    new = make_png(tmp_path / "sub" / "new.png")
    (tmp_path / "notes.txt").write_text("x")
    os.utime(old, (1_000, 1_000))
    os.utime(new, (2_000, 2_000))
    total, images = Gallery(tmp_path, tmp_path / "cache").page(0, 10)
    assert total == 2
    assert [i.name for i in images] == ["sub/new.png", "old.png"]
    total, images = Gallery(tmp_path, tmp_path / "cache").page(1, 10)
    assert [i.name for i in images] == ["old.png"]


def test_corrupt_png_is_listed(tmp_path: Path) -> None:
    (tmp_path / "bad.png").write_bytes(b"not a png")
    _, images = Gallery(tmp_path, tmp_path / "cache").page(0, 10)
    assert [(i.name, i.width, i.params) for i in images] == [("bad.png", 0, None)]


def test_thumbnail_is_cached_and_refreshed(tmp_path: Path, make_png) -> None:
    out = tmp_path / "out"
    src = make_png(out / "a.png", size=(640, 960))
    gallery = Gallery(out, tmp_path / "cache")
    thumb = gallery.thumbnail("a.png")
    assert thumb.suffix == ".webp" and thumb.exists()
    with Image.open(thumb) as img:
        assert img.size == (320, 480)
    first_mtime = thumb.stat().st_mtime
    assert gallery.thumbnail("a.png").stat().st_mtime == first_mtime  # cached
    os.utime(src, (first_mtime + 10, first_mtime + 10))
    assert gallery.thumbnail("a.png").stat().st_mtime >= first_mtime  # regenerated, no error


def test_delete_removes_png_and_thumbnail(tmp_path: Path, make_png) -> None:
    out = tmp_path / "out"
    make_png(out / "a.png")
    gallery = Gallery(out, tmp_path / "cache")
    thumb = gallery.thumbnail("a.png")
    gallery.delete("a.png")
    assert not (out / "a.png").exists() and not thumb.exists()
    with pytest.raises(FileNotFoundError):
        gallery.delete("a.png")
    with pytest.raises(FileNotFoundError):
        gallery.get("a.png")
