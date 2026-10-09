from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from comfy_client import ComfyUIUnavailableError
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from comfy_client import GenerationParams, apply_params, load_workflow
from config import PROJECT_DIR, Settings


def graph_for(params: GenerationParams) -> dict[str, Any]:
    template = load_workflow(PROJECT_DIR / "workflow_api.json")
    graph, _ = apply_params(template, params)
    return graph


@pytest.fixture
def make_png() -> Callable[..., Path]:
    def _make(path: Path, graph: Any = None, size: tuple[int, int] = (64, 96)) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        info = PngInfo()
        if graph is not None:
            info.add_text("prompt", graph if isinstance(graph, str) else json.dumps(graph))
        Image.new("RGB", size, (200, 120, 40)).save(path, pnginfo=info)
        return path

    return _make


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        comfyui_url="http://127.0.0.1:9",  # nothing listens here: connection refused fast
        workflow_path=PROJECT_DIR / "workflow_api.json",
        output_dir=tmp_path / "outputs",
        timeout=5,
        comfyui_output_dir=tmp_path / "comfy_out",
        cache_dir=tmp_path / "cache",
    )


@pytest.fixture(autouse=True)
def no_comfyui_checkpoint_lookup(monkeypatch):
    """Tests never reach a real ComfyUI: make the model catalog's lookup fail instantly."""
    from web.catalog import ModelCatalog

    async def offline(self):
        raise ComfyUIUnavailableError("offline in tests")

    monkeypatch.setattr(ModelCatalog, "_fetch_checkpoints", offline)
    async def offline_files(self, folder):
        raise ComfyUIUnavailableError("offline in tests")

    monkeypatch.setattr(ModelCatalog, "_fetch_model_files", offline_files)
