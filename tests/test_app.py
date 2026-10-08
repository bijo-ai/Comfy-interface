from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from comfy_client import ComfyUIUnavailableError, GenerationParams
from tests.conftest import graph_for
from web import app as app_module
from web.app import GalleryProvider, create_app

BASE_URL = "http://127.0.0.1"


async def quick_runner(params, on_progress, on_preview=None):
    on_progress(1, 2)
    on_progress(2, 2)
    return {"image": None, "seed": 7, "elapsed": 0.1}


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings, runner=quick_runner), base_url=BASE_URL) as test_client:
        yield test_client


def sse_events(client: TestClient, job_id: str) -> list[dict]:
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_index_served(client) -> None:
    response = client.get("/")
    assert response.status_code == 200 and "ComfyUI Studio" in response.text


def test_status_reports_offline_comfyui(client, settings) -> None:
    body = client.get("/api/status").json()
    assert body == {"comfyui": False, "output_dir": str(settings.comfyui_output_dir)}


def test_generate_streams_progress_then_done(client) -> None:
    job_id = client.post("/api/generate", json={"prompt": "a fox"}).json()["job_id"]
    assert sse_events(client, job_id) == [
        {"type": "progress", "step": 1, "total": 2, "preview": False},
        {"type": "progress", "step": 2, "total": 2, "preview": False},
        {"type": "done", "image": None, "seed": 7, "elapsed": 0.1},
    ]


def test_late_sse_subscriber_still_gets_done(client) -> None:
    job_id = client.post("/api/generate", json={"prompt": "a fox"}).json()["job_id"]
    time.sleep(0.2)
    assert sse_events(client, job_id)[-1]["type"] == "done"


def test_generate_rejects_bad_params(client) -> None:
    response = client.post("/api/generate", json={"prompt": "a fox", "width": 500})
    assert response.status_code == 422
    assert "multiple of 8" in response.json()["error"]
    response = client.post("/api/generate", json={"prompt": "a fox", "steps": "lots"})
    assert response.status_code == 422 and response.json()["error"].startswith("Invalid request")


def test_generate_busy_returns_409(settings) -> None:
    async def slow_runner(params, on_progress, on_preview=None):
        await asyncio.sleep(0.5)
        return {"image": None, "seed": 1, "elapsed": 0.5}

    with TestClient(create_app(settings, runner=slow_runner), base_url=BASE_URL) as client:
        assert client.post("/api/generate", json={"prompt": "a"}).status_code == 200
        response = client.post("/api/generate", json={"prompt": "b"})
        assert response.status_code == 409 and "already" in response.json()["error"]


def test_runner_error_is_streamed(settings) -> None:
    async def failing_runner(params, on_progress, on_preview=None):
        raise ComfyUIUnavailableError("Cannot connect to ComfyUI at x. Is ComfyUI running?")

    with TestClient(create_app(settings, runner=failing_runner), base_url=BASE_URL) as client:
        job_id = client.post("/api/generate", json={"prompt": "a"}).json()["job_id"]
        assert sse_events(client, job_id) == [
            {"type": "error", "message": "Cannot connect to ComfyUI at x. Is ComfyUI running?"}
        ]


def test_unknown_job_404(client) -> None:
    assert client.get("/api/jobs/nope/events").status_code == 404


def test_images_with_spaces_and_subfolders(client, settings, make_png) -> None:
    params = GenerationParams(prompt="a fox", width=64, height=96, seed=3)
    make_png(settings.comfyui_output_dir / "my art" / "a b.png", graph_for(params))
    body = client.get("/api/images").json()
    assert body["total"] == 1
    image = body["images"][0]
    assert image["name"] == "my art/a b.png" and image["params"]["seed"] == 3
    url = "/api/images/my%20art/a%20b.png"
    assert client.get(url).headers["content-type"] == "image/png"
    assert "attachment" in client.get(url + "?download=1").headers["content-disposition"]
    assert client.get("/api/thumbs/my%20art/a%20b.png").headers["content-type"] == "image/webp"
    assert client.delete(url).json() == {"deleted": "my art/a b.png"}
    assert client.get("/api/images").json()["total"] == 0


def test_image_deleted_outside_app(client, settings, make_png) -> None:
    path = make_png(settings.comfyui_output_dir / "gone.png")
    assert client.get("/api/images").json()["total"] == 1
    path.unlink()
    assert client.get("/api/images/gone.png").status_code == 404
    assert client.get("/api/thumbs/gone.png").status_code == 404
    assert client.delete("/api/images/gone.png").status_code == 404
    assert client.get("/api/images").json()["total"] == 0


def test_unsafe_names_rejected(client) -> None:
    assert client.get("/api/images/C:/evil.png").status_code == 400
    assert client.delete("/api/images/notes.txt").status_code == 400


def test_paging(client, settings, make_png) -> None:
    for i in range(5):
        make_png(settings.comfyui_output_dir / f"{i}.png")
    body = client.get("/api/images?offset=3&limit=10").json()
    assert body["total"] == 5 and len(body["images"]) == 2


def test_unknown_output_dir_returns_503(settings) -> None:
    unknown = replace(settings, comfyui_output_dir=None)
    with TestClient(create_app(unknown, runner=quick_runner), base_url=BASE_URL) as client:
        for response in (client.get("/api/images"), client.post("/api/generate", json={"prompt": "a"})):
            assert response.status_code == 503
            assert "COMFYUI_OUTPUT_DIR" in response.json()["error"]


def test_gallery_provider_recovers_when_comfyui_starts(settings, monkeypatch, tmp_path: Path) -> None:
    answers = iter([None, tmp_path])

    async def fake_get_output_dir(base_url: str):
        return next(answers)

    monkeypatch.setattr(app_module, "get_output_dir", fake_get_output_dir)
    provider = GalleryProvider(replace(settings, comfyui_output_dir=None), None)
    assert asyncio.run(provider.get()) is None
    gallery = asyncio.run(provider.get())
    assert gallery is not None and gallery.output_dir == tmp_path.resolve()


def test_foreign_host_header_rejected(settings) -> None:
    with TestClient(create_app(settings, runner=quick_runner), base_url="http://evil.example") as client:
        assert client.get("/api/images").status_code == 400


def test_job_preview_endpoint(settings) -> None:
    async def previewing_runner(params, on_progress, on_preview):
        on_preview(b"\xff\xd8\xff\xe0preview")
        on_progress(1, 1)
        return {"image": None, "seed": 1, "elapsed": 0.1}

    with TestClient(create_app(settings, runner=previewing_runner), base_url=BASE_URL) as client:
        job_id = client.post("/api/generate", json={"prompt": "a"}).json()["job_id"]
        sse_events(client, job_id)
        response = client.get(f"/api/jobs/{job_id}/preview")
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"
        assert response.headers["cache-control"] == "no-store"
        assert response.content == b"\xff\xd8\xff\xe0preview"
        assert client.get("/api/jobs/nope/preview").status_code == 404


def test_job_preview_404_before_first_frame(client) -> None:
    job_id = client.post("/api/generate", json={"prompt": "a"}).json()["job_id"]
    sse_events(client, job_id)
    assert client.get(f"/api/jobs/{job_id}/preview").status_code == 404
