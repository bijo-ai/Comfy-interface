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
    assert response.status_code == 200 and "LUMOS Studios" in response.text


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


# --- quality pack: models, styles, batch, upscale ------------------------------------------

from comfy_client import GenerationParams as _Params  # noqa: E402
from web.jobs import UpscaleRequest  # noqa: E402

DREAM = "DreamShaper_8_pruned.safetensors"
SDXL = "sd_xl_base_1.0.safetensors"
SD15 = "v1-5-pruned-emaonly.safetensors"


class FakeCatalog:
    def __init__(self, ckpts: list[str]) -> None:
        self.ckpts = ckpts

    async def available(self):
        from models import available_profiles

        return available_profiles(self.ckpts)


def recording_app(settings, seen: list, ckpts=(DREAM, SD15, SDXL)):
    async def runner(request, on_progress, on_preview=None):
        seen.append(request)
        return {"image": None, "images": [], "seed": 1, "elapsed": 0.1}

    return create_app(settings, runner=runner, catalog=FakeCatalog(list(ckpts)))


def test_models_endpoint(settings) -> None:
    with TestClient(recording_app(settings, [], ckpts=(SD15, SDXL)), base_url=BASE_URL) as client:
        body = client.get("/api/models").json()
    by_key = {m["key"]: m for m in body["models"]}
    assert body["default"] == "sd15"  # DreamShaper not installed here
    assert by_key["dreamshaper"]["available"] is False and by_key["sdxl"]["heavy"] is True
    assert by_key["sdxl"]["shapes"]["portrait"] == [832, 1216] and by_key["sdxl"]["max_batch"] == 1
    assert [s["key"] for s in body["styles"]][:2] == ["none", "photo"]


def test_generate_with_model_style_batch(settings) -> None:
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        body = {"prompt": "a cat", "model": "dreamshaper", "style": "anime", "batch": 4, "width": 512, "height": 768}
        job_id = client.post("/api/generate", json=body).json()["job_id"]
        sse_events(client, job_id)
    params = seen[0]
    assert (params.model, params.batch, params.width, params.height) == (DREAM, 4, 512, 768)
    assert params.prompt.startswith("a cat, ") and "anime" in params.prompt


def test_generate_defaults_to_default_model(settings) -> None:
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        sse_events(client, client.post("/api/generate", json={"prompt": "a cat"}).json()["job_id"])
    assert seen[0].model == DREAM and seen[0].batch == 1


def test_generate_rejects_gpu_unsafe_or_missing(settings) -> None:
    with TestClient(recording_app(settings, [], ckpts=(DREAM, SDXL)), base_url=BASE_URL) as client:
        sdxl4 = client.post("/api/generate", json={"prompt": "a", "model": "sdxl", "batch": 4, "width": 1024, "height": 1024})
        big4 = client.post("/api/generate", json={"prompt": "a", "model": "dreamshaper", "batch": 4, "width": 1024, "height": 1024})
        missing = client.post("/api/generate", json={"prompt": "a", "model": "sd15"})
        badstyle = client.post("/api/generate", json={"prompt": "a", "style": "glitter"})
        empty = client.post("/api/generate", json={"prompt": "  ", "style": "anime"})
    assert sdxl4.status_code == 422 and "1 image at a time" in sdxl4.json()["error"]
    assert big4.status_code == 422 and "768" in big4.json()["error"]
    assert missing.status_code == 422 and "isn't installed" in missing.json()["error"]
    assert badstyle.status_code == 422 and "Unknown style" in badstyle.json()["error"]
    assert empty.status_code == 422 and "prompt" in empty.json()["error"]


def test_upscale_starts_job_with_source_settings(settings, make_png) -> None:
    graph = graph_for(_Params(prompt="a fox", negative_prompt="blurry", width=512, height=768, seed=9))
    make_png(settings.comfyui_output_dir / "fox.png", graph, size=(512, 768))
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        response = client.post("/api/upscale", json={"name": "fox.png"})
        assert response.status_code == 200
        sse_events(client, response.json()["job_id"])
    request = seen[0]
    assert isinstance(request, UpscaleRequest)
    assert request.source == (settings.comfyui_output_dir / "fox.png").resolve()
    p = request.params
    assert (p.prompt, p.negative_prompt, p.seed, p.width, p.height, p.model) == ("a fox", "blurry", 9, 512, 768, SD15)


def test_upscale_without_settings_uses_fallbacks(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "plain.png", size=(512, 512))
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        sse_events(client, client.post("/api/upscale", json={"name": "plain.png"}).json()["job_id"])
    p = seen[0].params
    assert p.prompt == "high quality, detailed" and p.model == DREAM and p.seed is None


def test_upscale_refuses_large_or_sdxl(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "big.png", size=(1024, 1024))
    graph = graph_for(_Params(prompt="a", width=512, height=512, seed=1))
    graph["4"]["inputs"]["ckpt_name"] = SDXL
    make_png(settings.comfyui_output_dir / "xl.png", graph, size=(512, 512))
    with TestClient(recording_app(settings, []), base_url=BASE_URL) as client:
        big = client.post("/api/upscale", json={"name": "big.png"})
        xl = client.post("/api/upscale", json={"name": "xl.png"})
        missing = client.post("/api/upscale", json={"name": "nope.png"})
    assert big.status_code == 422 and "already large" in big.json()["error"]
    assert xl.status_code == 422 and "SDXL" in xl.json()["error"]
    assert missing.status_code == 404


def test_batch_payload_lists_existing_images(settings, make_png, monkeypatch) -> None:
    from comfy_client import GenerationResult

    for i in (0, 1, 3):  # image 2 is missing on disk
        make_png(settings.comfyui_output_dir / f"b{i}.png")

    async def fake_generate(params, settings_, **kwargs):
        refs = [{"filename": f"b{i}.png", "subfolder": "", "type": "output"} for i in range(4)]
        return GenerationResult(b"png", None, replace(params, seed=3), "p", 1.0, refs[0], refs)

    monkeypatch.setattr(app_module, "generate", fake_generate)
    with TestClient(create_app(settings, catalog=FakeCatalog([DREAM])), base_url=BASE_URL) as client:
        job_id = client.post("/api/generate", json={"prompt": "a", "batch": 4}).json()["job_id"]
        done = sse_events(client, job_id)[-1]
    assert [i["name"] for i in done["images"]] == ["b0.png", "b1.png", "b3.png"]
    assert done["image"]["name"] == "b0.png"


# --- image-to-image --------------------------------------------------------------------------

def _jpeg(size=(800, 600)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 50, 50)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_upload_source_then_img2img(settings) -> None:
    from web.jobs import Img2ImgRequest

    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        upload = client.post("/api/sources", content=_jpeg(), headers={"Content-Type": "image/jpeg"})
        assert upload.status_code == 200
        source = upload.json()
        assert (source["width"], source["height"]) == (800, 600)
        assert client.get(f"/api/sources/{source['id']}").headers["content-type"] == "image/png"
        body = {"prompt": "make it winter", "source_id": source["id"], "strength": 0.4, "model": "dreamshaper"}
        sse_events(client, client.post("/api/generate", json=body).json()["job_id"])
    request = seen[0]
    assert isinstance(request, Img2ImgRequest) and request.strength == 0.4
    assert (request.params.width, request.params.height, request.params.model) == (768, 576, DREAM)


def test_img2img_from_gallery_image(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "cat.png", size=(512, 768))
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        body = {"prompt": "as a watercolor", "source_name": "cat.png"}
        sse_events(client, client.post("/api/generate", json=body).json()["job_id"])
    assert seen[0].source == (settings.comfyui_output_dir / "cat.png").resolve()
    assert (seen[0].params.width, seen[0].params.height, seen[0].strength) == (512, 768, 0.55)


def test_img2img_errors(settings) -> None:
    with TestClient(recording_app(settings, []), base_url=BASE_URL) as client:
        not_image = client.post("/api/sources", content=b"hello", headers={"Content-Type": "image/png"})
        missing = client.post("/api/generate", json={"prompt": "a", "source_id": "0" * 32})
        gone = client.post("/api/generate", json={"prompt": "a", "source_name": "nope.png"})
    assert not_image.status_code == 422 and "isn't an image" in not_image.json()["error"]
    assert missing.status_code == 404 and gone.status_code == 404


# --- inpainting --------------------------------------------------------------------------------

INPAINT = "DreamShaper_8_INPAINTING.inpainting.safetensors"


def _mask(size=(800, 600), painted=True) -> bytes:
    import io

    from PIL import Image, ImageDraw

    mask = Image.new("RGB", size, "black")
    if painted:
        ImageDraw.Draw(mask).rectangle((300, 0, 500, 150), fill="white")
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


def _upload(client, data: bytes) -> str:
    return client.post("/api/sources", content=data, headers={"Content-Type": "image/png"}).json()["id"]


def test_inpaint_starts_job(settings) -> None:
    from web.jobs import InpaintRequest

    seen: list = []
    with TestClient(recording_app(settings, seen, ckpts=(DREAM, INPAINT)), base_url=BASE_URL) as client:
        models = client.get("/api/models").json()
        assert models["inpaint"] is True and "dreamshaper_inpaint" not in [m["key"] for m in models["models"]]
        body = {"source_id": _upload(client, _jpeg()), "mask_id": _upload(client, _mask()), "prompt": "a red beanie", "style": "photo"}
        response = client.post("/api/inpaint", json=body)
        assert response.status_code == 200
        sse_events(client, response.json()["job_id"])
    request = seen[0]
    assert isinstance(request, InpaintRequest)
    assert (request.params.model, request.params.width, request.params.height) == (INPAINT, 768, 576)
    assert request.params.prompt.startswith("a red beanie, ") and "photograph" in request.params.prompt


def test_inpaint_errors(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "pic.png", size=(512, 512))
    with TestClient(recording_app(settings, [], ckpts=(DREAM, INPAINT)), base_url=BASE_URL) as client:
        empty = client.post("/api/inpaint", json={"source_name": "pic.png", "mask_id": _upload(client, _mask(painted=False)), "prompt": "a hat"})
        no_prompt = client.post("/api/inpaint", json={"source_name": "pic.png", "mask_id": _upload(client, _mask()), "prompt": " "})
        missing = client.post("/api/inpaint", json={"source_name": "nope.png", "mask_id": _upload(client, _mask()), "prompt": "a hat"})
    assert empty.status_code == 422 and "Paint over" in empty.json()["error"]
    assert no_prompt.status_code == 422 and "prompt" in no_prompt.json()["error"]
    assert missing.status_code == 404
    with TestClient(recording_app(settings, [], ckpts=(DREAM,)), base_url=BASE_URL) as client:
        assert client.get("/api/models").json()["inpaint"] is False
        body = {"source_name": "pic.png", "mask_id": _upload(client, _mask()), "prompt": "a hat"}
        not_installed = client.post("/api/inpaint", json=body)
    assert not_installed.status_code == 422 and "Inpainting" in not_installed.json()["error"]


def test_inpaint_model_cannot_be_picked_for_normal_generation(settings) -> None:
    with TestClient(recording_app(settings, [], ckpts=(DREAM, INPAINT)), base_url=BASE_URL) as client:
        response = client.post("/api/generate", json={"prompt": "a", "model": "dreamshaper_inpaint"})
    assert response.status_code == 422


def test_inpainted_image_upscales_with_a_normal_model(settings, make_png) -> None:
    graph = graph_for(_Params(prompt="a hat", width=512, height=512, seed=2))
    graph["4"]["inputs"]["ckpt_name"] = INPAINT
    make_png(settings.comfyui_output_dir / "fixed.png", graph, size=(512, 512))
    seen: list = []
    with TestClient(recording_app(settings, seen, ckpts=(DREAM, INPAINT)), base_url=BASE_URL) as client:
        sse_events(client, client.post("/api/upscale", json={"name": "fixed.png"}).json()["job_id"])
    assert seen[0].params.model == DREAM


def test_images_endpoint_filters(client, settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "ComfyUI_00001_.png", graph_for(GenerationParams(prompt="a red fox", seed=1)))
    make_png(settings.comfyui_output_dir / "ComfyUI_inpaint_00001_.png", graph_for(GenerationParams(prompt="a hat", seed=2)))
    body = client.get("/api/images?q=fox").json()
    assert body["total"] == 1 and body["images"][0]["kind"] == "generated"
    assert client.get("/api/images?kind=fixed").json()["total"] == 1
    assert client.get("/api/images?kind=bogus").status_code == 422
    counts = client.get("/api/images/counts").json()
    assert counts == {"all": 2, "generated": 1, "edited": 0, "fixed": 1, "upscaled": 0}


# --- Edit studio: saving drafts, upscaling drafts ----------------------------------------------

def _png_bytes(size=(512, 512), color=(10, 120, 200)) -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_save_draft_into_gallery_keeps_parent_settings(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "ComfyUI_00001_.png", graph_for(_Params(prompt="a red fox", seed=5)))
    with TestClient(recording_app(settings, []), base_url=BASE_URL) as client:
        draft = _upload(client, _png_bytes((400, 400)))
        first = client.post("/api/save", json={"source_id": draft, "parent_name": "ComfyUI_00001_.png"})
        assert first.status_code == 200
        second = client.post("/api/save", json={"source_id": _upload(client, _png_bytes())}).json()
        listed = client.get("/api/images?kind=edited").json()
        found = client.get("/api/images?q=red fox").json()["total"]
        bad = client.post("/api/save", json={"source_id": "not-an-id"})
        gone = client.post("/api/save", json={"source_id": draft, "parent_name": "nope.png"})
    saved = first.json()
    assert saved["name"] == "LUMOS_edit_00001_.png" and saved["kind"] == "edited"
    assert (saved["width"], saved["height"]) == (400, 400) and saved["params"]["prompt"] == "a red fox"
    assert second["name"] == "LUMOS_edit_00002_.png" and second["params"] is None
    assert listed["total"] == 2 and found == 2  # the saved edit is searchable by its original prompt
    assert bad.status_code == 400 and gone.status_code == 404


def test_upscale_from_draft(settings, make_png) -> None:
    make_png(settings.comfyui_output_dir / "ComfyUI_00001_.png", graph_for(_Params(prompt="a red fox", seed=5)))
    seen: list = []
    with TestClient(recording_app(settings, seen), base_url=BASE_URL) as client:
        small = _upload(client, _png_bytes((512, 384)))
        big = _upload(client, _png_bytes((1024, 1024)))
        ok = client.post("/api/upscale", json={"source_id": small, "parent_name": "ComfyUI_00001_.png"})
        sse_events(client, ok.json()["job_id"])
        refused = client.post("/api/upscale", json={"source_id": big})
        neither = client.post("/api/upscale", json={})
    request = seen[0]
    assert (request.params.width, request.params.height, request.params.prompt) == (512, 384, "a red fox")
    assert refused.status_code == 422 and "already large" in refused.json()["error"]
    assert neither.status_code == 422
