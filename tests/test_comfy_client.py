from __future__ import annotations

import asyncio
import json
from pathlib import Path

from comfy_client import _wait_ws, output_dir_from_argv


class FakeWs:
    def __init__(self, messages: list[str | bytes]) -> None:
        self._messages = messages

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for message in self._messages:
            yield message


def msg(kind: str, **data) -> str:
    return json.dumps({"type": kind, "data": data})


def test_wait_ws_reports_progress_for_our_prompt_only() -> None:
    ws = FakeWs([
        msg("progress", prompt_id="other", value=1, max=9),
        b"\x00\x01binary-preview",
        msg("progress", prompt_id="p1", value=1, max=2),
        msg("progress", prompt_id="p1", value=2, max=2),
        msg("executing", prompt_id="p1", node=None),
    ])
    seen: list[tuple[int, int]] = []
    asyncio.run(_wait_ws(ws, "p1", lambda step, total: seen.append((step, total))))
    assert seen == [(1, 2), (2, 2)]


def test_wait_ws_without_callback_still_completes() -> None:
    ws = FakeWs([msg("progress", prompt_id="p1", value=1, max=1), msg("execution_success", prompt_id="p1")])
    asyncio.run(_wait_ws(ws, "p1", None))


def test_output_dir_from_argv() -> None:
    assert output_dir_from_argv(["main.py", "--output-directory", r"C:\out"]) == Path(r"C:\out")
    assert output_dir_from_argv(["main.py", r"--output-directory=C:\out"]) == Path(r"C:\out")
    assert output_dir_from_argv(["main.py", "--listen"]) is None
    assert output_dir_from_argv(["main.py", "--output-directory"]) is None


import struct
from dataclasses import replace

import pytest

import httpx

from comfy_client import ComfyClient, decode_preview

JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def legacy_frame(image: bytes, type_num: int = 1) -> bytes:
    return struct.pack(">I", 1) + struct.pack(">I", type_num) + image


def metadata_frame(image: bytes, metadata: dict) -> bytes:
    meta = json.dumps(metadata).encode()
    return struct.pack(">I", 4) + struct.pack(">I", len(meta)) + meta + image


def test_decode_preview_legacy_and_metadata_frames() -> None:
    assert decode_preview(legacy_frame(JPEG), "p1") == JPEG
    assert decode_preview(legacy_frame(b"\x89PNG", type_num=2), "p1") == b"\x89PNG"
    assert decode_preview(metadata_frame(JPEG, {"prompt_id": "p1", "image_type": "image/jpeg"}), "p1") == JPEG
    assert decode_preview(metadata_frame(JPEG, {"prompt_id": "other"}), "p1") is None
    assert decode_preview(struct.pack(">I", 3) + b"text frame", "p1") is None
    assert decode_preview(b"\x00\x01", "p1") is None


def test_wait_ws_passes_previews() -> None:
    ws = FakeWs([legacy_frame(JPEG), msg("executing", prompt_id="p1", node=None)])
    previews: list[bytes] = []
    asyncio.run(_wait_ws(ws, "p1", None, previews.append))
    assert previews == [JPEG]


def test_queue_prompt_requests_previews_only_when_asked() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"prompt_id": "p1"})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = ComfyClient("http://comfy", http)
            await client.queue_prompt({"1": {"class_type": "X", "inputs": {}}})
            await client.queue_prompt({"1": {"class_type": "X", "inputs": {}}}, previews=True)

    asyncio.run(scenario())
    assert "extra_data" not in bodies[0]
    assert bodies[1]["extra_data"] == {"preview_method": "auto"}  # TAESD when installed, else latent2rgb


from comfy_client import GenerationParams, _output_images, apply_params, load_workflow
from config import PROJECT_DIR


def test_apply_params_sets_model_and_batch() -> None:
    template = load_workflow(PROJECT_DIR / "workflow_api.json")
    params = GenerationParams(prompt="a", model="DreamShaper_8_pruned.safetensors", batch=4, seed=1)
    graph, roles = apply_params(template, params)
    assert graph["4"]["inputs"]["ckpt_name"] == "DreamShaper_8_pruned.safetensors"
    assert graph[roles.latent]["inputs"]["batch_size"] == 4
    untouched, _ = apply_params(template, GenerationParams(prompt="a", seed=1))
    assert untouched["4"]["inputs"]["ckpt_name"] == template["4"]["inputs"]["ckpt_name"]


def test_batch_must_be_1_to_4() -> None:
    import pytest

    from comfy_client import InvalidParamsError

    with pytest.raises(InvalidParamsError, match="batch"):
        GenerationParams(prompt="a", batch=0).validate()


def test_output_images_lists_all_outputs() -> None:
    images = [{"filename": f"ComfyUI_0000{i}_.png", "subfolder": "", "type": "output"} for i in range(4)]
    entry = {"status": {"status_str": "success"}, "outputs": {"9": {"images": images}}}
    assert _output_images(entry, "9") == images


def test_list_checkpoints_handles_both_combo_formats() -> None:
    old = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}}}}
    new = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": ["COMBO", {"options": ["c.safetensors"]}]}}}}
    answers = iter([old, new])

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/object_info/CheckpointLoaderSimple"
        return httpx.Response(200, json=next(answers))

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = ComfyClient("http://comfy", http)
            return await client.list_checkpoints(), await client.list_checkpoints()

    assert asyncio.run(scenario()) == (["a.safetensors", "b.safetensors"], ["c.safetensors"])


from comfy_client import UPSCALE_INPUT_NAME, build_upscale_workflow


def test_upscale_workflow_shape() -> None:
    params = GenerationParams(prompt="a fox", negative_prompt="blurry", seed=5)
    graph = build_upscale_workflow("DreamShaper_8_pruned.safetensors", UPSCALE_INPUT_NAME, params, cfg=7.0)
    by_class = {node["class_type"]: node["inputs"] for node in graph.values()}
    assert by_class["CheckpointLoaderSimple"]["ckpt_name"] == "DreamShaper_8_pruned.safetensors"
    assert by_class["LoadImage"]["image"] == UPSCALE_INPUT_NAME
    assert by_class["ImageScaleBy"]["scale_by"] == 2.0
    assert "VAEEncodeTiled" in by_class and "VAEDecodeTiled" in by_class  # tiled = low VRAM
    sampler = by_class["KSampler"]
    assert (sampler["seed"], sampler["denoise"], sampler["cfg"]) == (5, 0.4, 7.0)
    texts = {graph[sampler[key][0]]["inputs"]["text"] for key in ("positive", "negative")}
    assert texts == {"a fox", "blurry"}
    assert by_class["SaveImage"]["filename_prefix"] == "ComfyUI_upscaled"


def test_upload_image_multipart(tmp_path) -> None:
    source = tmp_path / "src.png"
    source.write_bytes(b"\x89PNG fake")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["type"] = request.headers["content-type"]
        seen["body"] = request.content
        return httpx.Response(200, json={"name": UPSCALE_INPUT_NAME, "subfolder": "", "type": "input"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            return await ComfyClient("http://comfy", http).upload_image(source, UPSCALE_INPUT_NAME)

    assert asyncio.run(scenario()) == UPSCALE_INPUT_NAME
    assert seen["path"] == "/upload/image" and seen["type"].startswith("multipart/form-data")
    assert b'name="overwrite"' in seen["body"] and b"PNG fake" in seen["body"]


from comfy_client import IMG2IMG_INPUT_NAME, build_img2img_workflow, fit_size


def test_img2img_workflow_shape() -> None:
    params = GenerationParams(prompt="make it winter", negative_prompt="blurry", steps=25, cfg=7.0, seed=3, batch=1)
    graph = build_img2img_workflow("DreamShaper_8_pruned.safetensors", IMG2IMG_INPUT_NAME, params, strength=0.55)
    by_class = {node["class_type"]: node["inputs"] for node in graph.values()}
    assert by_class["LoadImage"]["image"] == IMG2IMG_INPUT_NAME
    assert "VAEEncodeTiled" in by_class and "VAEDecodeTiled" in by_class
    sampler = by_class["KSampler"]
    assert (sampler["denoise"], sampler["steps"], sampler["cfg"], sampler["seed"]) == (0.55, 25, 7.0, 3)
    assert "RepeatLatentBatch" not in by_class
    assert by_class["SaveImage"]["filename_prefix"] == "ComfyUI_img2img"
    four = build_img2img_workflow("x.safetensors", IMG2IMG_INPUT_NAME, replace(params, batch=4), strength=0.55)
    repeat = next(n["inputs"] for n in four.values() if n["class_type"] == "RepeatLatentBatch")
    assert repeat["amount"] == 4


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((4000, 3000), 768, (768, 576)),  # big phone photo: shrunk to the GPU-safe size
        ((3000, 4000), 1024, (768, 1024)),
        ((500, 333), 768, (512, 336)),  # small: grown to a usable size, multiples of 8
        ((640, 640), 768, (640, 640)),
    ],
)
def test_fit_size(size, max_side, expected) -> None:
    assert fit_size(*size, max_side=max_side) == expected


from comfy_client import build_inpaint_workflow, inpaint_masks


def test_inpaint_workflow_keeps_unmasked_pixels() -> None:
    params = GenerationParams(prompt="a red beanie hat", negative_prompt="blurry", steps=25, cfg=7.0, seed=4)
    graph = build_inpaint_workflow("inpaint.safetensors", "src.png", "hard.png", "soft.png", params)
    by_class: dict = {}
    for node_id, node in graph.items():
        by_class.setdefault(node["class_type"], []).append((node_id, node["inputs"]))
    assert {inputs["image"] for _, inputs in by_class["LoadImageMask"]} == {"hard.png", "soft.png"}
    (_, encode), = by_class["VAEEncodeForInpaint"]
    (_, sampler), = by_class["KSampler"]
    assert sampler["denoise"] == 1.0 and sampler["latent_image"][0] == by_class["VAEEncodeForInpaint"][0][0]
    (_, composite), = by_class["ImageCompositeMasked"]
    source_node = by_class["LoadImage"][0][0]
    assert composite["destination"] == [source_node, 0]  # the original is the base: unmasked pixels are untouched
    soft_node = next(nid for nid, inputs in by_class["LoadImageMask"] if inputs["image"] == "soft.png")
    assert composite["mask"] == [soft_node, 0]
    (_, save), = by_class["SaveImage"]
    assert save["filename_prefix"] == "ComfyUI_inpaint"


def test_inpaint_masks_are_resized_dilated_and_feathered(tmp_path) -> None:
    import io

    from PIL import Image, ImageDraw

    mask_path = tmp_path / "mask.png"
    mask = Image.new("RGB", (200, 100), "black")
    ImageDraw.Draw(mask).rectangle((90, 40, 110, 60), fill="white")
    mask.save(mask_path)
    hard_png, soft_png = inpaint_masks(mask_path, 400, 200)
    hard = Image.open(io.BytesIO(hard_png)).convert("L")
    soft = Image.open(io.BytesIO(soft_png)).convert("L")
    assert hard.size == soft.size == (400, 200)
    assert {v for v, count in enumerate(hard.histogram()) if count} <= {0, 255}  # binary for the encoder
    assert hard.getpixel((200, 100)) == 255 and hard.getpixel((10, 10)) == 0
    assert hard.getpixel((176, 100)) == 255  # grown a little beyond the painted area (180)
    assert 0 < soft.getpixel((172, 100)) < 255  # feathered edge for a seamless blend


def test_empty_mask_is_rejected(tmp_path) -> None:
    from PIL import Image

    from comfy_client import InvalidParamsError

    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (64, 64), "black").save(mask_path)
    with pytest.raises(InvalidParamsError, match="Paint over"):
        inpaint_masks(mask_path, 64, 64)
