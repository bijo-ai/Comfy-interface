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
    assert bodies[1]["extra_data"] == {"preview_method": "latent2rgb"}
