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
