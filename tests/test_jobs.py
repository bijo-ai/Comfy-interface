from __future__ import annotations

import asyncio

import pytest

from comfy_client import ComfyUIUnavailableError, GenerationParams
from web.jobs import BusyError, JobManager

PARAMS = GenerationParams(prompt="a fox")


async def collect(job) -> list[dict]:
    return [event async for event in job.stream()]


def test_progress_then_done() -> None:
    async def runner(params, on_progress):
        on_progress(1, 2)
        await asyncio.sleep(0)
        on_progress(2, 2)
        return {"seed": 5}

    async def scenario():
        job = JobManager(runner).start(PARAMS)
        return await collect(job)

    assert asyncio.run(scenario()) == [
        {"type": "progress", "step": 1, "total": 2},
        {"type": "progress", "step": 2, "total": 2},
        {"type": "done", "seed": 5},
    ]


def test_comfy_error_becomes_error_event() -> None:
    async def runner(params, on_progress):
        raise ComfyUIUnavailableError("Cannot connect to ComfyUI")

    async def scenario():
        return await collect(JobManager(runner).start(PARAMS))

    assert asyncio.run(scenario()) == [{"type": "error", "message": "Cannot connect to ComfyUI"}]


def test_unexpected_error_becomes_error_event() -> None:
    async def runner(params, on_progress):
        raise RuntimeError("boom")

    async def scenario():
        return await collect(JobManager(runner).start(PARAMS))

    events = asyncio.run(scenario())
    assert events[0]["type"] == "error" and "boom" in events[0]["message"]


def test_busy_while_running_then_free() -> None:
    async def runner(params, on_progress):
        await asyncio.sleep(0.05)
        return {}

    async def scenario():
        manager = JobManager(runner)
        job = manager.start(PARAMS)
        with pytest.raises(BusyError):
            manager.start(PARAMS)
        await collect(job)
        assert manager.start(PARAMS).id != job.id

    asyncio.run(scenario())


def test_late_subscriber_gets_replay() -> None:
    async def runner(params, on_progress):
        on_progress(1, 1)
        return {"seed": 1}

    async def scenario():
        manager = JobManager(runner)
        job = manager.start(PARAMS)
        await collect(job)  # finish first
        assert manager.get(job.id) is job
        return await collect(job)  # subscribe again after completion

    assert [e["type"] for e in asyncio.run(scenario())] == ["progress", "done"]
