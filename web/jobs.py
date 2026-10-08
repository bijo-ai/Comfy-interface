"""Single active generation job with a replayable event log for SSE subscribers."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from comfy_client import ComfyUIError, GenerationParams, PreviewCallback, ProgressCallback

log = logging.getLogger(__name__)

Runner = Callable[[GenerationParams, ProgressCallback, PreviewCallback], Awaitable[dict[str, Any]]]
KEEP_FINISHED_JOBS = 20


class BusyError(Exception):
    pass


class Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.preview: bytes | None = None  # latest live preview frame; not part of the event log
        self.task: asyncio.Task[None] | None = None
        self._changed = asyncio.Event()

    def emit(self, event: dict[str, Any], final: bool = False) -> None:
        self.events.append(event)
        self.done = self.done or final
        self._changed.set()
        self._changed = asyncio.Event()

    async def stream(self) -> AsyncIterator[dict[str, Any]]:
        """Yield every event from the beginning, then live ones until the job finishes."""
        index = 0
        while True:
            while index < len(self.events):
                yield self.events[index]
                index += 1
            if self.done:
                return
            await self._changed.wait()


class JobManager:
    def __init__(self, runner: Runner) -> None:
        self._runner = runner
        self._jobs: dict[str, Job] = {}
        self._active: Job | None = None

    def start(self, params: GenerationParams) -> Job:
        if self._active is not None and not self._active.done:
            raise BusyError("An image is already being generated. Wait for it to finish.")
        job = Job(uuid.uuid4().hex)
        self._jobs[job.id] = job
        self._active = job
        job.task = asyncio.create_task(self._run(job, params))
        self._prune()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def _run(self, job: Job, params: GenerationParams) -> None:
        def on_progress(step: int, total: int) -> None:
            job.emit({"type": "progress", "step": step, "total": total, "preview": job.preview is not None})

        def on_preview(image: bytes) -> None:
            job.preview = image

        try:
            payload = await self._runner(params, on_progress, on_preview)
            job.emit({"type": "done", **payload}, final=True)
        except ComfyUIError as exc:
            job.emit({"type": "error", "message": str(exc)}, final=True)
        except Exception as exc:  # never let a job kill the server
            log.exception("Generation job failed")
            job.emit({"type": "error", "message": f"Unexpected error while generating: {exc}"}, final=True)

    def _prune(self) -> None:
        finished = [job_id for job_id, job in self._jobs.items() if job.done]
        for job_id in finished[:-KEEP_FINISHED_JOBS]:
            del self._jobs[job_id]
