"""FastAPI app: generation jobs with live progress, plus a gallery over ComfyUI's output folder."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from comfy_client import (
    DEFAULT_NEGATIVE,
    GenerationParams,
    InvalidParamsError,
    PreviewCallback,
    ProgressCallback,
    generate,
    get_output_dir,
    img2img,
    inpaint,
    upscale,
)
from config import Settings, load_settings
from models import INPAINT_KEY, PROFILES, profile_by_key
from styles import STYLES
from web.bot import BotRunner, build_application, register_commands
from web.bot_core import StudioBot
from web.builders import (
    DEFAULT_STRENGTH,
    build_generation,
    build_img2img,
    build_inpaint,
    build_upscale,
    resolve_profile,
)
from web.catalog import ModelCatalog
from web.gallery import KINDS, Gallery
from web.jobs import BusyError, Img2ImgRequest, InpaintRequest, JobManager, JobRequest, Runner, UpscaleRequest
from web.sources import SourceStore
from web.users import UserStore

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]  # blocks DNS-rebinding pages from reaching the API
MAX_PAGE = 200
PNG_SIGNATURE = bytes([0x89]) + b"PNG"
NO_GALLERY = (
    "ComfyUI's output folder is unknown. Start ComfyUI, or set COMFYUI_OUTPUT_DIR in .env."
)


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 8.0
    seed: int | None = None
    model: str | None = None  # profile key; None = default model
    style: str = "none"
    batch: int = 1
    source_id: str | None = None  # uploaded start image (image-to-image)
    source_name: str | None = None  # or a gallery image as the start image
    strength: float = DEFAULT_STRENGTH


class UpscaleBody(BaseModel):
    name: str


class InpaintBody(BaseModel):
    prompt: str
    mask_id: str  # uploaded mask: white = repaint
    source_id: str | None = None
    source_name: str | None = None
    negative_prompt: str = DEFAULT_NEGATIVE
    style: str = "none"
    seed: int | None = None


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class GalleryProvider:
    """Resolves ComfyUI's output folder lazily, so the app can start before ComfyUI does."""

    def __init__(self, settings: Settings, output_dir: Path | None) -> None:
        self._settings = settings
        self.output_dir = output_dir or settings.comfyui_output_dir
        self._gallery: Gallery | None = None

    async def get(self) -> Gallery | None:
        if self._gallery is None:
            if self.output_dir is None:
                self.output_dir = await get_output_dir(self._settings.comfyui_url)
            if self.output_dir is not None:
                self._gallery = Gallery(self.output_dir, self._settings.cache_dir)
        return self._gallery


def comfy_name(comfy_file: dict[str, str]) -> str:
    parts = (comfy_file.get("subfolder", ""), comfy_file["filename"])
    return "/".join(part for part in parts if part).replace("\\", "/")


def make_runner(settings: Settings, galleries: GalleryProvider) -> Runner:
    async def run(request: JobRequest, on_progress: ProgressCallback, on_preview: PreviewCallback) -> dict[str, Any]:
        if isinstance(request, UpscaleRequest):
            result = await upscale(request.source, request.params, settings, on_progress, on_preview)
        elif isinstance(request, Img2ImgRequest):
            result = await img2img(request.source, request.params, request.strength, settings, on_progress, on_preview)
        elif isinstance(request, InpaintRequest):
            result = await inpaint(request.source, request.mask, request.params, settings, on_progress, on_preview)
        else:
            result = await generate(request, settings, on_progress=on_progress, save_copy=False, on_preview=on_preview)
        gallery = await galleries.get()
        images = []
        for comfy_file in result.comfy_files if gallery is not None else []:
            try:
                images.append(gallery.get(comfy_name(comfy_file)).to_json())
            except (ValueError, FileNotFoundError):
                log.warning("Generated file %s not found in gallery", comfy_file)
        return {
            "image": images[0] if images else None,
            "images": images,
            "seed": result.params.seed,
            "elapsed": round(result.elapsed, 1),
        }

    return run


@contextmanager
def image_errors() -> Iterator[None]:
    try:
        yield
    except ValueError as exc:
        raise ApiError(400, "Invalid image name.") from exc
    except FileNotFoundError as exc:
        raise ApiError(404, "Image not found.") from exc
    except OSError as exc:
        raise ApiError(404, "Image could not be read.") from exc


def create_app(
    settings: Settings | None = None,
    runner: Runner | None = None,
    output_dir: Path | None = None,
    catalog: ModelCatalog | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    galleries = GalleryProvider(settings, output_dir)
    catalog = catalog or ModelCatalog(settings.comfyui_url)
    sources = SourceStore(settings.cache_dir / "sources")
    jobs = JobManager(runner or make_runner(settings, galleries))

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        bot_runner = None
        if settings.telegram_bot_token:
            studio = StudioBot(
                jobs, galleries, settings.telegram_allowed_user_id, catalog,
                UserStore(settings.telegram_users_path), sources=sources,
            )
            bot_runner = BotRunner(
                lambda: build_application(settings.telegram_bot_token, studio), on_started=register_commands
            )
            bot_runner.start()
        yield
        if bot_runner is not None:
            await bot_runner.stop()

    app = FastAPI(title="LUMOS Studios", docs_url=None, redoc_url=None, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(ApiError)
    async def on_api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc'][1:])}: {err['msg']}" for err in exc.errors()
        )
        return JSONResponse({"error": f"Invalid request: {details}"}, status_code=422)

    async def require_gallery() -> Gallery:
        gallery = await galleries.get()
        if gallery is None:
            raise ApiError(503, NO_GALLERY)
        return gallery

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=2) as http:
                online = (await http.get(f"{settings.comfyui_url}/system_stats")).is_success
        except httpx.HTTPError:
            online = False
        if online:
            await galleries.get()
        return {"comfyui": online, "output_dir": str(galleries.output_dir) if galleries.output_dir else None}

    def resolve_source(gallery: Gallery, source_id: str | None, source_name: str | None) -> Path | None:
        with image_errors():
            if source_id:
                return sources.path(source_id)
            if source_name:
                path = gallery.path(source_name)
                if not path.is_file():
                    raise FileNotFoundError(source_name)
                return path
        return None

    @app.post("/api/inpaint")
    async def start_inpaint(body: InpaintBody) -> dict[str, str]:
        gallery = await require_gallery()
        source = resolve_source(gallery, body.source_id, body.source_name)
        if source is None:
            raise ApiError(422, "Choose the image to fix.")
        with image_errors():
            mask = sources.path(body.mask_id)
        try:
            request = build_inpaint(
                source=source, mask=mask, prompt=body.prompt, negative_prompt=body.negative_prompt,
                style=body.style, seed=body.seed, available=await catalog.available(),
            )
        except InvalidParamsError as exc:
            raise ApiError(422, str(exc)) from exc
        return start_job(request)

    @app.post("/api/generate")
    async def start_generation(body: GenerateRequest) -> dict[str, str]:
        gallery = await require_gallery()
        source = resolve_source(gallery, body.source_id, body.source_name)
        try:
            profile = resolve_profile(body.model, await catalog.available())
            common = dict(
                prompt=body.prompt, negative_prompt=body.negative_prompt, steps=body.steps, cfg=body.cfg,
                seed=body.seed, batch=body.batch, style=body.style, profile=profile,
            )
            if source is not None:
                request = build_img2img(source=source, strength=body.strength, **common)
            else:
                request = build_generation(width=body.width, height=body.height, **common)
        except InvalidParamsError as exc:
            raise ApiError(422, str(exc)) from exc
        return start_job(request)

    @app.post("/api/sources")
    async def upload_source(request: Request) -> dict[str, Any]:
        try:
            source_id = sources.save(await request.body())
        except InvalidParamsError as exc:
            raise ApiError(422, str(exc)) from exc
        with Image.open(sources.path(source_id)) as img:
            width, height = img.size
        return {"id": source_id, "width": width, "height": height}

    @app.get("/api/sources/{source_id}")
    async def get_source(source_id: str) -> FileResponse:
        with image_errors():
            path = sources.path(source_id)
        return FileResponse(path, media_type="image/png")

    def start_job(request: JobRequest) -> dict[str, str]:
        try:
            job = jobs.start(request)
        except BusyError as exc:
            raise ApiError(409, str(exc)) from exc
        return {"job_id": job.id}

    @app.post("/api/upscale")
    async def start_upscale(body: UpscaleBody) -> dict[str, str]:
        gallery = await require_gallery()
        with image_errors():
            image = gallery.get(body.name)
        try:
            request = build_upscale(image, gallery.path(body.name), await catalog.available())
        except InvalidParamsError as exc:
            raise ApiError(422, str(exc)) from exc
        return start_job(request)

    @app.get("/api/models")
    async def list_models() -> dict[str, Any]:
        available = await catalog.available()
        default = resolve_profile(None, available)
        return {
            "models": [
                {
                    "key": p.key, "label": p.label, "ckpt": p.ckpt, "family": p.family, "heavy": p.heavy,
                    "shapes": {name: list(size) for name, size in p.shapes.items()},
                    "max_batch": p.max_batch, "upscale": p.upscale, "steps": p.steps, "cfg": p.cfg,
                    "available": p in available,
                }
                for p in PROFILES
                if p.selectable
            ],
            "default": default.key if default else None,
            "inpaint": profile_by_key(INPAINT_KEY) in available,
            "styles": [{"key": s.key, "label": s.label, "emoji": s.emoji} for s in STYLES],
        }

    @app.get("/api/jobs/{job_id}/events")
    async def job_events(job_id: str) -> StreamingResponse:
        job = jobs.get(job_id)
        if job is None:
            raise ApiError(404, "Unknown job.")

        async def stream() -> AsyncIterator[str]:
            async for event in job.stream():
                yield f"data: {json.dumps(event)}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/api/jobs/{job_id}/preview")
    async def job_preview(job_id: str) -> Response:
        job = jobs.get(job_id)
        if job is None or job.preview is None:
            raise ApiError(404, "No preview yet.")
        media_type = "image/png" if job.preview.startswith(PNG_SIGNATURE) else "image/jpeg"
        return Response(job.preview, media_type=media_type, headers={"Cache-Control": "no-store"})

    @app.get("/api/images")
    async def list_images(offset: int = 0, limit: int = 60, q: str = "", kind: str | None = None) -> dict[str, Any]:
        gallery = await require_gallery()
        if kind and kind not in KINDS:
            raise ApiError(422, f"Unknown kind {kind!r}. Use one of: {', '.join(KINDS)}.")
        total, images = gallery.page(max(offset, 0), min(max(limit, 1), MAX_PAGE), query=q, kind=kind or None)
        return {"total": total, "images": [image.to_json() for image in images]}

    @app.get("/api/images/counts")  # declared before /api/images/{name} so "counts" isn't taken as a file name
    async def image_counts() -> dict[str, int]:
        return (await require_gallery()).counts()

    @app.get("/api/images/{name:path}")
    async def get_image(name: str, download: bool = False) -> FileResponse:
        gallery = await require_gallery()
        with image_errors():
            path = gallery.path(name)
            if not path.is_file():
                raise FileNotFoundError(name)
        return FileResponse(path, media_type="image/png", filename=path.name if download else None)

    @app.get("/api/thumbs/{name:path}")
    async def get_thumbnail(name: str) -> FileResponse:
        gallery = await require_gallery()
        with image_errors():
            thumb = gallery.thumbnail(name)
        return FileResponse(thumb, media_type="image/webp")

    @app.delete("/api/images/{name:path}")
    async def delete_image(name: str) -> dict[str, str]:
        gallery = await require_gallery()
        with image_errors():
            gallery.delete(name)
        return {"deleted": name}

    return app
