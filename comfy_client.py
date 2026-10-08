"""ComfyUI client: workflow patching, queueing, waiting, and fetching results.

Independent of MCP so it can be tested and reused directly.
"""

from __future__ import annotations

import asyncio
import copy
from io import BytesIO
import json
import logging
import random
import struct
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import websockets
from PIL import Image, ImageFilter

from config import Settings

log = logging.getLogger(__name__)

Workflow = dict[str, dict[str, Any]]
ProgressCallback = Callable[[int, int], None]
PreviewCallback = Callable[[bytes], None]

MAX_SEED = 2**32 - 1
POLL_INTERVAL = 1.0
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, bad anatomy, text, watermark"


# --- Errors -----------------------------------------------------------------


class ComfyUIError(Exception):
    """Base error; messages are written to be shown to the LLM as-is."""


class ComfyUIUnavailableError(ComfyUIError):
    pass


class WorkflowError(ComfyUIError):
    pass


class GenerationTimeoutError(ComfyUIError):
    pass


class GenerationFailedError(ComfyUIError):
    pass


class InvalidParamsError(ComfyUIError):
    pass


# --- Parameters ---------------------------------------------------------------


@dataclass
class GenerationParams:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 8.0
    seed: int | None = None
    model: str | None = None  # checkpoint filename; None keeps the workflow's
    batch: int = 1

    def validate(self) -> None:
        problems: list[str] = []
        if not self.prompt.strip():
            problems.append("prompt must not be empty")
        for name, value in (("width", self.width), ("height", self.height)):
            if not 64 <= value <= 2048:
                problems.append(f"{name} must be between 64 and 2048 (got {value})")
            elif value % 8:
                problems.append(f"{name} must be a multiple of 8 (got {value})")
        if not 1 <= self.steps <= 150:
            problems.append(f"steps must be between 1 and 150 (got {self.steps})")
        if not 1.0 <= self.cfg <= 30.0:
            problems.append(f"cfg must be between 1.0 and 30.0 (got {self.cfg})")
        if self.seed is not None and not 0 <= self.seed <= MAX_SEED:
            problems.append(f"seed must be between 0 and {MAX_SEED} (got {self.seed})")
        if not 1 <= self.batch <= 4:
            problems.append(f"batch must be between 1 and 4 (got {self.batch})")
        if problems:
            raise InvalidParamsError("Invalid parameters: " + "; ".join(problems) + ".")

    def with_seed(self) -> GenerationParams:
        """Return a copy with a concrete seed (random if none was given)."""
        seed = self.seed if self.seed is not None else random.randint(0, MAX_SEED)
        return GenerationParams(**{**asdict(self), "seed": seed})


@dataclass
class GenerationResult:
    image: bytes
    saved_path: Path | None
    params: GenerationParams
    prompt_id: str
    elapsed: float
    comfy_file: dict[str, str]
    comfy_files: list[dict[str, str]] = field(default_factory=list)


# --- Workflow handling --------------------------------------------------------


def load_workflow(path: Path) -> Workflow:
    """Read the API-format workflow from disk. Each call returns a fresh object."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise WorkflowError(f"Workflow file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Workflow file is not valid JSON ({path}): {exc}") from exc
    if not isinstance(data, dict) or not all(
        isinstance(node, dict) and "class_type" in node for node in data.values()
    ):
        raise WorkflowError(
            f"{path.name} is not in ComfyUI API format. In ComfyUI use "
            "Workflow > Export (API) to save it."
        )
    return data


def find_nodes(workflow: Workflow, class_type: str) -> list[str]:
    return [node_id for node_id, node in workflow.items() if node["class_type"] == class_type]


def find_single_node(workflow: Workflow, class_type: str) -> str:
    matches = find_nodes(workflow, class_type)
    if len(matches) != 1:
        raise WorkflowError(
            f"Expected exactly one {class_type} node in the workflow, found {len(matches)}."
        )
    return matches[0]


def linked_node(workflow: Workflow, node_id: str, input_name: str, expected_class: str) -> str:
    """Follow `node_id`'s input link and check it points to a node of `expected_class`."""
    link = workflow[node_id]["inputs"].get(input_name)
    if not (isinstance(link, list) and len(link) == 2):
        raise WorkflowError(f"Node {node_id} input '{input_name}' is not connected to a node.")
    source_id = str(link[0])
    source = workflow.get(source_id)
    if source is None or source["class_type"] != expected_class:
        found = source["class_type"] if source else "nothing"
        raise WorkflowError(
            f"KSampler '{input_name}' should come from a {expected_class} node, "
            f"but is connected to {found} (node {source_id})."
        )
    return source_id


@dataclass(frozen=True)
class NodeRoles:
    sampler: str
    positive: str
    negative: str
    latent: str
    save: str


def identify_nodes(workflow: Workflow) -> NodeRoles:
    sampler = find_single_node(workflow, "KSampler")
    positive = linked_node(workflow, sampler, "positive", "CLIPTextEncode")
    negative = linked_node(workflow, sampler, "negative", "CLIPTextEncode")
    if positive == negative:
        raise WorkflowError("KSampler positive and negative use the same CLIPTextEncode node.")
    latent = find_single_node(workflow, "EmptyLatentImage")
    save = find_single_node(workflow, "SaveImage")
    return NodeRoles(sampler, positive, negative, latent, save)


def apply_params(template: Workflow, params: GenerationParams) -> tuple[Workflow, NodeRoles]:
    """Return a patched deep copy of `template`; the template is never mutated."""
    workflow = copy.deepcopy(template)
    roles = identify_nodes(workflow)
    workflow[roles.positive]["inputs"]["text"] = params.prompt
    workflow[roles.negative]["inputs"]["text"] = params.negative_prompt
    workflow[roles.latent]["inputs"].update(width=params.width, height=params.height)
    workflow[roles.sampler]["inputs"].update(steps=params.steps, cfg=params.cfg, seed=params.seed)
    workflow[roles.latent]["inputs"]["batch_size"] = params.batch
    if params.model:
        workflow[find_single_node(workflow, "CheckpointLoaderSimple")]["inputs"]["ckpt_name"] = params.model
    return workflow, roles


# --- ComfyUI HTTP / WebSocket client ------------------------------------------


class ComfyClient:
    def __init__(self, base_url: str, http: httpx.AsyncClient) -> None:
        self.base_url = base_url
        self.http = http
        self.client_id = uuid.uuid4().hex

    @property
    def ws_url(self) -> str:
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        return f"{scheme}://{parsed.netloc}/ws?clientId={self.client_id}"

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            return await self.http.request(method, f"{self.base_url}{path}", **kwargs)
        except httpx.ConnectError as exc:
            raise ComfyUIUnavailableError(
                f"Cannot connect to ComfyUI at {self.base_url}. Is ComfyUI running?"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ComfyUIUnavailableError(
                f"ComfyUI at {self.base_url} did not respond in time ({path})."
            ) from exc

    async def queue_prompt(self, workflow: Workflow, previews: bool = False) -> str:
        payload: dict[str, Any] = {"prompt": workflow, "client_id": self.client_id}
        if previews:  # per-prompt override; ComfyUI's own preview setting stays untouched
            payload["extra_data"] = {"preview_method": PREVIEW_METHOD}
        response = await self._request("POST", "/prompt", json=payload)
        if response.status_code == 400:
            raise WorkflowError("ComfyUI rejected the workflow: " + _describe_rejection(response))
        response.raise_for_status()
        return response.json()["prompt_id"]

    async def list_checkpoints(self) -> list[str]:
        response = await self._request("GET", "/object_info/CheckpointLoaderSimple")
        response.raise_for_status()
        spec = response.json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"]
        if spec and spec[0] == "COMBO":  # newer ComfyUI: ["COMBO", {"options": [...]}]
            return list(spec[1].get("options", []))
        return list(spec[0])

    async def upload_image(self, path: Path, name: str) -> str:
        return await self.upload_bytes(path.read_bytes(), name)

    async def upload_bytes(self, data: bytes, name: str) -> str:
        files = {"image": (name, data, "image/png")}
        response = await self._request("POST", "/upload/image", files=files, data={"overwrite": "true", "type": "input"})
        response.raise_for_status()
        return response.json()["name"]

    async def get_history(self, prompt_id: str) -> dict[str, Any] | None:
        response = await self._request("GET", f"/history/{prompt_id}")
        response.raise_for_status()
        return response.json().get(prompt_id)

    async def fetch_image(self, image_ref: dict[str, str]) -> bytes:
        params = {k: image_ref.get(k, "") for k in ("filename", "subfolder", "type")}
        response = await self._request("GET", "/view", params=params)
        response.raise_for_status()
        return response.content

    async def wait_for_completion(
        self, prompt_id: str, ws: Any, timeout: float, on_progress: ProgressCallback | None = None,
        on_preview: PreviewCallback | None = None,
    ) -> dict[str, Any]:
        """Wait via WebSocket if available, falling back to polling /history."""
        deadline = time.monotonic() + timeout
        if ws is not None:
            try:
                await asyncio.wait_for(_wait_ws(ws, prompt_id, on_progress, on_preview), timeout)
            except TimeoutError:
                raise _timeout_error(timeout) from None
            except websockets.ConnectionClosed:
                log.warning("WebSocket closed early; falling back to polling /history")
        return await self._poll_history(prompt_id, deadline, timeout)

    async def _poll_history(self, prompt_id: str, deadline: float, timeout: float) -> dict[str, Any]:
        while True:
            entry = await self.get_history(prompt_id)
            if entry and entry.get("status", {}).get("completed") is not None:
                if entry["status"].get("status_str") == "error" or entry.get("outputs"):
                    return entry
            if time.monotonic() >= deadline:
                raise _timeout_error(timeout)
            await asyncio.sleep(POLL_INTERVAL)


async def _wait_ws(
    ws: Any, prompt_id: str, on_progress: ProgressCallback | None, on_preview: PreviewCallback | None = None
) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):  # binary preview frames
            if on_preview is not None and (image := decode_preview(raw, prompt_id)) is not None:
                on_preview(image)
            continue
        message = json.loads(raw)
        data = message.get("data", {})
        if data.get("prompt_id") != prompt_id:
            continue
        kind = message.get("type")
        if kind == "progress" and on_progress is not None:
            on_progress(int(data.get("value", 0)), int(data.get("max", 0)))
        elif kind == "execution_error":
            raise GenerationFailedError(_describe_execution_error(data))
        elif kind == "execution_success" or (kind == "executing" and data.get("node") is None):
            return


async def _open_ws(url: str) -> Any:
    try:
        return await websockets.connect(url, max_size=None, open_timeout=5)
    except (OSError, websockets.WebSocketException, TimeoutError) as exc:
        log.warning("WebSocket unavailable (%s); will poll /history instead", exc)
        return None


PREVIEW_METHOD = "auto"  # TAESD when its decoder is in models/vae_approx, else ComfyUI falls back to latent2rgb
PREVIEW_IMAGE = 1  # ComfyUI BinaryEventTypes
PREVIEW_IMAGE_WITH_METADATA = 4


def decode_preview(frame: bytes, prompt_id: str) -> bytes | None:
    """Return the image bytes of a ComfyUI binary preview frame, or None for other frames."""
    if len(frame) < 8:
        return None
    event, header = struct.unpack(">II", frame[:8])
    if event == PREVIEW_IMAGE:  # header is the image type (1 JPEG, 2 PNG)
        return frame[8:]
    if event == PREVIEW_IMAGE_WITH_METADATA:  # header is the metadata JSON length
        try:
            metadata = json.loads(frame[8 : 8 + header])
        except ValueError:
            return None
        if metadata.get("prompt_id", prompt_id) != prompt_id:
            return None
        return frame[8 + header :]
    return None


OUTPUT_DIR_FLAG = "--output-directory"


def output_dir_from_argv(argv: list[str]) -> Path | None:
    """Find ComfyUI's --output-directory in its launch arguments."""
    for index, arg in enumerate(argv):
        if arg == OUTPUT_DIR_FLAG and index + 1 < len(argv):
            return Path(argv[index + 1])
        if arg.startswith(OUTPUT_DIR_FLAG + "="):
            return Path(arg.split("=", 1)[1])
    return None


async def get_output_dir(base_url: str) -> Path | None:
    try:
        async with httpx.AsyncClient(timeout=3) as http:
            response = await http.get(f"{base_url}/system_stats")
            response.raise_for_status()
    except httpx.HTTPError:
        return None
    return output_dir_from_argv(response.json().get("system", {}).get("argv", []))


def _timeout_error(timeout: float) -> GenerationTimeoutError:
    return GenerationTimeoutError(
        f"Image generation did not finish within {timeout:.0f}s. ComfyUI may be busy "
        "or the model is still loading; try again, or lower steps/size."
    )


def _describe_rejection(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:500]
    parts = [body.get("error", {}).get("message", "unknown error")]
    for node_id, node_err in body.get("node_errors", {}).items():
        for err in node_err.get("errors", []):
            parts.append(f"node {node_id} ({node_err.get('class_type')}): {err.get('details') or err.get('message')}")
    return "; ".join(parts)


def _describe_execution_error(data: dict[str, Any]) -> str:
    return (
        f"ComfyUI failed while running {data.get('node_type', 'a node')} "
        f"(node {data.get('node_id')}): {data.get('exception_message', '').strip()}"
    )


def _output_images(entry: dict[str, Any], preferred_node: str) -> list[dict[str, str]]:
    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for name, data in status.get("messages", []):
            if name == "execution_error":
                raise GenerationFailedError(_describe_execution_error(data))
        raise GenerationFailedError("ComfyUI reported an error while generating the image.")
    outputs = entry.get("outputs", {})
    ordered = [preferred_node, *(k for k in outputs if k != preferred_node)]
    for node_id in ordered:
        images = outputs.get(node_id, {}).get("images", [])
        if images:
            return images
    raise GenerationFailedError("ComfyUI finished but produced no output image.")


# --- Saving -------------------------------------------------------------------


def save_output(output_dir: Path, image: bytes, metadata: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{datetime.now():%Y%m%d-%H%M%S}_{metadata['params']['seed']}"
    image_path = output_dir / f"{stem}.png"
    image_path.write_bytes(image)
    sidecar = {**metadata, "image": image_path.name}
    image_path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return image_path


# --- Entry point --------------------------------------------------------------


async def generate(
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    save_copy: bool = True,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Run one txt2img generation end to end and save the result locally."""
    params.validate()
    params = params.with_seed()
    workflow, roles = apply_params(load_workflow(settings.workflow_path), params)
    return await _run(workflow, roles.save, params, settings, on_progress, on_preview, save_copy)


async def upscale(
    source: Path,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Upscale an existing PNG ×2 with a light img2img pass (tiled VAE keeps VRAM low).

    `params` carries the prompt, negative prompt, seed, cfg and checkpoint (`model`) to refine with.
    """
    if not params.model:
        raise InvalidParamsError("Upscaling needs a model.")
    params = params.with_seed()
    async with httpx.AsyncClient(timeout=30) as http:
        image_name = await ComfyClient(settings.comfyui_url, http).upload_image(source, UPSCALE_INPUT_NAME)
    workflow = build_upscale_workflow(params.model, image_name, params, params.cfg)
    result = await _run(workflow, UPSCALE_SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)
    result.params = replace(params, width=params.width * 2, height=params.height * 2, steps=UPSCALE_STEPS)
    return result


async def img2img(
    source: Path,
    params: GenerationParams,
    strength: float,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Repaint an existing picture following the prompt; `strength` (denoise) sets how much it may change.

    The source is resized to params.width × params.height first, so any photo stays within GPU-safe sizes.
    """
    if not params.model:
        raise InvalidParamsError("Image-to-image needs a model.")
    params.validate()
    params = params.with_seed()
    data = _resized_png(source, params.width, params.height)
    async with httpx.AsyncClient(timeout=30) as http:
        image_name = await ComfyClient(settings.comfyui_url, http).upload_bytes(data, IMG2IMG_INPUT_NAME)
    workflow = build_img2img_workflow(params.model, image_name, params, strength)
    return await _run(workflow, IMG2IMG_SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)


def fit_size(width: int, height: int, max_side: int, min_side: int = 512) -> tuple[int, int]:
    """Scale (width, height) so the long side is between min_side and max_side; multiples of 8."""
    long_side = max(width, height)
    scale = min(max(long_side, min_side), max_side) / long_side
    return (round(width * scale) // 8 * 8, round(height * scale) // 8 * 8)


def _resized_png(source: Path, width: int, height: int) -> bytes:
    with Image.open(source) as img:
        return _png(img.convert("RGB").resize((width, height), Image.Resampling.LANCZOS))


IMG2IMG_INPUT_NAME = "studio_img2img_src.png"
IMG2IMG_SAVE_NODE = "9"


def build_img2img_workflow(ckpt: str, image_name: str, params: GenerationParams, strength: float) -> Workflow:
    latent: list[Any] = ["3", 0]
    graph: Workflow = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["2", 0], "vae": ["1", 2], **TILED}},
    }
    if params.batch > 1:
        graph["4"] = {"class_type": "RepeatLatentBatch", "inputs": {"samples": ["3", 0], "amount": params.batch}}
        latent = ["4", 0]
    graph.update({
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["1", 1]}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": params.seed, "steps": params.steps, "cfg": params.cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": strength,
                "positive": ["5", 0], "negative": ["6", 0], "latent_image": latent,
            },
        },
        "8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["1", 2], **TILED}},
        IMG2IMG_SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI_img2img"}},
    })
    return graph


async def inpaint(
    source: Path,
    mask: Path,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None = None,
    on_preview: PreviewCallback | None = None,
) -> GenerationResult:
    """Regenerate only the painted (white) area of `mask`; everything else keeps the original pixels."""
    if not params.model:
        raise InvalidParamsError("Inpainting needs a model.")
    params.validate()
    params = params.with_seed()
    hard_mask, soft_mask = inpaint_masks(mask, params.width, params.height)
    async with httpx.AsyncClient(timeout=30) as http:
        client = ComfyClient(settings.comfyui_url, http)
        image_name = await client.upload_bytes(_resized_png(source, params.width, params.height), INPAINT_INPUT_NAME)
        hard_name = await client.upload_bytes(hard_mask, INPAINT_MASK_NAME)
        soft_name = await client.upload_bytes(soft_mask, INPAINT_SOFT_MASK_NAME)
    workflow = build_inpaint_workflow(params.model, image_name, hard_name, soft_name, params)
    return await _run(workflow, INPAINT_SAVE_NODE, params, settings, on_progress, on_preview, save_copy=False)


INPAINT_INPUT_NAME = "studio_inpaint_src.png"
INPAINT_MASK_NAME = "studio_inpaint_mask.png"
INPAINT_SOFT_MASK_NAME = "studio_inpaint_mask_soft.png"
INPAINT_SAVE_NODE = "9"
INPAINT_GROW_PX = 8  # the encoder sees a little more than the brush, so new content blends in
INPAINT_FEATHER_PX = 4


def inpaint_masks(mask: Path, width: int, height: int) -> tuple[bytes, bytes]:
    """(hard, soft) masks as PNGs at the target size: hard (grown, binary) for the encoder, soft for blending."""
    with Image.open(mask) as img:
        painted = img.convert("L").resize((width, height), Image.Resampling.NEAREST).point(lambda v: 255 if v > 127 else 0)
    if painted.getbbox() is None:
        raise InvalidParamsError("Paint over the area you want to change first.")
    hard = painted.filter(ImageFilter.MaxFilter(INPAINT_GROW_PX * 2 + 1))
    soft = hard.filter(ImageFilter.GaussianBlur(INPAINT_FEATHER_PX))
    return _png(hard.convert("RGB")), _png(soft.convert("RGB"))


def _png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def build_inpaint_workflow(ckpt: str, image_name: str, hard_mask: str, soft_mask: str, params: GenerationParams) -> Workflow:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "LoadImageMask", "inputs": {"image": hard_mask, "channel": "red"}},
        "4": {"class_type": "LoadImageMask", "inputs": {"image": soft_mask, "channel": "red"}},
        "5": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["2", 0], "vae": ["1", 2], "mask": ["3", 0], "grow_mask_by": 0}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["1", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["1", 1]}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": params.seed, "steps": params.steps, "cfg": params.cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0],
            },
        },
        "10": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["8", 0], "vae": ["1", 2], **TILED}},
        # paste the new area onto the original, so pixels outside the mask are exactly the original ones
        "11": {
            "class_type": "ImageCompositeMasked",
            "inputs": {"destination": ["2", 0], "source": ["10", 0], "x": 0, "y": 0, "resize_source": False, "mask": ["4", 0]},
        },
        INPAINT_SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "ComfyUI_inpaint"}},
    }


UPSCALE_INPUT_NAME = "studio_upscale_src.png"
UPSCALE_SAVE_NODE = "9"
UPSCALE_STEPS = 15
UPSCALE_DENOISE = 0.4  # enough to add detail without changing the picture
TILED = {"tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}


def build_upscale_workflow(ckpt: str, image_name: str, params: GenerationParams, cfg: float) -> Workflow:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "ImageScaleBy", "inputs": {"image": ["2", 0], "upscale_method": "lanczos", "scale_by": 2.0}},
        "4": {"class_type": "VAEEncodeTiled", "inputs": {"pixels": ["3", 0], "vae": ["1", 2], **TILED}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": params.prompt, "clip": ["1", 1]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": params.negative_prompt, "clip": ["1", 1]}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "seed": params.seed, "steps": UPSCALE_STEPS, "cfg": cfg,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": UPSCALE_DENOISE,
                "positive": ["5", 0], "negative": ["6", 0], "latent_image": ["4", 0],
            },
        },
        "8": {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["7", 0], "vae": ["1", 2], **TILED}},
        UPSCALE_SAVE_NODE: {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI_upscaled"}},
    }


async def _run(
    workflow: Workflow,
    save_node: str,
    params: GenerationParams,
    settings: Settings,
    on_progress: ProgressCallback | None,
    on_preview: PreviewCallback | None,
    save_copy: bool,
) -> GenerationResult:
    """Queue a workflow, wait for it, and fetch its first output image."""
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=30) as http:
        client = ComfyClient(settings.comfyui_url, http)
        ws = await _open_ws(client.ws_url)  # connect before queueing so no events are missed
        try:
            prompt_id = await client.queue_prompt(workflow, previews=on_preview is not None)
            entry = await client.wait_for_completion(prompt_id, ws, settings.timeout, on_progress, on_preview)
        finally:
            if ws is not None:
                await ws.close()
        image_refs = _output_images(entry, save_node)
        image_ref = image_refs[0]
        image = await client.fetch_image(image_ref)

    elapsed = time.monotonic() - started
    saved_path = None
    if save_copy:
        saved_path = save_output(
            settings.output_dir,
            image,
            {
                "params": asdict(params),
                "prompt_id": prompt_id,
                "comfyui_file": image_ref,
                "workflow": str(settings.workflow_path),
                "elapsed_seconds": round(elapsed, 2),
                "created": datetime.now().isoformat(timespec="seconds"),
            },
        )
    return GenerationResult(
        image=image,
        saved_path=saved_path,
        params=params,
        prompt_id=prompt_id,
        elapsed=elapsed,
        comfy_file=image_ref,
        comfy_files=image_refs,
    )
