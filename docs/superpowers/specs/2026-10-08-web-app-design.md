# ComfyUI Studio — local web app design

Date: 2026-10-08 · Status: approved in chat, pending spec review

## Goal

A modern, dark, single-page website running on Bijo's laptop that lets him type a prompt, watch it
generate with live progress, and browse every image ComfyUI has produced. It replaces the LM Studio
route (too heavy for a 6 GB GPU / 16 GB RAM laptop). A Telegram bot is a **separate follow-up
project** that will reuse the same generation code; it is out of scope here.

### Success criteria
- Prompt → image shown in the page, with a real step-by-step progress bar.
- Gallery shows all PNGs in ComfyUI's output folder (including ones made in the ComfyUI UI), newest first,
  fast even with hundreds of images.
- Per-image: open full size with prompt/settings, Reuse, Vary, Download, Delete.
- Clear message when ComfyUI is offline; the app never crashes on a failed generation.
- Starts with a double-click (`run_web.bat`), opens the browser.

### Non-goals (YAGNI)
Network/phone access, auth, multi-user, a generation queue, favorites, search, img2img, model switching,
live latent previews.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Stack | FastAPI + one static HTML/CSS/JS page | Python only, no Node build; reuses `comfy_client.py` directly |
| MCP | Not used by the web app | MCP is for LLM clients; `server.py` stays for later use |
| Gallery source | ComfyUI's output folder | Single source of truth; includes images made in ComfyUI itself |
| Image metadata | Parse the `prompt` tEXt chunk ComfyUI writes into each PNG (API-format graph) | Exact settings without sidecars; verified present on existing outputs |
| Progress transport | Server-Sent Events | One-way stream is all we need; simpler than WebSockets |
| Concurrency | One generation at a time | Avoids queue/timeout complexity; matches single-user use |
| Binding | `127.0.0.1` only | Local use; nothing exposed |

## Architecture

```
Browser (web/static/index.html, app.js, styles.css)
   │  fetch JSON  +  EventSource (SSE)
   ▼
web/app.py  (FastAPI, uvicorn, 127.0.0.1:7860)
   ├── web/jobs.py      single active job, progress events, result
   ├── web/gallery.py   list / metadata / thumbnails / delete over the output folder
   └── comfy_client.py  (existing) generate() with new progress callback
                           │ HTTP + WebSocket
                           ▼
                       ComfyUI 127.0.0.1:8188  ──writes──> ComfyUI output folder
```

### Units

**`comfy_client.py` (changes)**
- `generate(params, settings, on_progress=None, save_copy=True)`.
  - `on_progress(step: int, total: int)` is called for ComfyUI WebSocket `progress` messages matching our `prompt_id`.
  - `save_copy=False` skips writing to `outputs/` (the web app relies on ComfyUI's own saved file).
    The MCP server keeps the current default `True`.
- `GenerationResult` gains `comfy_file: dict` (the `filename/subfolder/type` from history) so the web app
  can point at the gallery entry. `saved_path` becomes `Path | None`.
- New `get_output_dir(base_url) -> Path | None`: reads `--output-directory` from `/system_stats` argv.

**`web/gallery.py`**
- `GalleryImage` dataclass: `name` (POSIX path relative to the output dir, e.g. `ComfyUI_00010_.png`
  or `sub/x.png`), `created` (mtime), `width`, `height`, and `params: dict | None`.
  `params` holds prompt, negative_prompt, width, height, steps, cfg and seed, or None if they can't be read.
- `list_images(output_dir) -> list[GalleryImage]`: recursive `*.png` scan, newest first. Metadata is cached
  in memory by `(path, mtime)`.
- `read_params(png_path) -> dict | None`: reads the `prompt` tEXt chunk with Pillow, then runs `identify_nodes`
  to extract the values. Any failure (no chunk, a different graph shape such as SDXL with refiner) → None.
- `thumbnail(output_dir, name, cache_dir) -> Path`: 320px-wide WebP in `cache/thumbs/`, regenerated when
  the source mtime changes.
- `resolve_safe(output_dir, name) -> Path`: rejects absolute paths, `..`, non-`.png`, and anything
  resolving outside `output_dir`. Raises `ValueError`. Used by serve, thumb and delete.
- `delete_image(output_dir, name)`: removes the PNG and its cached thumbnail.

**`web/jobs.py`**
- `JobManager` with at most one running job. `start(params) -> job_id` raises `BusyError` if one is running.
- Each job holds an `asyncio.Queue` of events consumed by the SSE endpoint:
  - `progress {step,total}`
  - `done {image: GalleryImage-json, seed, elapsed}`
  - `error {message}`
- Runs `comfy_client.generate(..., save_copy=False, on_progress=...)` as an asyncio task.
- A finished job is kept briefly so a late-connecting EventSource still gets the final event.

**`web/app.py`** (routes; all JSON errors are `{"error": "..."}`)

| Method | Path | Behaviour |
|---|---|---|
| GET | `/` | index.html; static files at `/static` |
| GET | `/api/status` | `{comfyui: bool, output_dir}`: pings ComfyUI `/system_stats` |
| POST | `/api/generate` | body = generate params → `{job_id}`; 409 if busy, 422 on invalid params (existing `validate()` messages) |
| GET | `/api/jobs/{id}/events` | SSE stream of progress / done / error |
| GET | `/api/images?offset=&limit=` | gallery page, newest first, plus `total` |
| GET | `/api/images/{name:path}` | full PNG (`?download=1` adds `Content-Disposition: attachment`) |
| GET | `/api/thumbs/{name:path}` | WebP thumbnail |
| DELETE | `/api/images/{name:path}` | deletes; 400 on unsafe name, 404 if missing |

Output dir resolution: `COMFYUI_OUTPUT_DIR` in `.env` → else `get_output_dir()` → else a startup error message.

### Frontend (`web/static/`)
Vanilla JS modules, no framework, no build. The look is dark and modern: a deep neutral background,
one accent colour, soft radii and restrained motion, with `prefers-reduced-motion` respected.
- **Composer**
  - Auto-growing prompt textarea; Ctrl+Enter generates.
  - Shape presets Square 512×512, Portrait 512×768, Landscape 768×512.
  - Collapsible Advanced panel: negative prompt, steps, CFG, seed (blank = random).
- **Generate button** turns into a progress bar ("Step 12 / 20") during the job. On `done` the new image
  appears in the result area and is prepended to the gallery.
- **Gallery:** responsive grid of thumbnails, lazy-loaded, with "Load more" paging (60 per page).
- **Lightbox:** full image, prompt, negative prompt, seed, size, steps, cfg and date. Actions:
  - **Reuse:** fills the composer with all params including the seed.
  - **Vary:** the same params with the seed cleared, generated immediately.
  - **Download** and **Delete** (Delete asks for an in-page confirmation, not `window.confirm`).
  - Reuse/Vary are disabled with a tooltip when `params` is null.
  - Esc closes the lightbox; arrow keys move to the previous/next image.
- **Status pill** (polls `/api/status` every 10s) and an offline banner. Errors show as toasts.

## Error handling
- ComfyUI down / timeout / workflow problems: the existing `ComfyUIError` messages travel as an SSE `error`
  event or a JSON error, and are shown as toasts.
- Unexpected exceptions in a job are logged and become a generic error event. The server keeps running.
- Corrupt or unreadable PNGs in the gallery are listed with `params: null`; thumbnail failure → placeholder.

## Testing
- `tests/test_gallery.py`:
  - `resolve_safe` rejects traversal, absolute and non-png names.
  - `read_params` on a fixture PNG built with an embedded `prompt` chunk, and on one without it.
  - Listing order; thumbnail caching.
- `tests/test_app.py`: FastAPI TestClient with a fake `generate`. Covers:
  - the generate → SSE progress → done flow
  - 409 when busy and 422 on bad params
  - the images list and delete
- Live check: start the app, run one real generation through the API, confirm it appears in `/api/images`,
  and fetch the page in a browser to check layout and interaction.

## Files
```
web/__init__.py, web/__main__.py (python -m web: start uvicorn + open browser)
web/app.py, web/jobs.py, web/gallery.py
web/static/index.html, styles.css, app.js
tests/test_gallery.py, tests/test_app.py
run_web.bat
```
New dependencies are `fastapi` and `pillow`, plus `pytest` and `httpx` (already installed) for dev.
README gets a "Web app" section.
