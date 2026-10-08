# Quality Pack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Model picker (DreamShaper default, SDXL heavy-limited), sharper TAESD previews, style presets, ×4 batches, and Upscale ×2, all in both Studio and the Telegram bot, without overloading a 6 GB GPU.

**Spec:** `docs/superpowers/specs/2026-10-08-quality-pack-design.md`

**Note on detail:** each task lists files, interfaces and named tests. The code is written test-first during execution,
not pre-written here.

## Global Constraints
- **GPU limits:** batch ≤ profile.max_batch; batch > 1 only at ≤ 768×768; upscale only for SD 1.5-family sources ≤ 768; tiled VAE for upscale.
- **Single queue:** one generation at a time across web and bot (the existing `JobManager`).
- **Compatibility:** the MCP server's `generate_image` keeps working unchanged (its workflow model, batch 1). All existing tests stay green.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus
1. **SDXL chosen together with ×4 or upscale**, from either the web or the bot: refused with a clear reason, and no GPU job starts.
2. **A gallery image from an unknown or other workflow:** upscale still works with a fallback prompt and model, or is cleanly refused if it's too big.
3. **A ×4 job with one image missing on disk:** the payload lists the images that exist, and nothing crashes.
4. **A model file deleted while Studio runs:** a clear error, and the picker refreshes.
5. **Telegram album and buttons:** the 🔍 n buttons map to the right image after a restart (an expired token gives "expired").

### Task 1: Models, previews, batch in the client
**Files:** create `models.py` and `tests/test_models.py`. Modify `comfy_client.py`, `web/gallery.py`, `tests/test_comfy_client.py` and `tests/test_gallery.py`.
- **`models.py`:**
  - `ModelProfile` (key, label, ckpt, family, shapes: dict[str, tuple[int,int]], max_batch, upscale, steps, cfg, heavy)
  - `PROFILES`, `DEFAULT_KEY = "dreamshaper"`
  - `available_profiles(ckpt_names) -> list[ModelProfile]`, `default_profile(available)`, `profile_for_ckpt(ckpt) -> ModelProfile | None`
  - `check_limits(profile, width, height, batch) -> None`, which raises `InvalidParamsError`
- **`comfy_client.py`:**
  - `GenerationParams.model: str | None = None` and `batch: int = 1` (validated 1..4)
  - `apply_params` sets ckpt and batch_size
  - `PREVIEW_METHOD = "auto"`
  - `GenerationResult.comfy_files: list[dict]` (all outputs; `comfy_file` = first)
  - `async list_checkpoints(base_url) -> list[str]` (from `/object_info/CheckpointLoaderSimple`)
- **`web/gallery.py`:** `params_from_graph` adds `"model"` (the CheckpointLoaderSimple ckpt_name if it's a string; otherwise the key is absent).
- **Tests:** `test_profiles_available_and_default`, `test_check_limits`, `test_apply_params_sets_model_and_batch`, `test_result_lists_all_outputs` (history with 4 images), `test_list_checkpoints` (MockTransport), `test_queue_prompt_requests_auto_previews`, `test_read_params_includes_model`.

### Task 2: Styles
**Files:** create `styles.py` and `tests/test_styles.py`.
- **`styles.py`:** `Style(key, label, emoji, positive, negative)`, `STYLES`, and `apply_style(prompt, negative, key) -> tuple[str, str]`. An unknown key raises `InvalidParamsError`; `none` is identity.
- **Tests:** `test_apply_style_appends_keywords`, `test_none_style_is_identity`, `test_unknown_style_rejected`.

### Task 3: Upscale in the client
**Files:** modify `comfy_client.py` and `tests/test_comfy_client.py`.
- **`build_upscale_workflow(ckpt, image_name, params, seed, cfg) -> Workflow`:** the node graph from the spec.
- **`ComfyClient.upload_image(path, name) -> str`:** multipart POST to `/upload/image`, returning the stored name.
- **`async upscale(source: Path, params: GenerationParams, settings, on_progress, on_preview) -> GenerationResult`:** refactor the shared queue/wait/fetch into `_run(workflow, save_node, …)`.
- **Tests:** `test_upscale_workflow_shape` (node classes, denoise 0.4, scale 2, tiled), `test_upload_image_multipart`.

### Task 4: Web API
**Files:** modify `web/jobs.py`, `web/app.py`, `tests/test_app.py` and `tests/test_jobs.py`.
- **`UpscaleRequest` dataclass:** jobs accept `GenerationParams | UpscaleRequest`.
- **Runner dispatch:** `make_runner` dispatches on the job type. The payload has `images` (list) and `image` (first).
- **`GET /api/models`:** `{models: [{key,label,family,shapes,max_batch,upscale,heavy,available,steps,cfg}], default, styles: [{key,label,emoji}]}`. Availability comes from `list_checkpoints`, cached for 30 s.
- **`POST /api/generate`:** gains `model` (a key), `style`, `batch`. It resolves the profile, checks limits, applies the style, and sets `params.model`.
- **`POST /api/upscale` `{name}`:** checks the source is ≤ 768 and its model profile allows upscale; uses the source params or the fallback; returns `job_id`.
- **Tests:** `test_models_endpoint`, `test_generate_with_model_style_batch`, `test_generate_rejects_sdxl_batch`, `test_upscale_starts_job`, `test_upscale_refuses_large_or_sdxl`, `test_batch_payload_lists_existing_images`.

### Task 5: Web UI
**Files:** `web/static/index.html`, `styles.css`, `app.js`.
- **Composer:**
  - a model `<select>` with the heavy label
  - shapes relabelled per model
  - style chips
  - a ×1 / ×4 segmented control, disabled with a reason when not allowed
- **Results:** batch results show as a 2×2 stage grid. The lightbox gets **🔍 Upscale ×2**, disabled with a tooltip when not allowed. Reuse restores the model.
- **Verify:** a Playwright run on a test port covering DreamShaper ×1 and ×4, a style, an upscale, and SDXL ×4 disabled. Screenshots, plus the old UI checks re-run.

### Task 6: Telegram
**Files:** `web/bot_core.py`, `web/bot.py`, `tests/test_bot_core.py`.
- **`StudioBot`:**
  - per-bot `model_key` and `style_key`
  - `/model` and `/style` reply with choice keyboards
  - `handle_choice(user, kind, key, chat)`
  - generation uses the model and style
  - photo buttons: 🔁 Vary · 🖼️ ×4 · 🔍 Upscale
  - ×4 sends an album, then a picker message
  - upscale sends a document
  - the limits produce replies
- **`Chat` protocol** gains `send_album(paths, caption)`, `send_document(path, caption)` and `send_choices(text, buttons)`. A button is `(label, data)` rows, and the photo buttons come from a list.
- **Tests:** `test_model_and_style_choice_persist`, `test_x4_sends_album_and_picker`, `test_upscale_sends_document`, `test_sdxl_refuses_x4_and_upscale`, `test_expired_tokens`.

### Task 7: Docs, live checks, review, release
- **README:** models, styles, ×4, upscale, GPU notes.
- **Live:** the full browser check, and Bijo checks Telegram.
- **Release:** a final review subagent, a fix pass, merge, push, then restart Studio.
