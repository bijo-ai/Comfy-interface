# Quality pack — design

Date: 2026-10-08 · Status: approved in chat ("yes")

Five features, shared by the Studio website and the Telegram bot. The hard constraint is that **nothing may overload the
6 GB laptop GPU**.

## 1. Models
`models.py` holds a fixed table of `ModelProfile`s. Only profiles whose checkpoint exists in ComfyUI are offered.

| key | label | checkpoint | family | shapes (sq / portrait / landscape) | max batch | upscale | steps / cfg | heavy |
|---|---|---|---|---|---|---|---|---|
| `dreamshaper` (default) | DreamShaper 8 | DreamShaper_8_pruned.safetensors | sd15 | 512² / 512×768 / 768×512 | 4 | yes | 25 / 7 | no |
| `sd15` | SD 1.5 base | v1-5-pruned-emaonly.safetensors | sd15 | same | 4 | yes | 20 / 8 | no |
| `sdxl` | SDXL (heavy, slow) | sd_xl_base_1.0.safetensors | sdxl | 1024² / 832×1216 / 1216×832 | 1 | no | 25 / 7 | yes |

- **Default fallback:** if DreamShaper is missing, the default falls back to the first available profile.
- **Per-request model:** `GenerationParams` gains `model: str | None` (the checkpoint filename) and `batch: int = 1`. `apply_params` sets `CheckpointLoaderSimple.ckpt_name` and `EmptyLatentImage.batch_size`.
- **Limits** (enforced by the web API and the bot):
  - `batch <= profile.max_batch`
  - `batch > 1` only when width and height are ≤ 768
  - the model must be available
- **Gallery Reuse:** `params_from_graph` also returns `model`, so Reuse restores the model too.

## 6. Sharper previews
Request `preview_method: "auto"` instead of `latent2rgb`. ComfyUI then uses TAESD when `vae_approx/taesd_decoder*` /
`taesdxl_decoder*` exist (downloaded), and otherwise falls back to latent2rgb by itself.

## 3. Styles
`styles.py` holds a fixed table: `none, photo, cinematic, oil, anime, render3d, watercolor`.
- **Each style:** a label, an emoji, positive keywords appended to the prompt (`prompt, keywords`), and negative keywords appended to the negative prompt.
- **Applying it:** `apply_style(prompt, negative, key) -> (prompt, negative)` runs before generation, so the gallery shows the full prompt.
- **Where to pick it:** the web composer has style chips. The bot has `/style` (an inline keyboard), and the choice sticks until changed.

## 4. Make 4
- **Generation:** batch 4 renders through `EmptyLatentImage.batch_size`. `generate()` reports all output images (`comfy_files`). Web jobs return `images: [GalleryImage…]`, with `image` kept as the first one for compatibility.
- **Web:** the composer has a ×1 / ×4 toggle (×4 disabled for SDXL or sizes over 768), and the stage shows a 2×2 grid. Clicking a tile opens it in the lightbox.
- **Bot:** a **🖼️ ×4** button under each photo makes 4 new variations, sent as an album. The album is followed by a message with buttons [🔍 1] [🔍 2] [🔍 3] [🔍 4] [🖼️ ×4 again].

## 2. Upscale ×2
- **Workflow:** built in code, not from the workflow file:
  - `CheckpointLoaderSimple`
  - `LoadImage` (the source PNG is uploaded with `/upload/image`, `type=input`, `overwrite=true`, under a fixed name `studio_upscale_src.png`)
  - `ImageScaleBy` (lanczos ×2)
  - `VAEEncodeTiled` (512 / 64)
  - two `CLIPTextEncode`
  - `KSampler` (source seed, 15 steps, the profile's cfg, `dpmpp_2m` / `karras`, denoise 0.4)
  - `VAEDecodeTiled` (512 / 64)
  - `SaveImage` (prefix `ComfyUI_upscaled`)
- **Prompt:** comes from the source image's params. If it has none, use "high quality, detailed" with the default negative, the source model if known, else the default SD 1.5 profile.
- **Allowed only when** the source is ≤ 768 on both sides and its model is not SDXL. Otherwise the API returns 422 and the UI disables the button with an explanation.
- **Web:** **🔍 Upscale ×2** in the lightbox. The result shows on the stage and at the start of the gallery.
- **Bot:** a **🔍 Upscale** button under each photo. The result is sent as a **document** (full resolution; Telegram compresses photos).

## Shared plumbing
- **One queue:** jobs keep going through the single `JobManager`, so the web and the bot share one GPU queue. A job is a `GenerationParams` or an `UpscaleRequest(source: Path, params: GenerationParams)`. The web runner dispatches on the type.
- **New web API:** `GET /api/models` returns profiles, availability, the default and the styles. `POST /api/generate` accepts `model`, `style` and `batch`. `POST /api/upscale` takes `{name}`.
- **Bot command:** `/model` (an inline keyboard), plus a footer line with the current model and style in `/help`.

## Errors
- **Limit violations:** a clear 422 message, or a bot reply.
- **Missing model** (deleted after startup): ComfyUI's validation error is passed through as it is today.
- **Upload failure:** the ComfyUI error is passed through.

## Testing
- **Unit tests:** profiles and availability, the limits, styles, `apply_params` with model/batch, `params_from_graph` returning the model, the upscale workflow shape, the upload call (MockTransport), and multi-image results.
- **API tests:** `/api/models`, generate with model/style/batch (including violations), and upscale (allowed / refused).
- **Bot tests:** `/model` and `/style` choices persisting, ×4 album plus picker, upscale sent as a document, and SDXL refusing ×4/upscale.
- **Live:** DreamShaper single and ×4, an SDXL single, an upscale, a style, browser screenshots, and a Telegram check by Bijo.
