# LUMOS Edit studio (Phase 2): Extend, Remove object, sharp ×4

Date: 2026-10-09 · Status: agreed in chat ("yes start"), details decided by Claude and tested on the RTX 4050

## Tools
| Tool | How it works | GPU / size rules |
|---|---|---|
| ↔️ Extend | Sides (left/top/right/bottom) × amount (15/25/40 % of the picture). The canvas starts as a blurred stretch of the picture (from flat grey the model may leave it grey), then `InpaintModelConditioning` + DreamShaper 8 inpainting at denoise 1.0 paints the new area plus a 16 px overlap strip. A neutral "scene continuing" prompt is the default (the original prompt duplicated its subjects). | AI canvas ≤ 1024 px. The result is scaled up and the original laid on top at its own resolution (capped at 2048 px). |
| 🧽 Remove | The brush mask is grown 24 px; the hole is pre-filled from its surroundings (multi-scale normalised box blur, numpy), then repainted at denoise 0.9 with an "empty background" prompt. Grow 12 / denoise 0.75 left ghost objects on a large subject (fox test). | Works at ≤ 768 px; the patch is pasted back onto the full-size original (Fix does the same now). |
| 🔎 ×4 Sharp | `UpscaleModelLoader` + `ImageUpscaleWithModel` with RealESRGAN x4plus (Comfy-Org safetensors, 66.9 MB). No diffusion, so any model's images work. | Sources ≤ 1024 px (→ 4096). ~10 s. |

## Surfaces
- **Studio Edit:** Remove and Extend tools (the brush block is shared with Fix), and Upscale gets ×2 Detail / ×4 Sharp.
- **API:** `POST /api/extend {sides, amount, prompt, source_id|source_name}`, `POST /api/remove {mask_id, source_*}`,
  and `POST /api/upscale` with `scale: 2|4`. `/api/models` reports `sharp`.
- **Telegram:** photo captions `/remove` (pink scribble → preview → ✅) and `/extend [prompt]` (Wider/Taller/All
  sides). Every result also gets ↔️ Extend and 🔎 Sharp ×4 buttons, and 🔁 Vary repeats an Extend or Remove.
- **Gallery kinds:** `ComfyUI_outpaint` → edited, `ComfyUI_remove` → fixed, `ComfyUI_upscaled_x4` → upscaled. The
  gallery reads prompts through `InpaintModelConditioning`.

## Next (Phase 3)
Background removal and face restore.
