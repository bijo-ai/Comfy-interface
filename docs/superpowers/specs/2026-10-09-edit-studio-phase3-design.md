# LUMOS Edit studio (Phase 3): Background and Faces

Date: 2026-10-09 · Status: "phase 3 green flag"; details decided by Claude and tested on the RTX 4050

## Tools
| Tool | How it works | Notes |
|---|---|---|
| 🌄 Background | ComfyUI's core `RemoveBackground` with BiRefNet (Comfy-Org, 444 MB) gives the subject mask. Choices: transparent (`InvertMask` → `JoinImageWithAlpha`, because that node inverts its alpha), white/black (`EmptyImage` + composite), blur (scaled to 384 px, `ImageBlur` radius 24 / sigma 10 (its maximum), scaled back), or a new scene (inpainting model paints the inverted, grown mask at ≤ 768 px, then the original subject is composited on top at full size). | 2–4 s, or ~11 s for a new scene. The subject always keeps its pixels. |
| 😊 Faces | Pass 1: MediaPipe face landmarker (core node, 5 MB model) → a face-oval mask fetched as a temporary image → blobs → square crops with 60 % context. Pass 2: for each face (≤ 6), crop → 512 px → img2img with `SetLatentNoiseMask` at denoise 0.35 (DreamShaper 8) → scale back → feathered paste. | ~15 s for two faces. The picture's own prompt is added (without it, an old man came out younger); tool prompts such as "empty background…" are ignored. |

## Surfaces
- **Studio Edit:** Background (five choices; a description box appears for "New scene") and Faces tools. See-through images sit on a checkerboard.
- **API:** `POST /api/background {mode, prompt, source_*}` and `POST /api/faces {source_*, parent_name}`. `/api/models` reports `background` and `faces`.
- **Telegram:** photo captions `/faces`, `/nobg` (sent as a PNG document so the transparency survives) and `/bg white|black|blur|<scene>`. Every result gets 😊 Faces and 🌄 Background buttons (Background asks Nothing/White/Black/Blur).
- **Transparency is kept** by uploads, Save (`LUMOS_edit_`) and gallery thumbnails (WEBP with alpha).
- **Gallery kinds:** `ComfyUI_background` → edited, `ComfyUI_faces` → fixed.
