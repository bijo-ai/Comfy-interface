# LUMOS Edit studio (Phase 1): design

Date: 2026-10-09 · Status: agreed in chat (Bijo: "B", then "do as you think is best")

## Goal
A third space next to Create and Gallery: **Edit**, a focused editor for one image. It holds today's AI tools
(Fix, Restyle, Upscale) and adds instant, GPU-free tools (crop, rotate, flip, adjust, filters). Every step is a
new **version**; nothing is overwritten. Later phases add AI tools (Extend/outpaint, Remove object, ×4
upscaler, then background removal and face restore), also as Telegram commands.

## Decisions
| Topic | Decision |
|---|---|
| Navigation | Tabs **Create · Edit · Gallery** (`#edit`). "✏️ Open in Edit" in the image viewer. Fix/Edit buttons there open Edit with that tool selected. Dropping or pasting a photo onto Edit starts a session. |
| Versions | A session is a list of versions. The history strip shows them all, and clicking one makes it current. New steps start from the current version and are appended to the list. |
| Saving | AI steps already land in the Gallery, as today. Browser edits create **drafts** (uploads in `cache/sources/`) until **💾 Save**, which writes `LUMOS_edit_NNNNN_.png` into ComfyUI's output folder and copies the parent image's embedded settings, so prompts stay searchable. |
| Gallery kind | The `LUMOS_edit_` prefix counts as **Edited**. |
| Live preview | AI tools show the pixel preview on the Edit canvas: the existing live element is moved into Edit for the job. The result becomes the next version, and the Create stage is untouched. |
| Instant tools | Rendered in the browser with canvas: crop (Free, 1:1, 4:5, 3:2, 16:9, 9:16; drag to draw, drag to move), rotate ±90°, flip, adjust (brightness, contrast, saturation, warmth), filters (Vivid, Matte, Noir, Warm film, Cool, Fade). Live CSS preview, then Apply bakes a draft. |
| Compare | A before/after slider between the original (version 1) and the current version. |
| View | Fit / 100% / wheel zoom, and drag to pan when zoomed. |
| GPU rules | Unchanged: the existing builders (size fitting, ×2 upscale only ≤768 px, SDXL limits) apply to Edit's AI tools. Drafts upscale through the same checks. |
| Telegram | Phase 1 adds nothing; the existing Vary / Upscale / photo edit / `/fix` stay. Phase 2 tools get commands. |

## Server changes
- `POST /api/save {source_id, parent_name?}` returns the saved `GalleryImage`. The PNG is re-encoded, and the parent's `prompt` text chunk is copied when present. The name is the next free `LUMOS_edit_%05d_.png` in the output folder.
- `POST /api/upscale` accepts `source_id` as well as `name`. Drafts are planned with the same `build_upscale` checks, with settings taken from the optional `parent_name`.
- `image_kind`: `LUMOS_edit` maps to `edited`.

## Frontend structure
- `web/static/edit.js` (ES module) holds all Edit-space code. `initEdit(deps)` receives the shared helpers from `app.js` (api, toast, imageUrl, runJob, state access, refresh hooks), so there are no globals.
- `runJob(path, body, w, h, {liveHost, onDone})` gains options so jobs can preview and finish inside Edit.

## Testing
- **pytest:** save (name sequence, metadata copied, kind = edited, bad id, path safety) and upscale from a draft (accepted / refused by size).
- **Browser:** open from the Gallery, then:
  - adjust → apply → draft → save (count +1, kind edited, prompt kept)
  - crop 1:1 (square result) and rotate (dimensions swap)
  - Fix with the brush in Edit (new version, untouched pixels kept)
  - Restyle and Upscale (new versions)
  - history click goes back, and the compare slider moves
  - drop a photo to start a session
  - phone layout
