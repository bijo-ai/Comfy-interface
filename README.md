# comfyui-mcp

A local MCP server that lets an LLM in **LM Studio** generate images with your local **ComfyUI**.
It exposes one tool:

```
generate_image(prompt, negative_prompt="blurry, low quality, ...", width=512, height=512,
               steps=20, cfg=8.0, seed=None)
```

The tool returns the PNG as MCP image content plus a short text summary (seed, size, steps, saved path).
Every image is also saved to `outputs/`, alongside a `.json` sidecar that records the parameters used.

```
LM Studio ──MCP (stdio or HTTP)──> server.py ──> comfy_client.py ──HTTP/WebSocket──> ComfyUI :8188
```

| File | Role |
|---|---|
| `server.py` | MCP layer: tool definition, transports, error → tool-error mapping |
| `comfy_client.py` | ComfyUI logic: load + patch workflow, queue, wait, fetch, save. No MCP imports |
| `config.py` | Reads settings from `.env` |
| `workflow_api.json` | The ComfyUI graph, in **API format** |
| `test_generate.py` | End-to-end test of the generation logic (no MCP) |
| `check_mcp.py` | Minimal MCP client: lists tools and optionally calls `generate_image` |
| `web/` | **ComfyUI Studio** web app: `app.py` (FastAPI routes), `jobs.py` (live progress), `gallery.py` (output folder), `static/` (the page) |
| `start_studio.bat` | One-click launcher: starts ComfyUI if needed, then the web app |
| `run_web.bat` | Starts only the web app |
| `assets/studio.ico` | App icon for a Desktop shortcut |
| `web/bot_core.py`, `web/bot.py` | Telegram bot: behaviour (commands, access lock, Vary) and python-telegram-bot wiring |
| `tests/` | pytest suite for the client, gallery, jobs, web API and bot |

## Web app: ComfyUI Studio

A local website for generating and browsing images without an LLM. Type a prompt and press **Generate**
(or Ctrl+Enter). A live progress bar shows each sampling step, then the image appears.

- **Shapes:** Square 512×512, Portrait 512×768, Landscape 768×512. **Advanced** has the negative prompt,
  width/height, steps, CFG and seed.
- **Gallery:** shows every PNG in ComfyUI's output folder, newest first. That includes images you made
  in ComfyUI itself, along with the prompt and settings ComfyUI stored in each file.
- **Full-size view:** click an image to open it. From there:
  - **Reuse** loads its settings into the form.
  - **Vary** makes the same image again with a new seed.
  - **Download** saves it.
  - **Delete** removes it from disk, after a confirmation.
  - ←/→ move between images and Esc closes.
- **Status:** a ComfyUI connected/offline indicator. The page reconnects by itself when ComfyUI starts.

**Start it:** double-click **`start_studio.bat`**. It starts Comfy Desktop if it isn't running, waits until
ComfyUI is ready, then starts the site and opens your browser. If the site is already running, it just
opens the page. For a Desktop icon, right-click `start_studio.bat` → *Send to* → *Desktop (create shortcut)*,
then use `assets/studio.ico` as its icon. If Comfy Desktop is installed somewhere else, set the
`COMFY_DESKTOP_EXE` environment variable to its `.exe`.

`run_web.bat` starts only the site, if ComfyUI is already running. Or run it yourself:

```powershell
uv run python -m web                 # opens http://127.0.0.1:7860 in your browser
uv run python -m web --port 8080 --no-browser
```

It only listens on `127.0.0.1`, so other devices can't reach it. ComfyUI's output folder is detected from
ComfyUI's `--output-directory` launch flag. If that fails (e.g. a portable ComfyUI install), set
`COMFYUI_OUTPUT_DIR` in `.env`. Thumbnails are cached in `cache/`, which is safe to delete.
The web app doesn't write copies to `outputs/`; that folder is only used by the MCP server.

**Tests:** `uv run pytest`

## Telegram bot

Generate images from your phone, from anywhere. The bot runs inside ComfyUI Studio, so `start_studio.bat`
starts it too. It only makes outgoing connections to Telegram, so nothing on your PC is exposed to the
internet. Images land in ComfyUI's output folder and show up in the Studio gallery.

**Setup (once):**
1. In Telegram, message **@BotFather**, send `/newbot`, and pick a name and a username ending in `bot`.
2. Put the token it gives you in `.env` as `TELEGRAM_BOT_TOKEN=...`. Keep it secret: anyone with it controls the bot.
3. Start Studio and send your bot any message. It replies with your numeric user ID.
4. Add `TELEGRAM_ALLOWED_USER_ID=<that number>` to `.env` and restart Studio. From then on the bot answers
   only you, and messages from anyone else are ignored.

**Use:**

| Send | Get |
|---|---|
| `a beach at sunset` | square 512×512 |
| `/portrait an old fisherman` | portrait 512×768 |
| `/landscape mountains at dawn` | landscape 768×512 |
| tap **🔁 Vary** under an image | same prompt and size, new seed |
| `/help` | these instructions |

The bot shows live progress ("🎨 Generating… step 12/20") and captions each image with its seed. It and the
website share one generator: if one is busy, the other says so and asks you to try again in a moment. The PC
must be on, online, and running Studio. If it starts offline, the bot keeps retrying every 30 seconds.
The Studio window shows `Telegram bot: on/off` at startup.

## Setup (Windows)

Requires [uv](https://docs.astral.sh/uv/) and ComfyUI running at `http://127.0.0.1:8188`
with the `v1-5-pruned-emaonly.safetensors` checkpoint installed.

```powershell
cd C:\path\to\Comfy-interface
uv sync                      # creates .venv with Python 3.12 + dependencies
copy .env.example .env       # then edit if needed
uv run python test_generate.py
```

> Antivirus HTTPS scanning (e.g. AVG/Avast) can break package downloads. If `uv sync` fails with a
> certificate error, run `$env:UV_NATIVE_TLS=1; uv sync` so uv uses the Windows certificate store.

### `.env`

| Key | Default | Meaning |
|---|---|---|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI base URL |
| `WORKFLOW_PATH` | `workflow_api.json` | API-format workflow (relative paths resolve from this folder) |
| `OUTPUT_DIR` | `outputs` | Where PNGs and JSON sidecars are written |
| `TIMEOUT` | `180` | Seconds to wait for a generation before giving up |
| `COMFYUI_OUTPUT_DIR` | *(auto-detected)* | ComfyUI's output folder, shown in the web gallery |
| `CACHE_DIR` | `cache` | Where the web app keeps gallery thumbnails |
| `TELEGRAM_BOT_TOKEN` | *(empty: bot off)* | Bot token from @BotFather |
| `TELEGRAM_ALLOWED_USER_ID` | *(empty)* | The only Telegram user the bot serves |

Real environment variables take precedence over `.env`.

## Exporting a workflow in API format

The server needs the **API** format. That is a flat `{"node_id": {"class_type": ..., "inputs": ...}}` map,
not the regular UI save, which contains `"nodes"` and `"links"`.

1. Open your workflow in ComfyUI and check it runs with **Run**.
2. Open the **Workflow** menu (top-left ComfyUI logo, then **File**) and choose **Export (API)**.
   - On older frontends, first enable **Settings → Comfy → Dev mode** ("Enable dev mode options").
     Then use the **Save (API Format)** button.
3. Save the file as `workflow_api.json` in this folder, or point `WORKFLOW_PATH` at it.

Nodes are located by `class_type`, not by ID, so any export works as long as it has:
exactly one `KSampler`, whose `positive` and `negative` inputs connect directly to `CLIPTextEncode` nodes;
one `EmptyLatentImage`; and one `SaveImage`. The positive and negative encoders are found by following
the KSampler's links, so node order and titles don't matter. Sampler, scheduler and checkpoint come
from the workflow file itself.

## Running the server

```powershell
# stdio (LM Studio launches it for you; you normally don't run this by hand)
uv run python server.py

# streamable HTTP, served at http://127.0.0.1:8000/mcp
uv run python server.py --transport http --port 8000
```

`--host` defaults to `127.0.0.1`. Keep it that way unless you mean to expose the server on your network.

### Verifying it

```powershell
uv run python check_mcp.py stdio                  # spawn over stdio, list tools
uv run python check_mcp.py stdio --bad --call     # also test validation + a real generation
uv run python check_mcp.py http http://127.0.0.1:8000/mcp --call   # against a running HTTP server
```

You can also use the MCP Inspector (needs Node.js):
`npx @modelcontextprotocol/inspector C:\path\to\Comfy-interface\.venv\Scripts\python.exe C:\path\to\Comfy-interface\server.py`

## LM Studio configuration

In LM Studio, open **Program** (right sidebar) → **Install** → **Edit mcp.json**, then add one of these.

### stdio (recommended)

LM Studio starts and stops the server itself. The venv's Python is called directly, so this works
even when `uv` isn't on LM Studio's PATH.

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "C:\\path\\to\\Comfy-interface\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\Comfy-interface\\server.py"]
    }
  }
}
```

Settings come from `.env`. To override one here, add an `"env"` block, e.g.
`"env": { "TIMEOUT": "300" }`.

### Streamable HTTP

First start the server yourself (`uv run python server.py --transport http --port 8000`) and leave it running:

```json
{
  "mcpServers": {
    "comfyui": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

## Model requirements

**The LM Studio model must support tool calling.** In LM Studio's model list these models show a
hammer/"Tool use" badge. Models that work well:

- Qwen3 (4B / 8B / 14B), or Qwen2.5 7B/14B Instruct
- Llama 3.1 8B Instruct / Llama 3.2 3B Instruct
- Mistral Nemo Instruct, Ministral 8B, Mistral Small
- gpt-oss-20b

Without tool-calling support the model just writes text and never calls `generate_image`.
Non-vision models can't see the returned image, but they still get the text summary with the seed and saved path.

ComfyUI and the LLM share your GPU (6 GB on an RTX 4050 laptop). A 4B–8B quantised model
(Q4) leaves room for SD 1.5. If generations time out or fail with out-of-memory errors, use a smaller
model, or offload some of its layers to the CPU in LM Studio.

## Errors returned to the LLM

The server never crashes on a failed generation. The tool returns an error result with a readable message instead:

| Situation | Message |
|---|---|
| ComfyUI not running | `Cannot connect to ComfyUI at http://127.0.0.1:8188. Is ComfyUI running?` |
| Bad arguments | `Invalid parameters: width must be a multiple of 8 (got 500); ...` |
| UI-format / missing / broken workflow | `... is not in ComfyUI API format. In ComfyUI use Workflow > Export (API) ...` |
| ComfyUI rejects graph (e.g. missing model) | `ComfyUI rejected the workflow: ... ckpt_name: 'x' not in [...]` |
| Too slow | `Image generation did not finish within 180s. ...` |

Validation limits: width/height 64–2048 and multiples of 8, steps 1–150, cfg 1.0–30.0, seed 0–4294967295.

## How a call works

1. `workflow_api.json` is read fresh, then patched on a deep copy; the file on disk is never changed.
2. The server opens a WebSocket to `/ws?clientId=...` *before* queueing, so no progress events are missed.
3. It POSTs `{"prompt": workflow, "client_id": ...}` to `/prompt`.
4. It waits for `execution_success`, or for `executing` with `node: null`. If the WebSocket can't connect
   or drops, it polls `/history/{prompt_id}` once per second until the timeout.
5. It reads the SaveImage output from history, downloads it through `/view`, saves it to `outputs/`
   with a sidecar, and returns it.
