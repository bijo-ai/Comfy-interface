"""MCP server exposing ComfyUI image generation as a `generate_image` tool.

    python server.py                         # stdio (for LM Studio command-based config)
    python server.py --transport http        # streamable HTTP on http://127.0.0.1:8000/mcp
"""

from __future__ import annotations

import argparse
import base64
import logging
import sys
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ImageContent, TextContent
from pydantic import Field

from comfy_client import DEFAULT_NEGATIVE, ComfyUIError, GenerationParams, GenerationResult, generate
from config import load_settings

# stdout carries the stdio protocol, so all logging goes to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("comfyui-mcp")

mcp = MCPServer(
    "comfyui",
    instructions="Generates images locally with ComfyUI (Stable Diffusion 1.5). "
    "Write prompts as comma-separated visual descriptions.",
)


def summarize(result: GenerationResult) -> str:
    p = result.params
    return (
        f"Generated {p.width}x{p.height} image in {result.elapsed:.1f}s "
        f"(seed {p.seed}, steps {p.steps}, cfg {p.cfg}). Saved to {result.saved_path}"
    )


@mcp.tool(structured_output=False)
async def generate_image(
    prompt: Annotated[str, Field(description='What to draw, e.g. "a red fox in a snowy forest, photo, detailed".')],
    negative_prompt: Annotated[str, Field(description="Things to avoid in the image.")] = DEFAULT_NEGATIVE,
    width: Annotated[int, Field(description="Width in pixels, multiple of 8 (64-2048). SD 1.5 works best at 512.")] = 512,
    height: Annotated[int, Field(description="Height in pixels, multiple of 8 (64-2048). SD 1.5 works best at 512.")] = 512,
    steps: Annotated[int, Field(description="Sampling steps (1-150). 20-30 is typical.")] = 20,
    cfg: Annotated[float, Field(description="Prompt adherence (1.0-30.0). 7-8 is typical.")] = 8.0,
    seed: Annotated[int | None, Field(description="Seed for reproducible results; omit for random.")] = None,
) -> list[TextContent | ImageContent]:
    """Generate an image from a text prompt using the local ComfyUI (Stable Diffusion 1.5).

    Returns the image plus a short summary with the seed and where it was saved.
    """
    params = GenerationParams(prompt, negative_prompt, width, height, steps, cfg, seed)
    try:
        result = await generate(params, load_settings())
    except ComfyUIError as exc:
        log.warning("generate_image failed: %s", exc)
        raise ToolError(str(exc)) from exc
    except Exception as exc:  # keep the server alive on anything unexpected
        log.exception("Unexpected error in generate_image")
        raise ToolError(f"Unexpected error while generating the image: {exc}") from exc

    log.info(summarize(result))
    return [
        TextContent(type="text", text=summarize(result)),
        ImageContent(type="image", data=base64.b64encode(result.image).decode(), mime_type="image/png"),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ComfyUI MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.transport == "http":
        log.info("Serving streamable HTTP on http://%s:%d/mcp", args.host, args.port)
        mcp.run("streamable-http", host=args.host, port=args.port)
    else:
        mcp.run("stdio")


if __name__ == "__main__":
    main()
