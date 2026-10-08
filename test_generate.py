"""End-to-end check of the ComfyUI generation logic (no MCP / LM Studio involved).

Run with:  uv run python test_generate.py
"""

from __future__ import annotations

import asyncio
import json
import sys

from comfy_client import (
    GenerationParams,
    InvalidParamsError,
    apply_params,
    generate,
    identify_nodes,
    load_workflow,
)
from config import load_settings


def check_node_detection() -> None:
    settings = load_settings()
    template = load_workflow(settings.workflow_path)
    original = json.dumps(template, sort_keys=True)
    roles = identify_nodes(template)
    print(f"[ok] node roles: {roles}")

    params = GenerationParams(prompt="test", width=640, height=384, steps=5, cfg=6.5, seed=42)
    patched, _ = apply_params(template, params)
    assert patched[roles.positive]["inputs"]["text"] == "test"
    assert patched[roles.latent]["inputs"]["width"] == 640
    assert patched[roles.sampler]["inputs"]["seed"] == 42
    assert json.dumps(template, sort_keys=True) == original, "template was mutated"
    print("[ok] params applied to a copy; template untouched")


def check_validation() -> None:
    bad = GenerationParams(prompt=" ", width=513, height=4096, steps=0, cfg=99)
    try:
        bad.validate()
    except InvalidParamsError as exc:
        print(f"[ok] validation rejects bad input: {exc}")
    else:
        raise AssertionError("validation should have failed")


async def check_generation() -> None:
    settings = load_settings()
    print(f"[..] generating via {settings.comfyui_url} (timeout {settings.timeout:.0f}s)")
    result = await generate(
        GenerationParams(prompt="a red fox sitting in a snowy forest, photo", steps=20),
        settings,
    )
    assert result.image.startswith(b"\x89PNG"), "output is not a PNG"
    assert result.saved_path.exists() and result.saved_path.with_suffix(".json").exists()
    print(
        f"[ok] generated {len(result.image) // 1024} KB PNG in {result.elapsed:.1f}s, "
        f"seed {result.params.seed}, saved to {result.saved_path}"
    )


def main() -> int:
    check_node_detection()
    check_validation()
    asyncio.run(check_generation())
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
