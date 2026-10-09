"""Which model profiles and upscaling models ComfyUI can actually run right now (cached briefly)."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

import httpx

from comfy_client import ComfyClient, ComfyUIError
from models import ModelProfile, available_profiles

log = logging.getLogger(__name__)

CACHE_SECONDS = 30.0


class ModelCatalog:
    def __init__(
        self,
        base_url: str,
        fetch: Callable[[], Awaitable[list[str]]] | None = None,
        fetch_upscalers: Callable[[], Awaitable[list[str]]] | None = None,
    ) -> None:
        self._base_url = base_url
        self._fetch = fetch or self._fetch_checkpoints
        self._fetch_upscalers = fetch_upscalers or self._fetch_upscale_models
        self._cached: list[ModelProfile] = []
        self._fetched_at = float("-inf")
        self._upscalers: list[str] = []
        self._upscalers_at = float("-inf")

    async def available(self) -> list[ModelProfile]:
        if time.monotonic() - self._fetched_at < CACHE_SECONDS:
            return self._cached
        try:
            names = await self._fetch()
        except (ComfyUIError, httpx.HTTPError, KeyError, ValueError) as exc:
            log.debug("Could not list checkpoints: %s", exc)
            return []  # not cached: ask again as soon as ComfyUI is back
        self._cached = available_profiles(names)
        self._fetched_at = time.monotonic()
        return self._cached

    async def upscalers(self) -> list[str]:
        """Upscaling model files (ComfyUI's models/upscale_models)."""
        if time.monotonic() - self._upscalers_at < CACHE_SECONDS:
            return self._upscalers
        try:
            self._upscalers = await self._fetch_upscalers()
        except (ComfyUIError, httpx.HTTPError, KeyError, ValueError) as exc:
            log.debug("Could not list upscaling models: %s", exc)
            return []
        self._upscalers_at = time.monotonic()
        return self._upscalers

    async def _fetch_upscale_models(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5) as http:
            return await ComfyClient(self._base_url, http).list_upscale_models()

    async def _fetch_checkpoints(self) -> list[str]:
        async with httpx.AsyncClient(timeout=5) as http:
            return await ComfyClient(self._base_url, http).list_checkpoints()
