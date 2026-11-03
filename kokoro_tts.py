"""Kokoro (OpenAI-compatible) speech output for the Fluxer realtime voice loop.

Used when no xAI API key is configured: replies are voiced by the configured
Kokoro TTS service and streamed as raw mono PCM16 chunks into an async sink,
mirroring the xAI Realtime client's sink contract so the voice loop's turn
logic stays unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

import httpx

logger = logging.getLogger("fluxer_kokoro_tts")

# Kokoro emits mono PCM16 at 24 kHz; FLUXER_VOICE_SAMPLE_RATE must match.
# The endpoint and the voice carry no default: both are deployment choices
# (which service, which language), so FLUXER_VOICE_KOKORO_URL and
# FLUXER_VOICE_KOKORO_VOICE must name them.
DEFAULT_KOKORO_MODEL = "kokoro"
KOKORO_SAMPLE_RATE = 24_000
_READ_CHUNK_BYTES = 4096


def env_value(name: str) -> str:
    return (os.getenv(name) or "").strip()


def kokoro_base_url() -> str:
    url = env_value("FLUXER_VOICE_KOKORO_URL").rstrip("/")
    if not url:
        raise ValueError(
            "FLUXER_VOICE_KOKORO_URL is not set: name the OpenAI-compatible TTS endpoint"
        )
    return url


def kokoro_model() -> str:
    return env_value("FLUXER_VOICE_KOKORO_MODEL") or DEFAULT_KOKORO_MODEL


def kokoro_voice() -> str:
    voice = env_value("FLUXER_VOICE_KOKORO_VOICE")
    if not voice:
        raise ValueError(
            "FLUXER_VOICE_KOKORO_VOICE is not set: name a voice the TTS service offers"
        )
    return voice


def build_kokoro_speech_request(
    text: str,
    *,
    voice: str,
    model: str,
    base_url: str,
    api_key: str = "",
) -> urllib.request.Request:
    payload = json.dumps(
        {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": "pcm",
            "stream": True,
        }
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(
        f"{base_url.rstrip('/')}/audio/speech",
        data=payload,
        method="POST",
        headers=headers,
    )


@dataclass
class KokoroSpeechResult:
    """Fields the voice loop consumes from an xAI Realtime audio result."""

    transcript: str = ""
    bytes_written: int = 0
    events_seen: list[str] = field(default_factory=list)


class KokoroSpeechClient:
    """Stream Kokoro TTS PCM16 audio into an async sink (xAI-compatible shape)."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        sample_rate: int = 24_000,
    ) -> None:
        self.base_url = (base_url or kokoro_base_url()).rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else os.getenv("FLUXER_VOICE_KOKORO_API_KEY", "")
        ).strip()
        self.model = model or kokoro_model()
        self.voice = voice or kokoro_voice()
        if sample_rate != KOKORO_SAMPLE_RATE:
            raise ValueError(
                f"Kokoro streams mono PCM16 at {KOKORO_SAMPLE_RATE} Hz and cannot "
                f"resample; got sample_rate={sample_rate}. Set "
                "FLUXER_VOICE_SAMPLE_RATE=24000 (or use the xAI client) instead."
            )
        self.sample_rate = sample_rate

    async def text_response_to_sink(
        self,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
    ) -> KokoroSpeechResult:
        return await self._speak_to_sink(
            text,
            on_audio_delta,
            timeout=timeout,
            first_audio_timeout=first_audio_timeout,
        )

    async def force_message_to_sink(
        self,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
        interruptible: bool = False,
    ) -> KokoroSpeechResult:
        return await self._speak_to_sink(
            text,
            on_audio_delta,
            timeout=timeout,
            first_audio_timeout=first_audio_timeout,
        )

    async def _speak_to_sink(
        self,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float,
        first_audio_timeout: Optional[float],
    ) -> KokoroSpeechResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        request = build_kokoro_speech_request(
            text,
            voice=self.voice,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
        )
        result = KokoroSpeechResult(
            transcript=text, events_seen=["kokoro_speech_request"]
        )

        async def _stream() -> None:
            # Pull directly from the async transport: the sink supplies
            # backpressure, and cancellation closes the socket even while the
            # peer stalls in headers or a body read. No shared executor workers
            # or queue.put futures can survive barge-in.
            async with AsyncExitStack() as stack:
                client = await stack.enter_async_context(
                    httpx.AsyncClient(timeout=timeout)
                )

                async def _first_chunk():
                    response = await stack.enter_async_context(
                        client.stream(
                            "POST",
                            request.full_url,
                            content=request.data,
                            headers=dict(request.header_items()),
                        )
                    )
                    response.raise_for_status()
                    result.events_seen.append(f"http_{response.status_code}")
                    chunks = response.aiter_bytes()
                    try:
                        chunk = await chunks.__anext__()
                    except StopAsyncIteration as exc:
                        raise RuntimeError("Kokoro speech returned no audio") from exc
                    return chunks, chunk

                first_wait = min(timeout, first_audio_timeout or timeout)
                chunks, chunk = await asyncio.wait_for(
                    _first_chunk(), timeout=first_wait
                )
                result.events_seen.append("first_audio_chunk")
                while True:
                    # Bound each sink call even if HTTP coalesces many chunks.
                    for offset in range(0, len(chunk), _READ_CHUNK_BYTES):
                        delta = chunk[offset : offset + _READ_CHUNK_BYTES]
                        result.bytes_written += len(delta)
                        await on_audio_delta(delta)
                    try:
                        chunk = await chunks.__anext__()
                    except StopAsyncIteration:
                        break

        try:
            # Includes connection setup, headers, reads AND sink backpressure.
            await asyncio.wait_for(_stream(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise RuntimeError(
                self._timeout_message(result, timeout, first_audio_timeout)
            ) from exc
        except httpx.HTTPError as exc:
            # URLs / server detail can contain credentials; report the class only.
            raise RuntimeError(
                f"Kokoro speech request failed ({type(exc).__name__})"
            ) from exc
        result.events_seen.append("kokoro_speech_complete")
        return result

    @staticmethod
    def _timeout_message(
        result: KokoroSpeechResult,
        timeout: float,
        first_audio_timeout: Optional[float],
    ) -> str:
        if not result.bytes_written:
            return f"Kokoro speech produced no audio within {first_audio_timeout or timeout}s"
        # Partial audio is a failed reply, not a short one: the voice loop must
        # not report a truncated turn as complete.
        return f"Kokoro speech did not finish within {timeout}s"
