"""Kokoro (OpenAI-compatible) speech output for the Fluxer realtime voice loop.

Used when no xAI API key is configured: replies are voiced by the cluster's
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
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("fluxer_kokoro_tts")

# Kokoro emits mono PCM16 at 24 kHz; FLUXER_VOICE_SAMPLE_RATE must match.
DEFAULT_KOKORO_URL = "http://kokoro.kokoro.svc.cluster.local:8000/v1"
DEFAULT_KOKORO_MODEL = "kokoro"
DEFAULT_KOKORO_VOICE = "af_bella"
_READ_CHUNK_BYTES = 4096


def kokoro_base_url() -> str:
    return (os.getenv("FLUXER_VOICE_KOKORO_URL") or DEFAULT_KOKORO_URL).strip().rstrip("/")


def kokoro_model() -> str:
    return (os.getenv("FLUXER_VOICE_KOKORO_MODEL") or DEFAULT_KOKORO_MODEL).strip()


def kokoro_voice() -> str:
    return (os.getenv("FLUXER_VOICE_KOKORO_VOICE") or DEFAULT_KOKORO_VOICE).strip()


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
            api_key if api_key is not None else os.getenv("FLUXER_VOICE_KOKORO_API_KEY", "")
        ).strip()
        self.model = model or kokoro_model()
        self.voice = voice or kokoro_voice()
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
            text, on_audio_delta, timeout=timeout, first_audio_timeout=first_audio_timeout
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
            text, on_audio_delta, timeout=timeout, first_audio_timeout=first_audio_timeout
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
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        result = KokoroSpeechResult(transcript=text, events_seen=["kokoro_speech_request"])

        def _produce() -> None:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result.events_seen.append(f"http_{getattr(response, 'status', 0)}")
                    while True:
                        chunk = response.read(_READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as exc:  # surfaced to the caller via the queue
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = asyncio.create_task(asyncio.to_thread(_produce), name="kokoro-speech-producer")
        try:
            while True:
                wait = timeout
                if not result.bytes_written and first_audio_timeout:
                    wait = min(timeout, first_audio_timeout)
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=wait)
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    if not result.bytes_written:
                        raise RuntimeError(
                            f"Kokoro speech produced no audio within {wait}s"
                        ) from exc
                    break
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise RuntimeError(f"Kokoro speech request failed: {item}") from item
                if not result.bytes_written:
                    result.events_seen.append("first_audio_chunk")
                result.bytes_written += len(item)
                await on_audio_delta(item)
            result.events_seen.append("kokoro_speech_complete")
            return result
        finally:
            if not producer.done():
                producer.cancel()
