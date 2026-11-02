"""Kokoro (OpenAI-compatible) speech output for the Fluxer realtime voice loop.

Used when no xAI API key is configured: replies are voiced by the cluster's
Kokoro TTS service and streamed as raw mono PCM16 chunks into an async sink,
mirroring the xAI Realtime client's sink contract so the voice loop's turn
logic stays unchanged.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("fluxer_kokoro_tts")

# Kokoro emits mono PCM16 at 24 kHz; FLUXER_VOICE_SAMPLE_RATE must match.
# The endpoint and the voice carry no default: both are deployment choices
# (which service, which language), so FLUXER_VOICE_KOKORO_URL and
# FLUXER_VOICE_KOKORO_VOICE must name them.
DEFAULT_KOKORO_MODEL = "kokoro"
KOKORO_SAMPLE_RATE = 24_000
_READ_CHUNK_BYTES = 4096
# 256 chunks is about 21 s of 24 kHz mono PCM16. The producer waits for space
# rather than dropping audio when it fills, so this is backpressure, not a cap.
_QUEUE_CHUNKS = 256
# Marks the end of the stream; distinct from a chunk so a falsy item cannot be
# mistaken for it.
_END = object()


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
            api_key if api_key is not None else os.getenv("FLUXER_VOICE_KOKORO_API_KEY", "")
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
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_CHUNKS)
        # urllib's read() blocks in a worker thread asyncio cannot cancel, so the
        # producer watches this flag and stops at the next chunk boundary; the
        # `with` block then closes the response.
        stop = threading.Event()
        pending_puts: set[concurrent.futures.Future[None]] = set()
        result = KokoroSpeechResult(transcript=text, events_seen=["kokoro_speech_request"])

        def _put_blocking(item: Any) -> bool:
            """Wait for queue space from the producer thread; False once stopped."""
            while not stop.is_set():
                try:
                    future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
                except RuntimeError:  # the event loop is gone
                    return False
                pending_puts.add(future)
                try:
                    future.result()
                    return True
                except (concurrent.futures.CancelledError, RuntimeError):
                    return False
                finally:
                    pending_puts.discard(future)
            return False

        def _produce() -> None:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    result.events_seen.append(f"http_{getattr(response, 'status', 0)}")
                    while not stop.is_set():
                        chunk = response.read(_READ_CHUNK_BYTES)
                        if not chunk:
                            break
                        # Backpressure, not a cap: a sink slower than the stream
                        # must not cost audio, so wait for space instead of
                        # dropping the chunk.
                        if not _put_blocking(chunk):
                            return
            except Exception as exc:  # surfaced to the caller via the queue
                if not stop.is_set():
                    _put_blocking(exc)
            finally:
                # Never dropped: a healthy stream that filled the queue must
                # still deliver the end marker, or the reply would time out.
                if not stop.is_set():
                    _put_blocking(_END)

        producer = asyncio.create_task(asyncio.to_thread(_produce), name="kokoro-speech-producer")
        started = loop.time()
        deadline = started + timeout
        first_deadline = deadline
        if first_audio_timeout:
            first_deadline = min(deadline, started + first_audio_timeout)
        try:
            while True:
                # One deadline for the whole reply: a chunk arriving just inside
                # the per-chunk wait must not extend the turn indefinitely.
                limit = deadline if result.bytes_written else first_deadline
                wait = limit - loop.time()
                if wait <= 0:
                    raise RuntimeError(self._timeout_message(result, timeout, first_audio_timeout))
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=wait)
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise RuntimeError(
                        self._timeout_message(result, timeout, first_audio_timeout)
                    ) from exc
                if item is _END:
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
            stop.set()
            for future in list(pending_puts):
                # A producer waiting for space must not outlive the consumer.
                future.cancel()
            if not producer.done():
                producer.cancel()

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
