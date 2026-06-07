"""Minimal xAI Realtime voice client for the Fluxer LiveKit spike.

This is intentionally small and explicit: connect to xAI's realtime websocket,
request PCM16 audio, collect `response.output_audio.delta` chunks, and write a
mono WAV file that the Fluxer LiveKit bridge can publish.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

_XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"


@dataclass(frozen=True)
class XAIRealtimeAudioResult:
    wav_path: Optional[Path]
    sample_rate: int
    bytes_written: int
    events_seen: tuple[str, ...]
    transcript: str = ""


class BargeInInterrupt(Exception):
    """Raised when fresh user speech interrupts assistant playback."""

    _fluxer_fast_close = True


class XAIRealtimeStreamError(RuntimeError):
    """xAI realtime stream failure with sanitized event context."""

    def __init__(self, message: str, *, events_seen: list[str] | tuple[str, ...], cause: BaseException | None = None) -> None:
        self.events_seen = tuple(events_seen)
        self.cause_type = type(cause).__name__ if cause is not None else None
        self.cause_message = str(cause) if cause is not None else ""
        tail = self.events_seen[-12:]
        detail = f"{message}; events_tail={tail}"
        if cause is not None:
            cause_text = str(cause) or repr(cause)
            detail += f"; cause={type(cause).__name__}: {cause_text}"
        super().__init__(detail)


def _xai_realtime_url(model: str) -> str:
    return f"{_XAI_REALTIME_URL}?model={model}"


def _decode_audio_delta(event: dict[str, Any]) -> bytes:
    delta = event.get("delta") or event.get("audio") or event.get("data")
    if not isinstance(delta, str) or not delta:
        return b""
    return base64.b64decode(delta, validate=True)


def _write_pcm16_wav(path: str | Path, pcm: bytes, *, sample_rate: int) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output


async def _connect_websocket(url: str, *, api_key: str) -> Any:
    import websockets

    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        return await websockets.connect(url, additional_headers=headers, open_timeout=15, close_timeout=5, max_size=None)
    except TypeError:  # websockets < 14 used extra_headers.
        return await websockets.connect(url, extra_headers=headers, open_timeout=15, close_timeout=5, max_size=None)


class XAIRealtimeVoiceClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: str = "grok-voice-latest",
        voice: str = "eve",
        sample_rate: int = 24_000,
        instructions: str = (
            "You are the configured assistant, warm, direct, and concise. "
            "Answer in the user's language when it is clear; otherwise default to English."
        ),
    ) -> None:
        self.api_key = (api_key or os.getenv("XAI_API_KEY") or "").strip()
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
        self.instructions = instructions

    async def text_response_to_wav(
        self,
        text: str,
        output_path: str | Path,
        *,
        timeout: float = 30.0,
    ) -> XAIRealtimeAudioResult:
        """Ask xAI Realtime to answer text and write returned PCM16 audio as WAV."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not text.strip():
            raise ValueError("text must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(self._text_response_to_wav_on_ws(ws, text, output_path, timeout=timeout), timeout=timeout + 1.0)
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    async def text_response_to_sink(
        self,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        """Ask xAI Realtime to answer text and stream returned PCM16 deltas to a sink."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not text.strip():
            raise ValueError("text must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(
                    self._text_response_to_sink_on_ws(
                        ws,
                        text,
                        on_audio_delta,
                        timeout=timeout,
                        first_audio_timeout=first_audio_timeout,
                    ),
                    timeout=timeout + 1.0,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    async def force_message_to_wav(
        self,
        text: str,
        output_path: str | Path,
        *,
        timeout: float = 30.0,
        interruptible: bool = False,
    ) -> XAIRealtimeAudioResult:
        """Use xAI's realtime `force_message` extension for verbatim TTS over websocket."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not text.strip():
            raise ValueError("text must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(
                    self._force_message_to_wav_on_ws(ws, text, output_path, timeout=timeout, interruptible=interruptible),
                    timeout=timeout + 1.0,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    async def force_message_to_sink(
        self,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
        interruptible: bool = False,
    ) -> XAIRealtimeAudioResult:
        """Use xAI `force_message` for verbatim TTS and stream PCM deltas to a sink."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not text.strip():
            raise ValueError("text must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(
                    self._force_message_to_sink_on_ws(
                        ws,
                        text,
                        on_audio_delta,
                        timeout=timeout,
                        first_audio_timeout=first_audio_timeout,
                        interruptible=interruptible,
                    ),
                    timeout=timeout + 1.0,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    async def _configure_session(self, ws: Any) -> None:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "voice": self.voice,
                        "instructions": self.instructions,
                        "turn_detection": None,
                        "audio": {
                            "input": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                            "output": {"format": {"type": "audio/pcm", "rate": self.sample_rate}},
                        },
                    },
                }
            )
        )

    async def audio_response_from_pcm16(
        self,
        pcm_audio: bytes,
        output_path: str | Path,
        *,
        timeout: float = 45.0,
    ) -> XAIRealtimeAudioResult:
        """Send mono PCM16 user audio into xAI Realtime and write the voice response."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not pcm_audio:
            raise ValueError("pcm_audio must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(
                    self._audio_response_from_pcm16_on_ws(ws, pcm_audio, output_path, timeout=timeout),
                    timeout=timeout + 1.0,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    async def audio_response_from_pcm16_to_sink(
        self,
        pcm_audio: bytes,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 45.0,
        first_audio_timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        """Stream xAI Realtime output PCM deltas to a sink as soon as they arrive."""

        if not self.api_key:
            raise RuntimeError("XAI_API_KEY is required for xAI Realtime")
        if not pcm_audio:
            raise ValueError("pcm_audio must not be empty")
        ws = await _connect_websocket(_xai_realtime_url(self.model), api_key=self.api_key)
        try:
            try:
                return await asyncio.wait_for(
                    self._audio_response_from_pcm16_to_sink_on_ws(
                        ws,
                        pcm_audio,
                        on_audio_delta,
                        timeout=timeout,
                        first_audio_timeout=first_audio_timeout,
                    ),
                    timeout=timeout + 1.0,
                )
            except (TimeoutError, asyncio.TimeoutError) as exc:
                raise XAIRealtimeStreamError(
                    f"xAI Realtime response did not finish within {timeout}s",
                    events_seen=[],
                    cause=exc,
                ) from exc
        finally:
            await ws.close()

    def _remaining_timeout(self, started: float, timeout: Optional[float]) -> Optional[float]:
        if timeout is None:
            return None
        return max(0.001, timeout - (asyncio.get_running_loop().time() - started))

    async def _wait_for_setup(self, awaitable: Awaitable[Any], *, started: float, timeout: Optional[float]) -> None:
        try:
            remaining = self._remaining_timeout(started, timeout)
            if remaining is None:
                await awaitable
            else:
                await asyncio.wait_for(awaitable, timeout=remaining)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise XAIRealtimeStreamError(
                f"xAI Realtime response did not finish within {timeout}s",
                events_seen=[],
                cause=exc,
            ) from exc

    async def _text_response_to_sink_on_ws(
        self,
        ws: Any,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_text_response_request(ws, text), started=started, timeout=timeout)
        return await self._collect_audio_to_sink(
            ws,
            on_audio_delta,
            timeout=self._remaining_timeout(started, timeout),
            first_audio_timeout=first_audio_timeout,
        )

    async def _text_response_to_wav_on_ws(
        self,
        ws: Any,
        text: str,
        output_path: str | Path,
        *,
        timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_text_response_request(ws, text), started=started, timeout=timeout)
        return await self._collect_audio_to_wav(ws, output_path, timeout=self._remaining_timeout(started, timeout))

    async def _send_text_response_request(self, ws: Any, text: str) -> None:
        await self._configure_session(ws)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))

    async def _audio_response_from_pcm16_on_ws(
        self,
        ws: Any,
        pcm_audio: bytes,
        output_path: str | Path,
        *,
        timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_audio_response_request(ws, pcm_audio), started=started, timeout=timeout)
        return await self._collect_audio_to_wav(ws, output_path, timeout=self._remaining_timeout(started, timeout))

    async def _audio_response_from_pcm16_to_sink_on_ws(
        self,
        ws: Any,
        pcm_audio: bytes,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 45.0,
        first_audio_timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_audio_response_request(ws, pcm_audio), started=started, timeout=timeout)
        return await self._collect_audio_to_sink(
            ws,
            on_audio_delta,
            timeout=self._remaining_timeout(started, timeout),
            first_audio_timeout=first_audio_timeout,
        )

    async def _send_audio_response_request(self, ws: Any, pcm_audio: bytes) -> None:
        await self._configure_session(ws)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "audio": base64.b64encode(pcm_audio).decode("ascii"),
                            }
                        ],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))

    async def _force_message_to_wav_on_ws(
        self,
        ws: Any,
        text: str,
        output_path: str | Path,
        *,
        timeout: Optional[float] = None,
        interruptible: bool,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_force_message_request(ws, text, interruptible=interruptible), started=started, timeout=timeout)
        return await self._collect_audio_to_wav(ws, output_path, timeout=self._remaining_timeout(started, timeout))

    async def _force_message_to_sink_on_ws(
        self,
        ws: Any,
        text: str,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: float = 30.0,
        first_audio_timeout: Optional[float] = None,
        interruptible: bool,
    ) -> XAIRealtimeAudioResult:
        started = asyncio.get_running_loop().time()
        await self._wait_for_setup(self._send_force_message_request(ws, text, interruptible=interruptible), started=started, timeout=timeout)
        return await self._collect_audio_to_sink(
            ws,
            on_audio_delta,
            timeout=self._remaining_timeout(started, timeout),
            first_audio_timeout=first_audio_timeout,
        )

    async def _send_force_message_request(self, ws: Any, text: str, *, interruptible: bool) -> None:
        await self._configure_session(ws)
        await ws.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "force_message",
                        "role": "assistant",
                        "interruptible": interruptible,
                        "content": [{"type": "output_text", "text": text}],
                    },
                }
            )
        )
        await ws.send(json.dumps({"type": "response.create"}))

    async def _collect_audio_to_wav(
        self,
        ws: Any,
        output_path: str | Path,
        *,
        timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        pcm_parts: list[bytes] = []

        async def append_delta(chunk: bytes) -> None:
            pcm_parts.append(chunk)

        result = await self._collect_audio_to_sink(ws, append_delta, timeout=timeout)
        pcm = b"".join(pcm_parts)
        wav_path = _write_pcm16_wav(output_path, pcm, sample_rate=self.sample_rate)
        return XAIRealtimeAudioResult(
            wav_path=wav_path,
            sample_rate=result.sample_rate,
            bytes_written=result.bytes_written,
            events_seen=result.events_seen,
            transcript=result.transcript,
        )

    async def _collect_audio_to_sink(
        self,
        ws: Any,
        on_audio_delta: Callable[[bytes], Awaitable[None]],
        *,
        timeout: Optional[float] = None,
        first_audio_timeout: Optional[float] = None,
    ) -> XAIRealtimeAudioResult:
        bytes_written = 0
        events: list[str] = []
        transcript_parts: list[str] = []
        iterator = ws.__aiter__()
        started = asyncio.get_running_loop().time()
        while True:
            read_timeout: Optional[float] = None
            try:
                remaining_timeout = None
                if timeout is not None:
                    remaining_timeout = max(0.001, timeout - (asyncio.get_running_loop().time() - started))
                read_timeout = remaining_timeout
                if bytes_written == 0 and first_audio_timeout is not None:
                    read_timeout = min(first_audio_timeout, remaining_timeout) if remaining_timeout is not None else first_audio_timeout
                raw = await asyncio.wait_for(anext(iterator), timeout=read_timeout) if read_timeout is not None else await anext(iterator)
            except StopAsyncIteration:
                break
            except (TimeoutError, asyncio.TimeoutError) as exc:
                if bytes_written == 0:
                    message = f"xAI Realtime emitted no audio within {read_timeout}s"
                else:
                    message = f"xAI Realtime response did not finish within {timeout}s"
                raise XAIRealtimeStreamError(
                    message,
                    events_seen=events,
                    cause=exc,
                ) from exc
            try:
                event = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise XAIRealtimeStreamError(
                    "xAI Realtime returned malformed JSON event",
                    events_seen=events,
                    cause=exc,
                ) from exc
            event_type = str(event.get("type") or "")
            if event_type:
                events.append(event_type)
            if event_type == "error":
                message = event.get("error") or event
                raise XAIRealtimeStreamError(
                    f"xAI Realtime error: {message}",
                    events_seen=events,
                )
            if event_type == "response.output_audio.delta":
                try:
                    chunk = _decode_audio_delta(event)
                except Exception as exc:
                    raise XAIRealtimeStreamError(
                        "xAI Realtime returned malformed audio delta",
                        events_seen=events,
                        cause=exc,
                    ) from exc
                if chunk:
                    try:
                        await on_audio_delta(chunk)
                    except Exception as exc:
                        if isinstance(exc, BargeInInterrupt):
                            raise
                        raise XAIRealtimeStreamError(
                            "xAI audio sink failed while handling output delta",
                            events_seen=events,
                            cause=exc,
                        ) from exc
                    bytes_written += len(chunk)
            elif event_type == "response.output_audio_transcript.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    transcript_parts.append(delta)
            elif event_type == "response.done":
                break
        if bytes_written <= 0:
            raise XAIRealtimeStreamError("xAI Realtime returned no audio", events_seen=events)
        return XAIRealtimeAudioResult(
            wav_path=None,
            sample_rate=self.sample_rate,
            bytes_written=bytes_written,
            events_seen=tuple(events),
            transcript="".join(transcript_parts),
        )
