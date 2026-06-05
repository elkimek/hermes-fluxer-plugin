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
from typing import Any, Optional

_XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"


@dataclass(frozen=True)
class XAIRealtimeAudioResult:
    wav_path: Path
    sample_rate: int
    bytes_written: int
    events_seen: tuple[str, ...]
    transcript: str = ""


def _xai_realtime_url(model: str) -> str:
    return f"{_XAI_REALTIME_URL}?model={model}"


def _decode_audio_delta(event: dict[str, Any]) -> bytes:
    delta = event.get("delta") or event.get("audio") or event.get("data")
    if not isinstance(delta, str) or not delta:
        return b""
    return base64.b64decode(delta)


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
        instructions: str = "You are Žofka, warm, direct, and concise.",
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
            return await asyncio.wait_for(self._text_response_to_wav_on_ws(ws, text, output_path), timeout=timeout)
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
            return await asyncio.wait_for(
                self._force_message_to_wav_on_ws(ws, text, output_path, interruptible=interruptible),
                timeout=timeout,
            )
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

    async def _text_response_to_wav_on_ws(self, ws: Any, text: str, output_path: str | Path) -> XAIRealtimeAudioResult:
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
        return await self._collect_audio_to_wav(ws, output_path)

    async def _force_message_to_wav_on_ws(
        self,
        ws: Any,
        text: str,
        output_path: str | Path,
        *,
        interruptible: bool,
    ) -> XAIRealtimeAudioResult:
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
        return await self._collect_audio_to_wav(ws, output_path)

    async def _collect_audio_to_wav(self, ws: Any, output_path: str | Path) -> XAIRealtimeAudioResult:
        pcm_parts: list[bytes] = []
        events: list[str] = []
        transcript_parts: list[str] = []
        async for raw in ws:
            event = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
            event_type = str(event.get("type") or "")
            if event_type:
                events.append(event_type)
            if event_type == "error":
                message = event.get("error") or event
                raise RuntimeError(f"xAI Realtime error: {message}")
            if event_type == "response.output_audio.delta":
                chunk = _decode_audio_delta(event)
                if chunk:
                    pcm_parts.append(chunk)
            elif event_type == "response.output_audio_transcript.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    transcript_parts.append(delta)
            elif event_type == "response.done":
                break
        pcm = b"".join(pcm_parts)
        if not pcm:
            raise RuntimeError(f"xAI Realtime returned no audio; events={events}")
        wav_path = _write_pcm16_wav(output_path, pcm, sample_rate=self.sample_rate)
        return XAIRealtimeAudioResult(
            wav_path=wav_path,
            sample_rate=self.sample_rate,
            bytes_written=len(pcm),
            events_seen=tuple(events),
            transcript="".join(transcript_parts),
        )
