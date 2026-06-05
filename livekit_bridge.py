"""Small Fluxer LiveKit smoke bridge.

This module is deliberately transport-only: it can connect to the LiveKit room
that Fluxer returns in VOICE_SERVER_UPDATE, then disconnect cleanly. It does not
publish/listen to audio yet. The goal is to prove the bot token can enter the
room before wiring realtime STT/TTS.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class LiveKitRoomLike(Protocol):
    local_participant: Any

    async def connect(self, url: str, token: str, options: Any = ...) -> None: ...

    async def disconnect(self, **kwargs: Any) -> None: ...

    def on(self, event: str, callback: Any = None) -> Any: ...


RoomFactory = Callable[[], LiveKitRoomLike]


@dataclass(frozen=True)
class FluxerLiveKitConnectionInfo:
    """Non-secret connection metadata captured after a successful smoke join."""

    endpoint: str
    guild_id: Optional[str]
    channel_id: Optional[str]
    connection_id: Optional[str]
    room_name: Optional[str] = None
    participant_identity: Optional[str] = None


def _load_livekit_room_factory() -> tuple[RoomFactory, Callable[..., Any]]:
    """Load LiveKit lazily so normal text/voice-message plugin use stays light."""

    try:
        from livekit import rtc  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message text tests.
        raise RuntimeError(
            "Fluxer realtime voice requires the optional dependency: "
            "pip install 'hermes-fluxer-plugin[realtime]'"
        ) from exc
    return rtc.Room, rtc.RoomOptions


def _load_livekit_audio_helpers() -> Any:
    """Load LiveKit audio helpers lazily for audible smoke tests."""

    try:
        from livekit import rtc  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via room loading path.
        raise RuntimeError(
            "Fluxer realtime voice requires the optional dependency: "
            "pip install 'hermes-fluxer-plugin[realtime]'"
        ) from exc
    return rtc


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _sine_pcm16_frame(
    *,
    start_sample: int,
    samples: int,
    sample_rate: int,
    frequency_hz: float,
    amplitude: float,
) -> bytes:
    amplitude_i16 = int(max(0.0, min(amplitude, 1.0)) * 32767)
    data = bytearray(samples * 2)
    for index in range(samples):
        sample = int(amplitude_i16 * math.sin(2 * math.pi * frequency_hz * (start_sample + index) / sample_rate))
        data[index * 2 : index * 2 + 2] = sample.to_bytes(2, byteorder="little", signed=True)
    return bytes(data)


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class FluxerLiveKitSmokeBridge:
    """Connect/disconnect proof for Fluxer's LiveKit voice-room handoff."""

    def __init__(self, *, room_factory: Optional[RoomFactory] = None, auto_subscribe: bool = False) -> None:
        self._room_factory = room_factory
        self._auto_subscribe = auto_subscribe
        self._room: Optional[LiveKitRoomLike] = None
        self.last_connection: Optional[FluxerLiveKitConnectionInfo] = None

    @property
    def connected(self) -> bool:
        return self._room is not None

    async def connect_from_voice_server_update(self, update: dict[str, Any]) -> FluxerLiveKitConnectionInfo:
        """Connect using a raw Fluxer VOICE_SERVER_UPDATE payload.

        The input may contain the ephemeral LiveKit token. The token is used only
        as a local argument to `Room.connect(...)`; it is never stored on `self`,
        returned, or logged.
        """

        endpoint = _string_or_none(update.get("endpoint"))
        token = _string_or_none(update.get("token"))
        if not endpoint:
            raise ValueError("Fluxer VOICE_SERVER_UPDATE did not include a LiveKit endpoint")
        if not token:
            raise ValueError("Fluxer VOICE_SERVER_UPDATE did not include a LiveKit token")

        await self.disconnect()

        room_factory = self._room_factory
        options: Any = None
        if room_factory is None:
            room_factory, room_options_factory = _load_livekit_room_factory()
            options = room_options_factory(auto_subscribe=self._auto_subscribe)

        room = room_factory()
        if options is None:
            await room.connect(endpoint, token)
        else:
            await room.connect(endpoint, token, options)

        info = FluxerLiveKitConnectionInfo(
            endpoint=endpoint,
            guild_id=_string_or_none(update.get("guild_id")),
            channel_id=_string_or_none(update.get("channel_id")),
            connection_id=_string_or_none(update.get("connection_id")),
            room_name=_string_or_none(getattr(room, "name", None)),
            participant_identity=_string_or_none(getattr(getattr(room, "local_participant", None), "identity", None)),
        )
        self._room = room
        self.last_connection = info
        logger.info(
            "Fluxer LiveKit smoke bridge connected endpoint=%s channel=%s guild=%s connection=%s",
            info.endpoint,
            info.channel_id or "<none>",
            info.guild_id or "<dm>",
            info.connection_id or "<none>",
        )
        return info

    async def publish_test_tone(
        self,
        *,
        duration_seconds: float = 1.0,
        frequency_hz: float = 440.0,
        sample_rate: int = 48_000,
        amplitude: float = 0.18,
        frame_ms: int = 20,
        track_name: str = "zofka-test-tone",
    ) -> None:
        """Publish a short mono PCM sine tone into the connected LiveKit room.

        This is intentionally only an audible smoke test: no microphone input,
        no STT, and no assistant loop. Keep amplitude modest to avoid blasting
        anyone who is already in the voice room.
        """

        if self._room is None:
            raise RuntimeError("Fluxer LiveKit smoke bridge is not connected")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

        rtc = _load_livekit_audio_helpers()
        source = rtc.AudioSource(sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await _maybe_await(self._room.local_participant.publish_track(track, options))
        logger.info("Fluxer LiveKit smoke bridge published test tone track sid=%s", getattr(publication, "sid", "<none>"))

        frame_samples = max(1, sample_rate * frame_ms // 1000)
        total_samples = max(1, int(sample_rate * duration_seconds))
        emitted = 0
        while emitted < total_samples:
            samples = min(frame_samples, total_samples - emitted)
            frame = rtc.AudioFrame(
                _sine_pcm16_frame(
                    start_sample=emitted,
                    samples=samples,
                    sample_rate=sample_rate,
                    frequency_hz=frequency_hz,
                    amplitude=amplitude,
                ),
                sample_rate,
                1,
                samples,
            )
            await _maybe_await(source.capture_frame(frame))
            emitted += samples
        await _maybe_await(source.wait_for_playout())
        close = getattr(source, "aclose", None)
        if close is not None:
            await _maybe_await(close())

    async def publish_wav_file(
        self,
        wav_path: str | Path,
        *,
        frame_ms: int = 20,
        track_name: str = "zofka-tts-smoke",
    ) -> None:
        """Publish a mono 16-bit PCM WAV file into the connected LiveKit room."""

        if self._room is None:
            raise RuntimeError("Fluxer LiveKit smoke bridge is not connected")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

        path = Path(wav_path)
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            if channels != 1 or sample_width != 2:
                raise ValueError("publish_wav_file requires mono 16-bit PCM WAV input")

            rtc = _load_livekit_audio_helpers()
            source = rtc.AudioSource(sample_rate, 1)
            track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
            options = rtc.TrackPublishOptions()
            options.source = rtc.TrackSource.SOURCE_MICROPHONE
            publication = await _maybe_await(self._room.local_participant.publish_track(track, options))
            logger.info("Fluxer LiveKit smoke bridge published WAV track sid=%s", getattr(publication, "sid", "<none>"))

            frame_samples = max(1, sample_rate * frame_ms // 1000)
            while True:
                pcm = wav.readframes(frame_samples)
                if not pcm:
                    break
                samples = len(pcm) // 2
                frame = rtc.AudioFrame(pcm, sample_rate, 1, samples)
                await _maybe_await(source.capture_frame(frame))
            await _maybe_await(source.wait_for_playout())
            close = getattr(source, "aclose", None)
            if close is not None:
                await _maybe_await(close())

    async def publish_pcm16(
        self,
        pcm: bytes,
        *,
        sample_rate: int = 24_000,
        frame_ms: int = 20,
        track_name: str = "zofka-realtime-response",
    ) -> None:
        """Publish mono PCM16 audio bytes into the connected LiveKit room."""

        if self._room is None:
            raise RuntimeError("Fluxer LiveKit smoke bridge is not connected")
        if not pcm:
            raise ValueError("pcm must not be empty")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be positive")

        rtc = _load_livekit_audio_helpers()
        source = rtc.AudioSource(sample_rate, 1)
        track = rtc.LocalAudioTrack.create_audio_track(track_name, source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await _maybe_await(self._room.local_participant.publish_track(track, options))
        logger.info("Fluxer LiveKit bridge published PCM track sid=%s", getattr(publication, "sid", "<none>"))

        frame_samples = max(1, sample_rate * frame_ms // 1000)
        frame_bytes = frame_samples * 2
        for offset in range(0, len(pcm), frame_bytes):
            chunk = pcm[offset : offset + frame_bytes]
            if len(chunk) % 2:
                chunk = chunk[:-1]
            if not chunk:
                continue
            samples = len(chunk) // 2
            frame = rtc.AudioFrame(chunk, sample_rate, 1, samples)
            await _maybe_await(source.capture_frame(frame))
        await _maybe_await(source.wait_for_playout())
        close = getattr(source, "aclose", None)
        if close is not None:
            await _maybe_await(close())

    def iter_remote_audio_pcm16(
        self,
        *,
        sample_rate: int = 24_000,
        frame_size_ms: int = 20,
        participant_identity: str | None = None,
    ) -> AsyncIterator[bytes]:
        """Stream PCM16 mono chunks from subscribed remote LiveKit audio tracks."""

        if self._room is None:
            raise RuntimeError("Fluxer LiveKit smoke bridge is not connected")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if frame_size_ms <= 0:
            raise ValueError("frame_size_ms must be positive")

        rtc = _load_livekit_audio_helpers()
        queue: asyncio.Queue[bytes] = asyncio.Queue()
        stream_tasks: list[asyncio.Task[Any]] = []

        def maybe_track_is_audio(track: Any) -> bool:
            kind = getattr(track, "kind", None)
            return str(kind).lower().endswith("audio") or track.__class__.__name__.lower().endswith("audiotrack")

        async def consume_track(track: Any, participant: Any) -> None:
            identity = getattr(participant, "identity", None)
            if participant_identity and identity != participant_identity:
                return
            stream = rtc.AudioStream.from_track(
                track=track,
                sample_rate=sample_rate,
                num_channels=1,
                frame_size_ms=frame_size_ms,
            )
            try:
                async for event in stream:
                    frame = event.frame
                    data = bytes(frame.data)
                    if data:
                        await queue.put(data)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await _maybe_await(close())

        def on_track_subscribed(track: Any, publication: Any, participant: Any) -> None:
            if maybe_track_is_audio(track):
                stream_tasks.append(asyncio.create_task(consume_track(track, participant)))

        room = self._room
        if room is None:
            raise RuntimeError("Fluxer LiveKit smoke bridge is not connected")

        async def generator() -> AsyncIterator[bytes]:
            room.on("track_subscribed", on_track_subscribed)
            for participant in getattr(room, "remote_participants", {}).values():
                for publication in getattr(participant, "track_publications", {}).values():
                    track = getattr(publication, "track", None)
                    if track is not None and maybe_track_is_audio(track):
                        stream_tasks.append(asyncio.create_task(consume_track(track, participant)))
            try:
                while True:
                    yield await queue.get()
            finally:
                for task in stream_tasks:
                    task.cancel()
                for task in stream_tasks:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

        return generator()

    async def collect_remote_audio_pcm16(
        self,
        *,
        duration_seconds: float = 4.0,
        sample_rate: int = 24_000,
        frame_size_ms: int = 20,
        participant_identity: str | None = None,
        timeout: float = 30.0,
    ) -> bytes:
        """Collect PCM16 mono audio from a subscribed remote LiveKit track.

        Returned bytes are suitable for xAI Realtime `input_audio_buffer.append`
        when the xAI session is configured with the same sample rate.
        """

        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        target_bytes = int(sample_rate * duration_seconds) * 2
        collected = bytearray()
        async with asyncio.timeout(timeout):
            async for chunk in self.iter_remote_audio_pcm16(
                sample_rate=sample_rate,
                frame_size_ms=frame_size_ms,
                participant_identity=participant_identity,
            ):
                collected.extend(chunk)
                if len(collected) >= target_bytes:
                    return bytes(collected[:target_bytes])
        return bytes(collected)

    async def disconnect(self) -> None:
        room = self._room
        self._room = None
        if room is not None:
            result = room.disconnect()
            if asyncio.iscoroutine(result):
                await result
