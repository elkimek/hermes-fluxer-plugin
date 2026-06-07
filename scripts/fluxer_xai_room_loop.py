#!/usr/bin/env python3
"""Continuous Fluxer voice room ↔ xAI Realtime turn loop.

This is the first daemon-shaped prototype after the one-turn smoke:
1. join a Fluxer voice channel,
2. connect to the LiveKit room from VOICE_SERVER_UPDATE,
3. segment remote PCM16 audio into speech turns,
4. send each turn to xAI Realtime,
5. publish Grok Voice audio back into the same Fluxer LiveKit room,
6. keep listening until max turns/runtime or Ctrl-C.

Secrets stay in-process only. Printed diagnostics contain safe metadata only.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from livekit_bridge import FluxerLiveKitSmokeBridge  # noqa: E402
from xai_realtime import XAIRealtimeVoiceClient  # noqa: E402

logger = logging.getLogger("fluxer_xai_room_loop")


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _redact_exception_message(exc: Exception, *secrets: str | None) -> str:
    message = str(exc) or repr(exc)
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted-token]")
    message = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted-token]", message)
    message = re.sub(r"token=([A-Za-z0-9._~+/=-]+)", "token=[redacted-token]", message, flags=re.IGNORECASE)
    message = re.sub(r'("token"\s*:\s*")([^"]+)(")', r"\1[redacted-token]\3", message, flags=re.IGNORECASE)
    return message[:500]


class BargeInInterrupt(Exception):
    """Raised when fresh user speech interrupts assistant playback."""


@dataclass
class BargeInCapture:
    """State shared between assistant playback and the barge-in listener."""

    event: asyncio.Event = field(default_factory=asyncio.Event)
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    pcm: bytes = b""
    captured_audio_seconds: float = 0.0
    chunks_seen: int = 0
    first_chunk_seconds: float | None = None
    max_rms: int = 0
    voiced_ms: int = 0
    detected_seconds: float | None = None


DEFAULT_INSTRUCTIONS = """You are the configured assistant speaking with a user in a Fluxer voice room.
Answer in the user's language when it is clear; otherwise default to English.
Ignore background music, lyrics, radio, TV, and room noise. Respond only to speech that sounds directed at the assistant or clearly part of the conversation.
Be warm, direct, concise, and natural for realtime voice. Default to one short sentence. Do not ask multiple follow-up questions.
""".strip()

WAKE_GATE_INSTRUCTIONS = """You are a strict realtime voice gate for a Fluxer voice room.
Listen to the user's audio. If the speech is not clearly directed at the assistant or clearly part of the conversation, or if it sounds like music, lyrics, radio, TV, or background noise, reply with exactly: IGNORE
If the user clearly addresses the assistant or continues the conversation, reply with exactly: RESPOND
Use English only.
""".strip()


def _pcm16_rms(pcm: bytes) -> int:
    if len(pcm) < 2:
        return 0
    if len(pcm) % 2:
        pcm = pcm[:-1]
    total = 0
    samples = len(pcm) // 2
    for offset in range(0, len(pcm), 2):
        sample = int.from_bytes(pcm[offset : offset + 2], byteorder="little", signed=True)
        total += sample * sample
    return int((total / samples) ** 0.5) if samples else 0


def _pcm16_duration_seconds(pcm: bytes, *, sample_rate: int) -> float:
    if sample_rate <= 0:
        return 0.0
    return (len(pcm) // 2) / sample_rate


def _barge_in_carryover_decision(args: argparse.Namespace, pcm: bytes, *, sample_rate: int) -> tuple[bytes | None, bool, float]:
    """Return usable carryover PCM, discard flag, and duration.

    Very short barge-in snippets are useful as an interruption signal but are
    too partial to send to xAI as the next user turn. Re-listen instead.
    """

    duration = _pcm16_duration_seconds(pcm, sample_rate=sample_rate)
    min_seconds = getattr(args, "min_segment_ms", 750) / 1000
    discarded = bool(pcm and duration < min_seconds)
    return (None if discarded else (pcm or None), discarded, duration)


async def _speech_segments(
    chunks: AsyncIterator[bytes],
    *,
    sample_rate: int,
    energy_threshold: int,
    silence_ms: int,
    end_padding_ms: int,
    min_segment_ms: int,
    max_segment_seconds: float,
) -> AsyncIterator[bytes]:
    """Yield PCM16 speech-ish segments from a chunk iterator using simple VAD.

    `silence_ms` controls how quickly we decide the speaker stopped. Only
    `end_padding_ms` of that final silence is kept in the PCM sent to xAI, so
    short natural pauses stay intact but we do not ask the model to process a
    long dead tail.
    """

    if end_padding_ms < 0 or end_padding_ms > silence_ms:
        raise ValueError("end_padding_ms must be between 0 and silence_ms")

    bytes_per_ms = sample_rate * 2 / 1000
    silence_bytes_limit = int(bytes_per_ms * silence_ms)
    end_padding_bytes = int(bytes_per_ms * end_padding_ms)
    min_bytes = int(bytes_per_ms * min_segment_ms)
    max_bytes = int(sample_rate * max_segment_seconds) * 2

    segment = bytearray()
    trailing_silence = 0
    in_segment = False

    async for chunk in chunks:
        if not chunk:
            continue
        rms = _pcm16_rms(chunk)
        voiced = rms >= energy_threshold
        if voiced:
            in_segment = True
            trailing_silence = 0
            segment.extend(chunk)
        elif in_segment:
            segment.extend(chunk)
            trailing_silence += len(chunk)

        if in_segment and (trailing_silence >= silence_bytes_limit or len(segment) >= max_bytes):
            if trailing_silence > end_padding_bytes:
                trim_bytes = trailing_silence - end_padding_bytes
                if trim_bytes > 0:
                    del segment[-trim_bytes:]
            pcm = bytes(segment)
            segment.clear()
            trailing_silence = 0
            in_segment = False
            if len(pcm) >= min_bytes:
                yield pcm
    if in_segment and segment:
        if trailing_silence > end_padding_bytes:
            trim_bytes = trailing_silence - end_padding_bytes
            if trim_bytes > 0:
                del segment[-trim_bytes:]
        pcm = bytes(segment)
        if len(pcm) >= min_bytes:
            yield pcm


async def _capture_one_speech_segment(
    args: argparse.Namespace,
    bridge: FluxerLiveKitSmokeBridge,
    *,
    timeout: float,
) -> bytes:
    """Capture one speech segment, then close the LiveKit consumer to drop stale backlog."""

    chunks = bridge.iter_remote_audio_pcm16(
        sample_rate=args.sample_rate,
        frame_size_ms=args.frame_ms,
        participant_identity=args.participant_identity,
        participant_identity_prefix=getattr(args, "participant_identity_prefix", None),
    )
    segments = _speech_segments(
        chunks,
        sample_rate=args.sample_rate,
        energy_threshold=args.energy_threshold,
        silence_ms=args.silence_ms,
        end_padding_ms=args.end_padding_ms,
        min_segment_ms=args.min_segment_ms,
        max_segment_seconds=args.max_segment_seconds,
    )
    try:
        return await asyncio.wait_for(anext(segments), timeout=timeout)
    finally:
        close_segments = getattr(segments, "aclose", None)
        if close_segments is not None:
            await close_segments()
        close_chunks = getattr(chunks, "aclose", None)
        if close_chunks is not None:
            await close_chunks()


async def _wait_for_barge_in(
    args: argparse.Namespace,
    bridge: FluxerLiveKitSmokeBridge,
    capture_or_event: BargeInCapture | asyncio.Event,
) -> None:
    """Detect sustained fresh user speech and retain it as carryover PCM."""

    if isinstance(capture_or_event, BargeInCapture):
        capture = capture_or_event
    else:
        capture = BargeInCapture(event=capture_or_event)

    chunks = bridge.iter_remote_audio_pcm16(
        sample_rate=args.sample_rate,
        frame_size_ms=args.frame_ms,
        participant_identity=args.participant_identity,
        participant_identity_prefix=getattr(args, "participant_identity_prefix", None),
    )
    bytes_per_ms = args.sample_rate * 2 / 1000
    silence_ms = getattr(args, "silence_ms", 600)
    end_padding_ms = getattr(args, "end_padding_ms", min(180, silence_ms))
    min_segment_ms = getattr(args, "min_segment_ms", args.barge_in_min_ms)
    max_segment_seconds = getattr(args, "max_segment_seconds", 8.0)
    silence_bytes_limit = int(bytes_per_ms * silence_ms)
    end_padding_bytes = int(bytes_per_ms * end_padding_ms)
    min_bytes = int(bytes_per_ms * min_segment_ms)
    max_bytes = int(args.sample_rate * max_segment_seconds) * 2
    segment = bytearray()
    trailing_silence = 0
    voiced_ms = 0
    in_segment = False
    started = time.monotonic()
    logger.debug(
        "barge listener started threshold=%s min_ms=%s participant_identity=%s",
        args.barge_in_energy_threshold,
        args.barge_in_min_ms,
        args.participant_identity,
    )
    try:
        async for chunk in chunks:
            if capture.stop_event.is_set():
                return
            if not chunk:
                continue
            rms = _pcm16_rms(chunk)
            now = time.monotonic()
            capture.chunks_seen += 1
            if capture.first_chunk_seconds is None:
                capture.first_chunk_seconds = now - started
                logger.debug("barge listener first chunk after %.3fs", capture.first_chunk_seconds)
            capture.max_rms = max(capture.max_rms, rms)
            voiced = rms >= args.barge_in_energy_threshold
            if voiced:
                in_segment = True
                trailing_silence = 0
                voiced_ms += args.frame_ms
                capture.voiced_ms = voiced_ms
                segment.extend(chunk)
                if getattr(args, "verbose", False):
                    logger.debug(
                        "barge listener voiced chunk=%s rms=%s max_rms=%s voiced_ms=%s threshold=%s",
                        capture.chunks_seen,
                        rms,
                        capture.max_rms,
                        voiced_ms,
                        args.barge_in_energy_threshold,
                    )
                if voiced_ms >= args.barge_in_min_ms:
                    if not capture.event.is_set():
                        capture.detected_seconds = time.monotonic() - started
                        logger.info(
                            "Barge-in detected after %.3fs chunks=%s max_rms=%s voiced_ms=%s",
                            capture.detected_seconds,
                            capture.chunks_seen,
                            capture.max_rms,
                            voiced_ms,
                        )
                    capture.event.set()
            elif in_segment:
                voiced_ms = 0
                segment.extend(chunk)
                trailing_silence += len(chunk)
            else:
                voiced_ms = 0

            if in_segment and (trailing_silence >= silence_bytes_limit or len(segment) >= max_bytes):
                if trailing_silence > end_padding_bytes:
                    trim_bytes = trailing_silence - end_padding_bytes
                    if trim_bytes > 0:
                        del segment[-trim_bytes:]
                if len(segment) >= min_bytes and capture.event.is_set():
                    capture.pcm = bytes(segment)
                    capture.captured_audio_seconds = _pcm16_duration_seconds(capture.pcm, sample_rate=args.sample_rate)
                    capture.ready.set()
                return
    finally:
        if segment and capture.event.is_set() and not capture.ready.is_set():
            capture.pcm = bytes(segment)
            capture.captured_audio_seconds = _pcm16_duration_seconds(capture.pcm, sample_rate=args.sample_rate)
            capture.ready.set()
        close_chunks = getattr(chunks, "aclose", None)
        if close_chunks is not None:
            await close_chunks()


async def _conversation_loop(args: argparse.Namespace, bridge: FluxerLiveKitSmokeBridge) -> dict[str, Any]:
    turns: list[dict[str, Any]] = []
    started = time.monotonic()
    xai = XAIRealtimeVoiceClient(
        model=args.xai_model,
        voice=args.xai_voice,
        sample_rate=args.sample_rate,
        instructions=args.xai_instructions,
    )
    gate_xai = XAIRealtimeVoiceClient(
        model=args.xai_model,
        voice=args.xai_voice,
        sample_rate=args.sample_rate,
        instructions=args.wake_gate_instructions,
    )
    carryover_pcm: bytes | None = None
    while True:
        if args.max_turns and len(turns) >= args.max_turns:
            break
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            break
        remaining = args.max_runtime_seconds - (time.monotonic() - started) if args.max_runtime_seconds else 30.0
        from_barge_in_carryover = carryover_pcm is not None
        if carryover_pcm is not None:
            pcm = carryover_pcm
            carryover_pcm = None
            capture_seconds = 0.0
            captured_audio_seconds = _pcm16_duration_seconds(pcm, sample_rate=args.sample_rate)
        else:
            try:
                capture_started = time.monotonic()
                pcm = await _capture_one_speech_segment(args, bridge, timeout=max(1.0, remaining))
                capture_seconds = time.monotonic() - capture_started
                captured_audio_seconds = _pcm16_duration_seconds(pcm, sample_rate=args.sample_rate)
            except (TimeoutError, asyncio.TimeoutError, StopAsyncIteration):
                break
        turn_no = len(turns) + 1
        logger.info("Captured speech turn %s bytes=%s rms=%s", turn_no, len(pcm), _pcm16_rms(pcm))
        gate_transcript = ""
        gate_seconds = 0.0
        if not args.disable_wake_gate:
            gate_wav = str(Path(tempfile.gettempdir()) / f"fluxer_xai_room_loop_gate_{turn_no}.wav")
            try:
                gate_started = time.monotonic()
                gate_result = await gate_xai.audio_response_from_pcm16(pcm, gate_wav, timeout=args.xai_timeout)
                gate_seconds = time.monotonic() - gate_started
                gate_transcript = gate_result.transcript.strip()
                gate_decision = gate_transcript.upper().strip().split()
                should_respond = bool(gate_decision and gate_decision[0] == "RESPOND")
            except Exception as exc:
                gate_transcript = f"{type(exc).__name__}: wake gate failed"
                should_respond = False
            if not should_respond:
                logger.info("Wake gate ignored turn %s transcript=%r", turn_no, gate_transcript)
                turns.append(
                    {
                        "turn": turn_no,
                        "captured_pcm_bytes": len(pcm),
                        "gate_transcript": gate_transcript,
                        "published": False,
                        "ignored": True,
                        "from_barge_in_carryover": from_barge_in_carryover,
                        "timing": {
                            "capture_seconds": round(capture_seconds, 3),
                            "captured_audio_seconds": round(captured_audio_seconds, 3),
                            "gate_seconds": round(gate_seconds, 3),
                        },
                    }
                )
                if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                    break
                continue

        first_audio_seconds: float | None = None
        barge_in_seconds: float | None = None
        publisher: Any | None = None
        barge_in_capture = BargeInCapture()
        try:
            xai_started = time.monotonic()
            barge_in_task: asyncio.Task[Any] | None = None
            publisher = bridge.pcm16_publisher(
                sample_rate=args.sample_rate,
                frame_ms=args.frame_ms,
                track_name=f"fluxer-xai-room-loop-{turn_no}",
            )
            await publisher.__aenter__()
            arm_barge_after_first_audio = bool(getattr(args, "barge_in_after_first_audio_only", False))
            if not args.disable_barge_in and not arm_barge_after_first_audio:
                barge_in_task = asyncio.create_task(_wait_for_barge_in(args, bridge, barge_in_capture))

            async def publish_delta(chunk: bytes) -> None:
                nonlocal first_audio_seconds, barge_in_seconds, barge_in_task
                assert publisher is not None

                async def should_interrupt() -> bool:
                    nonlocal barge_in_seconds
                    if barge_in_capture.event.is_set():
                        barge_in_seconds = time.monotonic() - xai_started
                        return True
                    return False

                if await should_interrupt():
                    await publisher.interrupt()
                    raise BargeInInterrupt("user interrupted assistant speech")
                if first_audio_seconds is None:
                    first_audio_seconds = time.monotonic() - xai_started
                    if arm_barge_after_first_audio and not args.disable_barge_in and barge_in_task is None:
                        barge_in_task = asyncio.create_task(_wait_for_barge_in(args, bridge, barge_in_capture))
                write_interruptible = getattr(publisher, "write_interruptible", None)
                if write_interruptible is not None:
                    interrupted = await write_interruptible(chunk, should_interrupt)
                    if interrupted:
                        raise BargeInInterrupt("user interrupted assistant speech")
                else:
                    await publisher.write(chunk)

            try:
                xai_task = asyncio.create_task(
                    xai.audio_response_from_pcm16_to_sink(
                        pcm,
                        publish_delta,
                        timeout=args.xai_timeout,
                        first_audio_timeout=args.xai_first_audio_timeout,
                    )
                )
                barge_event_task: asyncio.Task[Any] | None = None
                if not args.disable_barge_in and not arm_barge_after_first_audio:
                    barge_event_task = asyncio.create_task(barge_in_capture.event.wait())
                try:
                    if barge_event_task is None:
                        xai_result = await xai_task
                    else:
                        done, _pending = await asyncio.wait(
                            {xai_task, barge_event_task},
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if barge_event_task in done and barge_in_capture.event.is_set() and not xai_task.done():
                            assert publisher is not None
                            barge_in_seconds = time.monotonic() - xai_started
                            await publisher.interrupt()
                            xai_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await xai_task
                            raise BargeInInterrupt("user interrupted assistant speech before xAI audio")
                        xai_result = await xai_task
                finally:
                    if barge_event_task is not None:
                        barge_event_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await barge_event_task
                xai_seconds = time.monotonic() - xai_started
                if barge_in_capture.event.is_set():
                    assert publisher is not None
                    barge_in_seconds = time.monotonic() - xai_started
                    await publisher.interrupt()
                    raise BargeInInterrupt("user interrupted assistant speech")
            finally:
                if barge_in_task is not None:
                    if barge_in_capture.event.is_set():
                        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                            await asyncio.wait_for(
                                barge_in_capture.ready.wait(),
                                timeout=getattr(args, "barge_in_capture_timeout", 2.0),
                            )
                    barge_in_capture.stop_event.set()
                    barge_in_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                        await barge_in_task
                publish_started = time.monotonic()
                if publisher is not None and publisher.interrupted:
                    await publisher.close(wait_for_playout=False, flush_remainder=False)
                elif publisher is not None:
                    await publisher.close()
            publish_seconds = time.monotonic() - publish_started
        except BargeInInterrupt:
            logger.info("Barge-in interrupted turn %s", turn_no)
            raw_carryover_pcm = barge_in_capture.pcm or b""
            carryover_pcm, carryover_discarded, raw_carryover_seconds = _barge_in_carryover_decision(
                args,
                raw_carryover_pcm,
                sample_rate=args.sample_rate,
            )
            min_carryover_seconds = getattr(args, "min_segment_ms", 750) / 1000
            if carryover_discarded:
                logger.info(
                    "Discarding short barge-in carryover for turn %s duration=%.3fs min=%.3fs",
                    turn_no,
                    raw_carryover_seconds,
                    min_carryover_seconds,
                )
            turns.append(
                {
                    "turn": turn_no,
                    "captured_pcm_bytes": len(pcm),
                    "gate_transcript": gate_transcript,
                    "published": False,
                    "interrupted": True,
                    "partial_response_bytes": getattr(publisher, "bytes_published", 0),
                    "publisher_queue_before_interrupt_seconds": round(
                        getattr(publisher, "last_queue_duration_before_interrupt", 0.0) or 0.0,
                        3,
                    ),
                    "publisher_queue_after_clear_seconds": round(
                        getattr(publisher, "last_queue_duration_after_clear", 0.0) or 0.0,
                        3,
                    ),
                    "barge_in_carryover_pcm_bytes": len(raw_carryover_pcm),
                    "barge_in_carryover_discarded": carryover_discarded,
                    "barge_in_diagnostic": {
                        "chunks_seen": barge_in_capture.chunks_seen,
                        "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                        if barge_in_capture.first_chunk_seconds is not None
                        else None,
                        "max_rms": barge_in_capture.max_rms,
                        "voiced_ms": barge_in_capture.voiced_ms,
                        "detected_seconds": round(barge_in_capture.detected_seconds, 3)
                        if barge_in_capture.detected_seconds is not None
                        else None,
                    },
                    "from_barge_in_carryover": from_barge_in_carryover,
                    "timing": {
                        "capture_seconds": round(capture_seconds, 3),
                        "captured_audio_seconds": round(captured_audio_seconds, 3),
                        "gate_seconds": round(gate_seconds, 3),
                        "first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                        "barge_in_seconds": round(barge_in_seconds, 3) if barge_in_seconds is not None else None,
                        "barge_in_captured_audio_seconds": round(barge_in_capture.captured_audio_seconds, 3),
                    },
                }
            )
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                break
            continue
        except Exception as exc:
            if publisher is not None:
                with contextlib.suppress(Exception):
                    if getattr(publisher, "interrupted", False):
                        await publisher.close(wait_for_playout=False, flush_remainder=False)
                    else:
                        await publisher.close()
            error_text = str(exc) or repr(exc)
            logger.warning("xAI response/publish failed for turn %s: %s: %s", turn_no, type(exc).__name__, error_text)
            turns.append(
                {
                    "turn": turn_no,
                    "captured_pcm_bytes": len(pcm),
                    "gate_transcript": gate_transcript,
                    "published": False,
                    "error": f"{type(exc).__name__}: {error_text}",
                    "barge_in_diagnostic": {
                        "chunks_seen": barge_in_capture.chunks_seen,
                        "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                        if barge_in_capture.first_chunk_seconds is not None
                        else None,
                        "max_rms": barge_in_capture.max_rms,
                        "voiced_ms": barge_in_capture.voiced_ms,
                        "detected_seconds": round(barge_in_capture.detected_seconds, 3)
                        if barge_in_capture.detected_seconds is not None
                        else None,
                    },
                    "from_barge_in_carryover": from_barge_in_carryover,
                    "timing": {
                        "capture_seconds": round(capture_seconds, 3),
                        "captured_audio_seconds": round(captured_audio_seconds, 3),
                        "gate_seconds": round(gate_seconds, 3),
                    },
                }
            )
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                break
            continue
        turns.append(
            {
                "turn": turn_no,
                "captured_pcm_bytes": len(pcm),
                "gate_transcript": gate_transcript,
                "xai_response_bytes": xai_result.bytes_written,
                "xai_transcript": xai_result.transcript,
                "xai_events_tail": list(xai_result.events_seen[-5:]),
                "barge_in_diagnostic": {
                    "chunks_seen": barge_in_capture.chunks_seen,
                    "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                    if barge_in_capture.first_chunk_seconds is not None
                    else None,
                    "max_rms": barge_in_capture.max_rms,
                    "voiced_ms": barge_in_capture.voiced_ms,
                    "detected_seconds": round(barge_in_capture.detected_seconds, 3)
                    if barge_in_capture.detected_seconds is not None
                    else None,
                },
                "published": True,
                "from_barge_in_carryover": from_barge_in_carryover,
                "timing": {
                    "capture_seconds": round(capture_seconds, 3),
                    "captured_audio_seconds": round(captured_audio_seconds, 3),
                    "gate_seconds": round(gate_seconds, 3),
                    "xai_seconds": round(xai_seconds, 3),
                    "first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                    "playout_drain_seconds": round(publish_seconds, 3),
                    "after_capture_seconds": round(gate_seconds + xai_seconds + publish_seconds, 3),
                },
            }
        )
        if args.max_turns and len(turns) >= args.max_turns:
            break
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            break
    return {"turns": turns, "turn_count": len(turns)}


async def _diagnose_barge_in(args: argparse.Namespace, bridge: FluxerLiveKitSmokeBridge) -> dict[str, Any]:
    """Probe whether remote user audio is received while a local LiveKit track is active."""

    started = time.monotonic()
    chunks = bridge.iter_remote_audio_pcm16(
        sample_rate=args.sample_rate,
        frame_size_ms=args.frame_ms,
        participant_identity=args.participant_identity,
        participant_identity_prefix=getattr(args, "participant_identity_prefix", None),
    )
    publisher = bridge.pcm16_publisher(
        sample_rate=args.sample_rate,
        frame_ms=args.frame_ms,
        track_name="fluxer-barge-diagnostic-tone",
    )
    frame_samples = max(1, args.sample_rate * args.frame_ms // 1000)
    tone_phase = 0.0
    tone_step = 2.0 * math.pi * args.diagnostic_tone_hz / args.sample_rate

    def tone_frame() -> bytes:
        nonlocal tone_phase
        samples = bytearray()
        for _ in range(frame_samples):
            sample = int(args.diagnostic_tone_amplitude * math.sin(tone_phase))
            samples.extend(sample.to_bytes(2, byteorder="little", signed=True))
            tone_phase = (tone_phase + tone_step) % (2.0 * math.pi)
        return bytes(samples)
    max_rms = 0
    voiced_ms = 0
    chunks_seen = 0
    detected_at: float | None = None
    first_chunk_at: float | None = None

    async def publish_tone() -> None:
        await publisher.__aenter__()
        end_at = time.monotonic() + args.diagnose_seconds
        while time.monotonic() < end_at and detected_at is None:
            await publisher.write(tone_frame())
            await asyncio.sleep(args.frame_ms / 1000)

    publish_task = asyncio.create_task(publish_tone())
    iterator = chunks.__aiter__()
    try:
        while True:
            remaining = args.diagnose_seconds - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(anext(iterator), timeout=min(1.0, remaining))
            except (TimeoutError, asyncio.TimeoutError):
                continue
            except StopAsyncIteration:
                break
            now = time.monotonic()
            if first_chunk_at is None:
                first_chunk_at = now - started
            chunks_seen += 1
            rms = _pcm16_rms(chunk)
            max_rms = max(max_rms, rms)
            if rms >= args.barge_in_energy_threshold:
                voiced_ms += args.frame_ms
            else:
                voiced_ms = 0
            logger.info(
                "barge probe chunk=%s rms=%s max_rms=%s voiced_ms=%s threshold=%s",
                chunks_seen,
                rms,
                max_rms,
                voiced_ms,
                args.barge_in_energy_threshold,
            )
            if voiced_ms >= args.barge_in_min_ms:
                detected_at = now - started
                await publisher.interrupt()
                break
            if now - started >= args.diagnose_seconds:
                break
    finally:
        close_chunks = getattr(chunks, "aclose", None)
        if close_chunks is not None:
            await close_chunks()
        publish_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await publish_task
        if not getattr(publisher, "interrupted", False):
            await publisher.close(wait_for_playout=False, flush_remainder=False)

    return {
        "mode": "barge_in_diagnostic",
        "diagnose_seconds": args.diagnose_seconds,
        "sample_rate": args.sample_rate,
        "frame_ms": args.frame_ms,
        "threshold": args.barge_in_energy_threshold,
        "min_ms": args.barge_in_min_ms,
        "tone_hz": args.diagnostic_tone_hz,
        "tone_amplitude": args.diagnostic_tone_amplitude,
        "chunks_seen": chunks_seen,
        "first_chunk_seconds": round(first_chunk_at, 3) if first_chunk_at is not None else None,
        "max_rms": max_rms,
        "detected": detected_at is not None,
        "detected_seconds": round(detected_at, 3) if detected_at is not None else None,
        "publisher_interrupted": getattr(publisher, "interrupted", False),
        "bytes_published": getattr(publisher, "bytes_published", 0),
    }


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("adapter").setLevel(logging.DEBUG)
        logging.getLogger("livekit_bridge").setLevel(logging.DEBUG)
    for noisy_secret_logger in ("websockets", "websockets.client", "httpcore", "httpx"):
        logging.getLogger(noisy_secret_logger).setLevel(logging.INFO)
    token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not token:
        print("FLUXER_BOT_TOKEN is required", file=sys.stderr)
        return 2

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": token,
                "base_url": os.getenv("FLUXER_BASE_URL", ""),
                "gateway_url": os.getenv("FLUXER_GATEWAY_URL", ""),
                "allow_all_users": env_truthy("FLUXER_ALLOW_ALL_USERS"),
            },
        )
    )
    bridge = FluxerLiveKitSmokeBridge(auto_subscribe=True)
    connected = asyncio.Event()
    result: dict[str, Any] = {"mode": "continuous_turn_loop"}

    async def on_voice_server_update(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        try:
            info = await bridge.connect_from_voice_server_update(raw_update)
            result["safe_update"] = safe_update
            result["connection"] = {
                "endpoint": info.endpoint,
                "guild_id": info.guild_id,
                "channel_id": info.channel_id,
                "connection_id": info.connection_id,
                "room_name": info.room_name,
                "participant_identity": info.participant_identity,
            }
            connected.set()
        except Exception as exc:  # token intentionally not included
            result["error"] = type(exc).__name__
            result["message"] = _redact_exception_message(exc, str(raw_update.get("token") or ""))
            connected.set()

    adapter.set_voice_server_update_handler(on_voice_server_update)
    try:
        if not await adapter.connect():
            print("Fluxer adapter did not connect to gateway", file=sys.stderr)
            return 1
        if not await adapter.wait_until_gateway_ready(timeout=10):
            print("Fluxer gateway did not emit READY before timeout", file=sys.stderr)
            return 1
        sent = await adapter.send_voice_state_update(
            args.channel_id,
            guild_id=args.guild_id,
            self_mute=not args.unmute,
            self_deaf=False,
        )
        if not sent:
            print("VOICE_STATE_UPDATE was not sent; websocket unavailable", file=sys.stderr)
            return 1
        try:
            await asyncio.wait_for(connected.wait(), timeout=args.connect_timeout)
        except (TimeoutError, asyncio.TimeoutError):
            print(
                f"No VOICE_SERVER_UPDATE received within {args.connect_timeout}s; "
                "check channel-id, guild-id, and FLUXER_BOT_TOKEN",
                file=sys.stderr,
            )
            return 1
        if result.get("error"):
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        if args.diagnose_barge_in:
            try:
                diagnostic_result = await asyncio.wait_for(
                    _diagnose_barge_in(args, bridge),
                    timeout=args.diagnose_seconds + 30,
                )
            except (TimeoutError, asyncio.TimeoutError):
                print(
                    f"Barge-in diagnostic exceeded safety timeout of {args.diagnose_seconds + 30}s",
                    file=sys.stderr,
                )
                return 1
            result.update(diagnostic_result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if diagnostic_result.get("detected") else 1
        try:
            loop_result = await asyncio.wait_for(
                _conversation_loop(args, bridge),
                timeout=args.max_runtime_seconds + 30,
            )
        except (TimeoutError, asyncio.TimeoutError):
            print(
                f"Conversation loop exceeded safety timeout of {args.max_runtime_seconds + 30}s",
                file=sys.stderr,
            )
            return 1
        result.update(loop_result)
        result["published_turn_count"] = sum(1 for turn in result.get("turns", []) if turn.get("published"))
        result["ignored_turn_count"] = sum(1 for turn in result.get("turns", []) if turn.get("ignored"))
        result["interrupted_turn_count"] = sum(1 for turn in result.get("turns", []) if turn.get("interrupted"))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("published_turn_count", 0) > 0 or result.get("interrupted_turn_count", 0) > 0 else 1
    finally:
        try:
            await adapter.send_voice_state_update(None, guild_id=args.guild_id)
        except Exception:
            pass
        await bridge.disconnect()
        await adapter.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous Fluxer voice ↔ xAI Realtime room loop")
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--guild-id")
    parser.add_argument("--participant-identity", help="Only capture this exact remote LiveKit participant identity")
    parser.add_argument("--participant-identity-prefix", help="Only capture remote LiveKit participants whose identity starts with this prefix")
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=120.0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--energy-threshold", type=int, default=350)
    parser.add_argument("--silence-ms", type=int, default=600)
    parser.add_argument("--end-padding-ms", type=int, default=180, help="Final silence kept in captured PCM after end-of-turn detection")
    parser.add_argument("--min-segment-ms", type=int, default=750)
    parser.add_argument("--max-segment-seconds", type=float, default=8.0)
    parser.add_argument("--unmute", action="store_true", help="Join unmuted; default muted")
    parser.add_argument("--xai-model", default="grok-voice-latest")
    parser.add_argument("--xai-voice", default="eve")
    parser.add_argument("--xai-timeout", type=float, default=45.0)
    parser.add_argument("--xai-first-audio-timeout", type=float, default=8.0, help="Abort a response turn if xAI emits no audio delta by this many seconds")
    parser.add_argument("--xai-instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--wake-gate-instructions", default=WAKE_GATE_INSTRUCTIONS)
    parser.add_argument("--disable-wake-gate", action="store_true", help="Answer every captured speech segment")
    parser.add_argument("--disable-barge-in", action="store_true", help="Do not monitor for user interruption while assistant audio is streaming")
    parser.add_argument("--barge-in-after-first-audio-only", action="store_true", help="Test mode: arm barge-in only after first assistant audio so the initial user utterance tail cannot cancel before playback")
    parser.add_argument("--barge-in-energy-threshold", type=int, default=700)
    parser.add_argument("--barge-in-min-ms", type=int, default=240)
    parser.add_argument("--barge-in-capture-timeout", type=float, default=2.0, help="How long to wait for the interrupting utterance to finish so it can become the next prompt")
    parser.add_argument("--diagnose-barge-in", action="store_true", help="Run an audible LiveKit-only barge-in receive/publish diagnostic instead of xAI conversation")
    parser.add_argument("--diagnose-seconds", type=float, default=20.0, help="Seconds to run --diagnose-barge-in")
    parser.add_argument("--diagnostic-tone-hz", type=float, default=440.0, help="Tone frequency for --diagnose-barge-in audible probe")
    parser.add_argument("--diagnostic-tone-amplitude", type=int, default=1800, help="PCM16 amplitude for --diagnose-barge-in audible probe")
    parser.add_argument("--verbose", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
