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
import json
import logging
import os
import sys
import tempfile
import time
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

DEFAULT_INSTRUCTIONS = """You are Žofka speaking with Elkim in a Fluxer voice room.
Always answer in English unless Elkim explicitly asks for Czech. Do not answer in Spanish.
Treat Žofka, Zofka, Jefka, and occasional voice-ASR name confusions like Jessica as your name; do not correct the name out loud.
Ignore background music, lyrics, radio, TV, and room noise. Respond only to speech that sounds directed at Žofka or clearly part of the conversation.
Be warm, direct, concise, and natural for realtime voice. One or two short sentences unless Elkim asks for depth.
""".strip()

WAKE_GATE_INSTRUCTIONS = """You are a strict realtime voice gate for Žofka in a noisy room.
Listen to the user's audio. If the speech does not clearly address Žofka/Zofka/Jefka or an obvious ASR confusion of that name, or if it sounds like music, lyrics, radio, TV, or background noise, reply with exactly: IGNORE
If the user clearly addresses Žofka/Zofka/Jefka or an obvious ASR confusion of that name, reply with exactly: RESPOND
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


async def _speech_segments(
    chunks: AsyncIterator[bytes],
    *,
    sample_rate: int,
    energy_threshold: int,
    silence_ms: int,
    min_segment_ms: int,
    max_segment_seconds: float,
) -> AsyncIterator[bytes]:
    """Yield PCM16 speech-ish segments from a chunk iterator using simple VAD."""

    bytes_per_ms = sample_rate * 2 / 1000
    silence_bytes_limit = int(bytes_per_ms * silence_ms)
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
            pcm = bytes(segment)
            segment.clear()
            trailing_silence = 0
            in_segment = False
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
    )
    segments = _speech_segments(
        chunks,
        sample_rate=args.sample_rate,
        energy_threshold=args.energy_threshold,
        silence_ms=args.silence_ms,
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
    while True:
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            break
        remaining = args.max_runtime_seconds - (time.monotonic() - started) if args.max_runtime_seconds else 30.0
        try:
            capture_started = time.monotonic()
            pcm = await _capture_one_speech_segment(args, bridge, timeout=max(1.0, remaining))
            capture_seconds = time.monotonic() - capture_started
        except TimeoutError:
            break
        turn_no = len(turns) + 1
        logger.info("Captured speech turn %s bytes=%s rms=%s", turn_no, len(pcm), _pcm16_rms(pcm))
        gate_transcript = ""
        gate_seconds = 0.0
        if not args.disable_wake_gate:
            gate_wav = str(Path(tempfile.gettempdir()) / f"zofka_xai_room_loop_gate_{turn_no}.wav")
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
                        "timing": {
                            "capture_seconds": round(capture_seconds, 3),
                            "gate_seconds": round(gate_seconds, 3),
                        },
                    }
                )
                if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                    break
                continue

        output_wav = str(Path(tempfile.gettempdir()) / f"zofka_xai_room_loop_turn_{turn_no}.wav")
        try:
            xai_started = time.monotonic()
            xai_result = await xai.audio_response_from_pcm16(pcm, output_wav, timeout=args.xai_timeout)
            xai_seconds = time.monotonic() - xai_started
            publish_started = time.monotonic()
            await bridge.publish_wav_file(output_wav, track_name=f"zofka-xai-room-loop-{turn_no}")
            publish_seconds = time.monotonic() - publish_started
        except Exception as exc:
            logger.warning("xAI response/publish failed for turn %s: %s", turn_no, type(exc).__name__)
            turns.append(
                {
                    "turn": turn_no,
                    "captured_pcm_bytes": len(pcm),
                    "gate_transcript": gate_transcript,
                    "published": False,
                    "error": f"{type(exc).__name__}: response failed",
                    "timing": {
                        "capture_seconds": round(capture_seconds, 3),
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
                "published": True,
                "timing": {
                    "capture_seconds": round(capture_seconds, 3),
                    "gate_seconds": round(gate_seconds, 3),
                    "xai_seconds": round(xai_seconds, 3),
                    "publish_seconds": round(publish_seconds, 3),
                    "after_capture_seconds": round(gate_seconds + xai_seconds + publish_seconds, 3),
                },
            }
        )
        if args.max_turns and sum(1 for turn in turns if turn.get("published")) >= args.max_turns:
            break
        if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
            break
    return {"turns": turns, "turn_count": len(turns)}


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
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
                "allow_all_users": True,
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
            result["message"] = str(exc)
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
        await asyncio.wait_for(connected.wait(), timeout=args.connect_timeout)
        if result.get("error"):
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        loop_result = await asyncio.wait_for(_conversation_loop(args, bridge), timeout=args.max_runtime_seconds + 30)
        result.update(loop_result)
        result["published_turn_count"] = sum(1 for turn in result.get("turns", []) if turn.get("published"))
        result["ignored_turn_count"] = sum(1 for turn in result.get("turns", []) if turn.get("ignored"))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("published_turn_count", 0) > 0 else 1
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
    parser.add_argument("--participant-identity", help="Only capture this remote LiveKit participant identity")
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=120.0)
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--energy-threshold", type=int, default=350)
    parser.add_argument("--silence-ms", type=int, default=900)
    parser.add_argument("--min-segment-ms", type=int, default=600)
    parser.add_argument("--max-segment-seconds", type=float, default=8.0)
    parser.add_argument("--unmute", action="store_true", help="Join unmuted; default muted")
    parser.add_argument("--xai-model", default="grok-voice-latest")
    parser.add_argument("--xai-voice", default="eve")
    parser.add_argument("--xai-timeout", type=float, default=45.0)
    parser.add_argument("--xai-instructions", default=DEFAULT_INSTRUCTIONS)
    parser.add_argument("--wake-gate-instructions", default=WAKE_GATE_INSTRUCTIONS)
    parser.add_argument("--disable-wake-gate", action="store_true", help="Answer every captured speech segment")
    parser.add_argument("--verbose", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
