#!/usr/bin/env python3
"""STT-backed Fluxer voice room loop.

This is the "actually listen" prototype:
1. join a Fluxer voice channel,
2. capture audio only from a targeted LiveKit participant/user prefix,
3. transcribe that PCM with Hermes STT,
4. ask xAI Realtime to speak a text-grounded answer,
5. publish the resulting WAV back into Fluxer LiveKit.

It intentionally avoids xAI's direct audio-understanding path because live tests
showed it produced generic filler ("hey, ready to chat?") even when transport
was working.
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import contextlib
import json
import logging
import os
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(os.getenv("HERMES_AGENT_ROOT", "/home/elkim/.hermes/hermes-agent"))
for candidate in (ROOT, HERMES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from livekit_bridge import FluxerLiveKitSmokeBridge  # noqa: E402
from scripts.fluxer_xai_room_loop import _capture_one_speech_segment  # noqa: E402
from tools.transcription_tools import (  # noqa: E402
    _transcribe_elevenlabs,
    _transcribe_groq,
    _transcribe_xai,
    transcribe_audio,
)
from xai_realtime import XAIRealtimeVoiceClient  # noqa: E402

logger = logging.getLogger("fluxer_stt_voice_loop")

DEFAULT_TEXT_SYSTEM = """You are Žofka in a live Fluxer voice chat with Elkim.
Answer the transcript directly and briefly. No filler greetings unless Elkim greeted you.
If STT writes Shevka, Shovka, Jefka, Zofka, Jovka, or Jessica, treat it as Žofka.
Correct obvious ASR homophones when context is clear, e.g. "past" or "plast" can mean "plus" in arithmetic.
Speak English by default. Do not switch to Czech just because STT produced Czech-looking syllables; use Czech only if Elkim explicitly asks for Czech or clearly speaks Czech.
Never use Spanish.
""".strip()


def load_env_file(path: Path) -> None:
    """Load simple KEY=value lines without shell-sourcing secrets."""

    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def write_pcm16_wav(path: Path, pcm: bytes, *, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return path


def build_answer_prompt(transcript: str, *, history: list[dict[str, str]], system: str = DEFAULT_TEXT_SYSTEM) -> str:
    """Build a compact text prompt for a voice answer grounded in STT text."""

    transcript = transcript.strip()
    history_lines: list[str] = []
    for item in history[-6:]:
        user = (item.get("user") or "").strip()
        assistant = (item.get("assistant") or "").strip()
        if user:
            history_lines.append(f"Elkim: {user}")
        if assistant:
            history_lines.append(f"Žofka: {assistant}")
    history_text = "\n".join(history_lines) or "(none)"
    return (
        f"{system}\n\n"
        f"Recent voice-chat history:\n{history_text}\n\n"
        f"Latest STT transcript from Elkim: {transcript!r}\n\n"
        "Speak Žofka's next reply now, grounded only in the latest transcript and relevant history."
    )


def transcribe_with_provider(file_path: str, *, provider: str, model: str | None) -> dict[str, Any]:
    """Transcribe with an explicit provider for this spike loop."""

    if provider == "auto":
        return transcribe_audio(file_path, model=model)
    if provider == "local":
        return transcribe_audio(file_path, model=model)
    if provider == "groq":
        groq_model = model if model and model not in {"tiny.en", "base.en", "small.en", "medium.en"} else "whisper-large-v3-turbo"
        return _transcribe_groq(file_path, groq_model)
    if provider == "xai":
        return _transcribe_xai(file_path, model or "grok-stt")
    if provider == "elevenlabs":
        elevenlabs_model = model if model and model not in {"tiny.en", "base.en", "small.en", "medium.en"} else "scribe_v2"
        return _transcribe_elevenlabs(file_path, elevenlabs_model)
    raise ValueError(f"Unsupported STT provider: {provider}")


def safe_stt_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {key: result.get(key) for key in ("success", "transcript", "provider", "model", "error")}


async def run_stt_voice_loop(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(Path(args.env_file).expanduser())
    bot_token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("FLUXER_BOT_TOKEN is required")
    if not os.getenv("XAI_API_KEY", "").strip():
        raise RuntimeError("XAI_API_KEY is required")

    adapter = FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": bot_token, "allow_all_users": True})
    )
    bridge = FluxerLiveKitSmokeBridge(auto_subscribe=True)
    connected = asyncio.Event()
    finished = asyncio.Event()
    result: dict[str, Any] = {
        "mode": "stt_backed_voice_loop",
        "turn_count": 0,
        "published_turn_count": 0,
        "ignored_turn_count": 0,
        "turns": [],
    }
    history: list[dict[str, str]] = []

    async def handler(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        try:
            info = await bridge.connect_from_voice_server_update(raw_update)
            result["connection"] = {
                "endpoint": info.endpoint,
                "guild_id": info.guild_id,
                "channel_id": info.channel_id,
                "connection_id": info.connection_id,
                "room_name": info.room_name,
                "participant_identity": info.participant_identity,
            }
            result["safe_update"] = safe_update
            connected.set()

            # Let Fluxer publish/subscription state settle before the first fixed window.
            await asyncio.sleep(args.initial_settle_seconds)

            for turn_no in range(1, args.max_turns + 1):
                turn_started = time.monotonic()
                if args.capture_mode == "vad":
                    pcm = await _capture_one_speech_segment(args, bridge, timeout=args.capture_timeout)
                else:
                    pcm = await bridge.collect_remote_audio_pcm16(
                        duration_seconds=args.capture_window_seconds,
                        sample_rate=args.sample_rate,
                        frame_size_ms=args.frame_ms,
                        participant_identity=args.participant_identity,
                        participant_identity_prefix=args.participant_identity_prefix,
                        timeout=args.capture_timeout,
                    )
                wav_path = Path(tempfile.gettempdir()) / f"zofka_stt_loop_input_{turn_no}.wav"
                write_pcm16_wav(wav_path, pcm, sample_rate=args.sample_rate)
                stt_started = time.monotonic()
                stt_result = transcribe_with_provider(
                    str(wav_path),
                    provider=args.stt_provider,
                    model=args.stt_model,
                )
                stt_seconds = time.monotonic() - stt_started
                transcript = (stt_result.get("transcript") or "").strip()
                turn: dict[str, Any] = {
                    "turn": turn_no,
                    "captured_pcm_bytes": len(pcm),
                    "captured_audio_seconds": round(len(pcm) / 2 / args.sample_rate, 3),
                    "input_rms": audioop.rms(pcm, 2) if pcm else 0,
                    "stt": safe_stt_summary(stt_result),
                    "stt_seconds": round(stt_seconds, 3),
                }

                if not transcript:
                    result["ignored_turn_count"] += 1
                    turn["published"] = False
                    turn["reason"] = "empty_stt_transcript"
                    result["turns"].append(turn)
                    if args.stop_on_empty_stt:
                        break
                    continue

                prompt = build_answer_prompt(transcript, history=history)
                voice = XAIRealtimeVoiceClient(
                    sample_rate=args.sample_rate,
                    voice=args.voice,
                    instructions="Speak Žofka's answer naturally, briefly, and without extra preamble.",
                )
                xai_started = time.monotonic()
                first_audio_seconds: float | None = None
                publisher = bridge.pcm16_publisher(
                    sample_rate=args.sample_rate,
                    frame_ms=args.frame_ms,
                    track_name=f"zofka-stt-loop-reply-{turn_no}",
                )
                await publisher.__aenter__()

                async def publish_delta(chunk: bytes) -> None:
                    nonlocal first_audio_seconds
                    if first_audio_seconds is None:
                        first_audio_seconds = time.monotonic() - xai_started
                    await publisher.write(chunk)

                try:
                    xai_result = await voice.text_response_to_sink(
                        prompt,
                        publish_delta,
                        timeout=args.xai_timeout,
                        first_audio_timeout=args.xai_first_audio_timeout,
                    )
                finally:
                    await publisher.__aexit__(None, None, None)
                xai_seconds = time.monotonic() - xai_started
                history.append({"user": transcript, "assistant": xai_result.transcript})
                turn.update(
                    {
                        "published": True,
                        "reply_transcript": xai_result.transcript,
                        "reply_bytes": xai_result.bytes_written,
                        "xai_events_tail": list(xai_result.events_seen[-5:]),
                        "timing": {
                            "turn_seconds": round(time.monotonic() - turn_started, 3),
                            "stt_seconds": round(stt_seconds, 3),
                            "xai_first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                            "xai_seconds": round(xai_seconds, 3),
                        },
                    }
                )
                result["published_turn_count"] += 1
                result["turns"].append(turn)

            result["turn_count"] = len(result["turns"])
        except Exception as exc:
            logger.exception("STT-backed Fluxer voice loop failed")
            result["error"] = type(exc).__name__
            result["message"] = str(exc)
        finally:
            finished.set()

    adapter.set_voice_server_update_handler(handler)
    await adapter.connect()
    try:
        await adapter.wait_until_gateway_ready(timeout=args.connect_timeout)
        await adapter.send_voice_state_update(args.channel_id, guild_id=args.guild_id, self_mute=True, self_deaf=False)
        await asyncio.wait_for(connected.wait(), timeout=args.connect_timeout)
        await asyncio.wait_for(finished.wait(), timeout=args.max_runtime_seconds)
    finally:
        with contextlib.suppress(Exception):
            await adapter.send_voice_state_update(None, guild_id=args.guild_id)
        await bridge.disconnect()
        await adapter.disconnect()
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT-backed Fluxer voice room loop")
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--guild-id")
    parser.add_argument("--participant-identity", help="Only capture this exact remote LiveKit participant identity")
    parser.add_argument(
        "--participant-identity-prefix",
        help="Only capture remote LiveKit participants whose identity starts with this prefix, e.g. user_<FluxerUserId>_",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--capture-mode", choices=("vad", "fixed"), default="fixed")
    parser.add_argument("--capture-window-seconds", type=float, default=3.0)
    parser.add_argument("--capture-timeout", type=float, default=25.0)
    parser.add_argument("--initial-settle-seconds", type=float, default=0.8)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--energy-threshold", type=int, default=550)
    parser.add_argument("--silence-ms", type=int, default=500)
    parser.add_argument("--end-padding-ms", type=int, default=120)
    parser.add_argument("--min-segment-ms", type=int, default=500)
    parser.add_argument("--max-segment-seconds", type=float, default=6.0)
    parser.add_argument("--voice", default="eve")
    parser.add_argument("--stt-provider", choices=("auto", "local", "groq", "xai", "elevenlabs"), default="local")
    parser.add_argument("--stt-model", default="medium.en", help="STT model; local default medium.en for accuracy, Groq default whisper-large-v3-turbo, ElevenLabs default scribe_v2")
    parser.add_argument("--xai-timeout", type=float, default=45.0)
    parser.add_argument("--xai-first-audio-timeout", type=float, default=12.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=180.0)
    parser.add_argument("--env-file", default="/home/elkim/.hermes/.env")
    parser.add_argument("--stop-on-empty-stt", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    result = asyncio.run(run_stt_voice_loop(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
