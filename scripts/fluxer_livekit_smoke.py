#!/usr/bin/env python3
"""Join a Fluxer voice channel and connect to its LiveKit room once.

Usage:
  FLUXER_BOT_TOKEN=... python scripts/fluxer_livekit_smoke.py --channel-id <voice-channel-id> [--guild-id <guild-id>]

This script is intentionally a smoke probe, not the realtime the assistant loop. It:
1. connects the standalone Fluxer adapter gateway,
2. sends opcode-4 VOICE_STATE_UPDATE for the requested channel,
3. waits for VOICE_SERVER_UPDATE,
4. connects the LiveKit SDK using the ephemeral token,
5. prints only non-secret metadata, then leaves/disconnects.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from livekit_bridge import FluxerLiveKitSmokeBridge  # noqa: E402
from xai_realtime import XAIRealtimeVoiceClient  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
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
    generated_wav_path: str | None = None
    if args.xai_text:
        generated_wav_path = str(Path(tempfile.gettempdir()) / "fluxer_xai_realtime_fluxer.wav")
        xai = XAIRealtimeVoiceClient(model=args.xai_model, voice=args.xai_voice, instructions=args.xai_instructions)
        if args.xai_force_message:
            await xai.force_message_to_wav(args.xai_text, generated_wav_path, timeout=args.xai_timeout)
        else:
            await xai.text_response_to_wav(args.xai_text, generated_wav_path, timeout=args.xai_timeout)
    bridge = FluxerLiveKitSmokeBridge(auto_subscribe=args.auto_subscribe)
    connected = asyncio.Event()
    result: dict[str, Any] = {}

    async def handle_voice_server_update(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        try:
            info = await bridge.connect_from_voice_server_update(raw_update)
            if args.pre_publish_delay > 0:
                await asyncio.sleep(args.pre_publish_delay)
            if args.tone_seconds > 0:
                await bridge.publish_test_tone(
                    duration_seconds=args.tone_seconds,
                    frequency_hz=args.tone_hz,
                    amplitude=args.tone_amplitude,
                )
            if args.wav_path:
                await bridge.publish_wav_file(args.wav_path)
            if generated_wav_path:
                await bridge.publish_wav_file(generated_wav_path, track_name="fluxer-xai-realtime")
            if args.post_publish_hold > 0:
                await asyncio.sleep(args.post_publish_hold)
            result["safe_update"] = safe_update
            result["connection"] = {
                "endpoint": info.endpoint,
                "guild_id": info.guild_id,
                "channel_id": info.channel_id,
                "connection_id": info.connection_id,
                "room_name": info.room_name,
                "participant_identity": info.participant_identity,
                "tone_published": args.tone_seconds > 0,
                "wav_published": bool(args.wav_path),
                "xai_realtime_published": bool(generated_wav_path),
            }
            connected.set()
        except Exception as exc:  # token intentionally not included
            result["error"] = f"{type(exc).__name__}: {exc}"
            connected.set()

    adapter.set_voice_server_update_handler(handle_voice_server_update)
    try:
        if not await adapter.connect():
            print("Fluxer adapter did not connect to gateway", file=sys.stderr)
            return 1
        if not await adapter.wait_until_gateway_ready(timeout=min(args.timeout, 10.0)):
            print("Fluxer gateway connected but did not reach READY before voice join", file=sys.stderr)
            return 1
        sent = await adapter.send_voice_state_update(
            args.channel_id,
            guild_id=args.guild_id,
            self_mute=not args.unmute,
            self_deaf=not args.listen,
        )
        if not sent:
            print("Failed to send Fluxer VOICE_STATE_UPDATE", file=sys.stderr)
            return 1
        try:
            await asyncio.wait_for(connected.wait(), timeout=args.timeout)
        except asyncio.TimeoutError:
            print(f"Timed out waiting {args.timeout:.1f}s for VOICE_SERVER_UPDATE/LiveKit connect", file=sys.stderr)
            return 1
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if "error" not in result else 1
    finally:
        try:
            await adapter.send_voice_state_update(None, guild_id=args.guild_id)
        except Exception:
            pass
        await bridge.disconnect()
        await adapter.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fluxer LiveKit presence-only smoke probe")
    parser.add_argument("--channel-id", required=True, help="Fluxer voice channel id to join")
    parser.add_argument("--guild-id", default=None, help="Fluxer guild/server id, omit for DM/group call")
    parser.add_argument("--timeout", type=float, default=20.0, help="Seconds to wait for VOICE_SERVER_UPDATE")
    parser.add_argument("--unmute", action="store_true", help="Join unmuted; default is muted for smoke safety")
    parser.add_argument("--listen", action="store_true", help="Join undeafened/listening; default is deaf for smoke safety")
    parser.add_argument("--auto-subscribe", action="store_true", help="Ask LiveKit SDK to auto-subscribe to room tracks")
    parser.add_argument("--pre-publish-delay", type=float, default=0.0, help="Seconds to remain in the room before publishing test audio")
    parser.add_argument("--post-publish-hold", type=float, default=0.0, help="Seconds to remain in the room after publishing test audio")
    parser.add_argument("--tone-seconds", type=float, default=0.0, help="Publish a short sine tone after joining; 0 disables audio publishing")
    parser.add_argument("--tone-hz", type=float, default=440.0, help="Sine tone frequency for --tone-seconds")
    parser.add_argument("--tone-amplitude", type=float, default=0.18, help="Sine tone amplitude, 0.0-1.0")
    parser.add_argument("--wav-path", help="Publish a mono 16-bit PCM WAV clip after joining")
    parser.add_argument("--xai-text", help="Ask xAI Realtime for a voice response to this text, then publish it")
    parser.add_argument("--xai-force-message", action="store_true", help="Use xAI Realtime force_message instead of model response")
    parser.add_argument("--xai-model", default="grok-voice-latest")
    parser.add_argument("--xai-voice", default="eve")
    parser.add_argument("--xai-timeout", type=float, default=30.0)
    parser.add_argument("--xai-instructions", default="You are the assistant, warm, direct, and concise. Keep this voice reply short.")
    parser.add_argument("--verbose", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
