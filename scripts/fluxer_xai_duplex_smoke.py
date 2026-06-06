#!/usr/bin/env python3
"""First Fluxer ↔ xAI Realtime duplex smoke probe.

This joins a Fluxer voice channel, collects a short PCM16 mono slice from a
remote participant's subscribed LiveKit audio track, sends that audio to xAI
Realtime, writes the returned Grok Voice audio to WAV, then publishes it back to
Fluxer LiveKit.

It is still a smoke probe, not the final daemon: one captured turn in, one
assistant audio turn out.
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


def _safe_room_snapshot(room: Any) -> dict[str, Any]:
    """Return non-secret LiveKit room diagnostics for smoke debugging."""
    participants = []
    for participant in getattr(room, "remote_participants", {}).values():
        publications = []
        for publication in getattr(participant, "track_publications", {}).values():
            track = getattr(publication, "track", None)
            publications.append(
                {
                    "sid": getattr(publication, "sid", None),
                    "kind": str(getattr(publication, "kind", None)),
                    "source": str(getattr(publication, "source", None)),
                    "subscribed": bool(getattr(publication, "subscribed", False)),
                    "muted": bool(getattr(publication, "muted", False)),
                    "has_track": track is not None,
                    "track_kind": str(getattr(track, "kind", None)) if track is not None else None,
                    "track_class": track.__class__.__name__ if track is not None else None,
                }
            )
        participants.append(
            {
                "identity": getattr(participant, "identity", None),
                "sid": getattr(participant, "sid", None),
                "publications": publications,
            }
        )
    return {
        "room_name": getattr(room, "name", None),
        "remote_participant_count": len(participants),
        "remote_participants": participants,
    }

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from livekit_bridge import FluxerLiveKitSmokeBridge  # noqa: E402
from xai_realtime import XAIRealtimeVoiceClient  # noqa: E402


async def run(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
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
    result: dict[str, Any] = {}

    async def on_voice_server_update(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        try:
            info = await bridge.connect_from_voice_server_update(raw_update)
            result["room_before_capture"] = _safe_room_snapshot(bridge._room)
            pcm = await bridge.collect_remote_audio_pcm16(
                duration_seconds=args.listen_seconds,
                sample_rate=args.sample_rate,
                frame_size_ms=args.frame_ms,
                participant_identity=args.participant_identity,
                timeout=args.listen_timeout,
            )
            output_wav = str(Path(tempfile.gettempdir()) / "fluxer_xai_duplex_response.wav")
            xai = XAIRealtimeVoiceClient(
                model=args.xai_model,
                voice=args.xai_voice,
                sample_rate=args.sample_rate,
                instructions=args.xai_instructions,
            )
            xai_result = await xai.audio_response_from_pcm16(pcm, output_wav, timeout=args.xai_timeout)
            await bridge.publish_wav_file(output_wav, track_name="fluxer-xai-duplex-response")
            result["safe_update"] = safe_update
            result["connection"] = {
                "endpoint": info.endpoint,
                "guild_id": info.guild_id,
                "channel_id": info.channel_id,
                "connection_id": info.connection_id,
                "room_name": info.room_name,
                "participant_identity": info.participant_identity,
                "captured_pcm_bytes": len(pcm),
                "xai_response_bytes": xai_result.bytes_written,
                "xai_events_tail": list(xai_result.events_seen[-5:]),
                "duplex_turn_published": True,
            }
            connected.set()
        except Exception as exc:  # token intentionally not included
            result["error"] = type(exc).__name__
            result["message"] = str(exc)
            if bridge._room is not None:
                result["room_at_error"] = _safe_room_snapshot(bridge._room)
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
        await asyncio.wait_for(connected.wait(), timeout=args.timeout)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("connection", {}).get("duplex_turn_published") else 1
    finally:
        try:
            await adapter.send_voice_state_update(None, guild_id=args.guild_id)
        except Exception:
            pass
        await bridge.disconnect()
        await adapter.disconnect()


def main() -> int:
    parser = argparse.ArgumentParser(description="Fluxer voice → xAI Realtime → Fluxer voice one-turn smoke")
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--guild-id")
    parser.add_argument("--participant-identity", help="Only capture this remote LiveKit participant identity")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--listen-timeout", type=float, default=45.0)
    parser.add_argument("--listen-seconds", type=float, default=3.0)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--frame-ms", type=int, default=20)
    parser.add_argument("--unmute", action="store_true", help="Join unmuted; default muted")
    parser.add_argument("--xai-model", default="grok-voice-latest")
    parser.add_argument("--xai-voice", default="eve")
    parser.add_argument("--xai-timeout", type=float, default=45.0)
    parser.add_argument("--xai-instructions", default="You are the assistant. Reply warmly and directly in one short sentence.")
    parser.add_argument("--verbose", action="store_true")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
