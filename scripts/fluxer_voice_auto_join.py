#!/usr/bin/env python3
"""Auto-join Fluxer voice when a target user joins a voice channel.

This is a local supervisor for the realtime-voice spike. It listens to Fluxer
Gateway VOICE_STATE_UPDATE events, starts `fluxer_stt_voice_loop.py` when a
configured user joins an allowed voice channel, and stops the loop when that
user leaves.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(os.getenv("HERMES_AGENT_ROOT", "/home/elkim/.hermes/hermes-agent"))
for candidate in (ROOT, HERMES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from scripts.fluxer_stt_voice_loop import load_env_file  # noqa: E402

logger = logging.getLogger("fluxer_voice_auto_join")


def split_csv(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def voice_state_user_id(data: dict[str, Any]) -> str:
    user_id = data.get("user_id") or data.get("userId")
    if user_id:
        return str(user_id)
    member_raw = data.get("member")
    member = member_raw if isinstance(member_raw, dict) else {}
    user_raw = member.get("user")
    user = user_raw if isinstance(user_raw, dict) else {}
    if user.get("id"):
        return str(user["id"])
    return ""


def voice_state_channel_id(data: dict[str, Any]) -> str | None:
    value = data.get("channel_id", data.get("channelId"))
    return str(value) if value not in (None, "") else None


def voice_state_guild_id(data: dict[str, Any]) -> str | None:
    value = data.get("guild_id", data.get("guildId"))
    return str(value) if value not in (None, "") else None


class FluxerVoiceAutoJoinSupervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.target_user_ids = split_csv(args.target_user_ids)
        self.allowed_channel_ids = split_csv(args.channel_ids)
        self.allowed_guild_ids = split_csv(args.guild_ids)
        self.process: asyncio.subprocess.Process | None = None
        self.active_channel_id: str | None = None
        self.active_guild_id: str | None = None
        self.last_start_monotonic = 0.0

    def should_watch_user(self, user_id: str) -> bool:
        return bool(user_id and (not self.target_user_ids or user_id in self.target_user_ids))

    def should_join_channel(self, *, guild_id: str | None, channel_id: str | None) -> bool:
        if not channel_id:
            return False
        if self.allowed_channel_ids and channel_id not in self.allowed_channel_ids:
            return False
        if self.allowed_guild_ids and (guild_id or "") not in self.allowed_guild_ids:
            return False
        return True

    def build_voice_loop_command(self, *, guild_id: str | None, channel_id: str) -> list[str]:
        python = self.args.python
        script = str(ROOT / "scripts" / "fluxer_stt_voice_loop.py")
        target = next(iter(self.target_user_ids), "")
        participant_prefix = self.args.participant_identity_prefix or (f"user_{target}_" if target else "")
        cmd = [
            python,
            script,
            "--channel-id",
            channel_id,
            "--brain-provider",
            self.args.brain_provider,
            "--stt-provider",
            self.args.stt_provider,
            "--stt-model",
            self.args.stt_model,
            "--voice",
            self.args.voice,
            "--max-turns",
            str(self.args.max_turns),
            "--capture-timeout",
            str(self.args.capture_timeout),
            "--silence-ms",
            str(self.args.silence_ms),
            "--end-padding-ms",
            str(self.args.end_padding_ms),
            "--min-segment-ms",
            str(self.args.min_segment_ms),
            "--max-segment-seconds",
            str(self.args.max_segment_seconds),
            "--max-runtime-seconds",
            str(self.args.max_runtime_seconds),
            "--turn-log-jsonl",
            self.args.turn_log_jsonl,
        ]
        if guild_id:
            cmd.extend(["--guild-id", guild_id])
        if participant_prefix:
            cmd.extend(["--participant-identity-prefix", participant_prefix])
        if self.args.elevenlabs_language_code is not None:
            cmd.extend(["--elevenlabs-language-code", self.args.elevenlabs_language_code])
        return cmd

    async def start_voice_loop(self, *, guild_id: str | None, channel_id: str) -> None:
        if self.process and self.process.returncode is None:
            logger.info("voice loop already running for channel=%s", self.active_channel_id)
            return
        now = asyncio.get_running_loop().time()
        if now - self.last_start_monotonic < self.args.start_cooldown_seconds:
            logger.info("voice loop start suppressed by cooldown")
            return
        self.last_start_monotonic = now
        self.active_channel_id = channel_id
        self.active_guild_id = guild_id
        cmd = self.build_voice_loop_command(guild_id=guild_id, channel_id=channel_id)
        logger.info("starting voice loop for guild=%s channel=%s", guild_id or "<none>", channel_id)
        self.process = await asyncio.create_subprocess_exec(*cmd, cwd=str(ROOT))
        asyncio.create_task(self._watch_process(self.process), name="fluxer-auto-join-voice-loop-watch")

    async def stop_voice_loop(self, reason: str) -> None:
        proc = self.process
        if not proc or proc.returncode is not None:
            return
        logger.info("stopping voice loop: %s", reason)
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.args.stop_timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning("voice loop did not exit after terminate; killing")
            proc.kill()
            await proc.wait()

    async def _watch_process(self, proc: asyncio.subprocess.Process) -> None:
        code = await proc.wait()
        if self.process is proc:
            logger.info("voice loop exited code=%s", code)
            self.process = None
            self.active_channel_id = None
            self.active_guild_id = None

    async def handle_voice_state_update(self, data: dict[str, Any]) -> None:
        user_id = voice_state_user_id(data)
        if not self.should_watch_user(user_id):
            return
        channel_id = voice_state_channel_id(data)
        guild_id = voice_state_guild_id(data)
        if channel_id is None:
            await self.stop_voice_loop(f"target user {user_id} left voice")
            return
        if not self.should_join_channel(guild_id=guild_id, channel_id=channel_id):
            return
        await self.start_voice_loop(guild_id=guild_id, channel_id=channel_id)


async def run(args: argparse.Namespace) -> int:
    load_env_file(Path(args.env_file).expanduser())
    bot_token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("FLUXER_BOT_TOKEN is required")
    supervisor = FluxerVoiceAutoJoinSupervisor(args)
    adapter = FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": bot_token, "allow_all_users": True}))
    adapter.set_voice_state_update_handler(supervisor.handle_voice_state_update)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)
    if not await adapter.connect():
        raise RuntimeError("Fluxer adapter did not connect to gateway")
    try:
        await adapter.wait_until_gateway_ready(timeout=args.connect_timeout)
        logger.info(
            "auto-join supervisor armed target_users=%s channels=%s guilds=%s",
            sorted(supervisor.target_user_ids) or ["<any>"],
            sorted(supervisor.allowed_channel_ids) or ["<any>"],
            sorted(supervisor.allowed_guild_ids) or ["<any>"],
        )
        await stop_event.wait()
    finally:
        await supervisor.stop_voice_loop("supervisor shutting down")
        await adapter.disconnect()
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-join Fluxer voice when target user joins")
    parser.add_argument("--target-user-ids", default=os.getenv("FLUXER_AUTO_JOIN_USER_IDS", "1503635769218148907"))
    parser.add_argument("--channel-ids", default=os.getenv("FLUXER_AUTO_JOIN_CHANNEL_IDS", "1510905670319210500"))
    parser.add_argument("--guild-ids", default=os.getenv("FLUXER_AUTO_JOIN_GUILD_IDS", "1510905670319210496"))
    parser.add_argument("--participant-identity-prefix", default=os.getenv("FLUXER_AUTO_JOIN_PARTICIPANT_PREFIX", ""))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--env-file", default="/home/elkim/.hermes/.env")
    parser.add_argument("--brain-provider", choices=("auto", "xai-fast", "xai", "hermes"), default="auto")
    parser.add_argument("--stt-provider", choices=("auto", "local", "groq", "xai", "elevenlabs"), default="elevenlabs")
    parser.add_argument("--stt-model", default="scribe_v2")
    parser.add_argument("--elevenlabs-language-code", default="eng")
    parser.add_argument("--voice", default="eve")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--capture-timeout", type=float, default=90.0)
    parser.add_argument("--silence-ms", type=int, default=850)
    parser.add_argument("--end-padding-ms", type=int, default=180)
    parser.add_argument("--min-segment-ms", type=int, default=1200)
    parser.add_argument("--max-segment-seconds", type=float, default=9.0)
    parser.add_argument("--max-runtime-seconds", type=float, default=3600.0)
    parser.add_argument("--connect-timeout", type=float, default=30.0)
    parser.add_argument("--start-cooldown-seconds", type=float, default=5.0)
    parser.add_argument("--stop-timeout-seconds", type=float, default=8.0)
    parser.add_argument("--turn-log-jsonl", default="/tmp/zofka_fluxer_auto_voice_turns.jsonl")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    for noisy_secret_logger in ("websockets", "websockets.client", "httpcore", "httpx", "urllib3"):
        logging.getLogger(noisy_secret_logger).setLevel(logging.INFO)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
