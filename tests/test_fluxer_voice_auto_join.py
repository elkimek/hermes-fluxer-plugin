from __future__ import annotations

import pytest

from scripts.fluxer_voice_auto_join import (
    FluxerVoiceAutoJoinSupervisor,
    parse_args,
    split_csv,
    voice_state_channel_id,
    voice_state_guild_id,
    voice_state_user_id,
)


def test_split_csv_and_voice_state_extractors():
    assert split_csv(" a, b ,,c ") == {"a", "b", "c"}
    nested = {"member": {"user": {"id": "user-1"}}, "channelId": "voice-1", "guildId": "guild-1"}
    assert voice_state_user_id(nested) == "user-1"
    assert voice_state_channel_id(nested) == "voice-1"
    assert voice_state_guild_id(nested) == "guild-1"
    assert voice_state_channel_id({"channel_id": None}) is None


def test_supervisor_builds_voice_loop_command_with_target_prefix():
    args = parse_args(
        [
            "--target-user-ids",
            "user-1",
            "--channel-ids",
            "voice-1",
            "--guild-ids",
            "guild-1",
            "--python",
            "/py",
            "--turn-log-jsonl",
            "/tmp/turns.jsonl",
        ]
    )
    sup = FluxerVoiceAutoJoinSupervisor(args)

    cmd = sup.build_voice_loop_command(guild_id="guild-1", channel_id="voice-1")

    assert cmd[0] == "/py"
    assert "scripts/fluxer_stt_voice_loop.py" in cmd[1]
    assert ["--channel-id", "voice-1"] == cmd[cmd.index("--channel-id") : cmd.index("--channel-id") + 2]
    assert ["--guild-id", "guild-1"] == cmd[cmd.index("--guild-id") : cmd.index("--guild-id") + 2]
    assert ["--participant-identity-prefix", "user_user-1_"] == cmd[
        cmd.index("--participant-identity-prefix") : cmd.index("--participant-identity-prefix") + 2
    ]


@pytest.mark.asyncio
async def test_supervisor_starts_on_target_join_and_stops_on_leave(monkeypatch):
    args = parse_args(["--target-user-ids", "user-1", "--channel-ids", "voice-1", "--guild-ids", "guild-1"])
    sup = FluxerVoiceAutoJoinSupervisor(args)
    events = []

    async def fake_start(*, guild_id, channel_id):
        events.append(("start", guild_id, channel_id))

    async def fake_stop(reason):
        events.append(("stop", reason))

    monkeypatch.setattr(sup, "start_voice_loop", fake_start)
    monkeypatch.setattr(sup, "stop_voice_loop", fake_stop)

    await sup.handle_voice_state_update({"user_id": "intruder", "guild_id": "guild-1", "channel_id": "voice-1"})
    await sup.handle_voice_state_update({"user_id": "user-1", "guild_id": "guild-2", "channel_id": "voice-1"})
    await sup.handle_voice_state_update({"user_id": "user-1", "guild_id": "guild-1", "channel_id": "voice-1"})
    await sup.handle_voice_state_update({"user_id": "user-1", "guild_id": "guild-1", "channel_id": None})

    assert events == [
        ("start", "guild-1", "voice-1"),
        ("stop", "target user user-1 left voice"),
    ]
