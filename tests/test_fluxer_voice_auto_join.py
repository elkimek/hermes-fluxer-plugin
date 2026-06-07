from __future__ import annotations

import asyncio

import pytest

from scripts.fluxer_voice_auto_join import (
    FluxerVoiceAutoJoinSupervisor,
    parse_args,
    run,
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
    assert ["--brain-provider", "hermes"] == cmd[cmd.index("--brain-provider") : cmd.index("--brain-provider") + 2]
    assert ["--guild-id", "guild-1"] == cmd[cmd.index("--guild-id") : cmd.index("--guild-id") + 2]
    assert ["--participant-identity-prefix", "user_user-1_"] == cmd[
        cmd.index("--participant-identity-prefix") : cmd.index("--participant-identity-prefix") + 2
    ]
    assert ["--silence-ms", "850"] == cmd[cmd.index("--silence-ms") : cmd.index("--silence-ms") + 2]
    assert ["--end-padding-ms", "180"] == cmd[cmd.index("--end-padding-ms") : cmd.index("--end-padding-ms") + 2]
    assert ["--max-segment-seconds", "9.0"] == cmd[
        cmd.index("--max-segment-seconds") : cmd.index("--max-segment-seconds") + 2
    ]
    assert ["--barge-in-energy-threshold", "300"] == cmd[
        cmd.index("--barge-in-energy-threshold") : cmd.index("--barge-in-energy-threshold") + 2
    ]
    assert ["--barge-in-min-ms", "120"] == cmd[cmd.index("--barge-in-min-ms") : cmd.index("--barge-in-min-ms") + 2]
    assert "--elevenlabs-language-code" not in cmd


def test_supervisor_with_empty_targets_watches_nobody(monkeypatch):
    monkeypatch.delenv("FLUXER_VOICE_TARGET_USER_IDS", raising=False)
    monkeypatch.delenv("FLUXER_AUTO_JOIN_USER_IDS", raising=False)
    args = parse_args(["--channel-ids", "voice-1"])
    sup = FluxerVoiceAutoJoinSupervisor(args)

    assert sup.target_user_ids == set()
    assert sup.should_watch_user("user-1") is False


@pytest.mark.asyncio
async def test_auto_join_run_refuses_empty_target_users(monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.delenv("FLUXER_VOICE_TARGET_USER_IDS", raising=False)
    monkeypatch.delenv("FLUXER_AUTO_JOIN_USER_IDS", raising=False)
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)
    args = parse_args(["--channel-ids", "voice-1", "--env-file", "/tmp/definitely-missing-fluxer-env"])

    assert await run(args) == 0


@pytest.mark.asyncio
async def test_auto_join_run_fails_if_gateway_ready_timeout(monkeypatch):
    events = []

    class FakeAdapter:
        def __init__(self, config):
            pass

        def set_voice_state_update_handler(self, handler):
            events.append("handler_set")

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            events.append(("ready", timeout))
            return False

        async def disconnect(self):
            events.append("disconnect")

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setattr("scripts.fluxer_voice_auto_join.FluxerAdapter", FakeAdapter)
    args = parse_args([
        "--target-user-ids",
        "user-1",
        "--channel-ids",
        "voice-1",
        "--connect-timeout",
        "0.01",
    ])

    with pytest.raises(RuntimeError, match="READY"):
        await run(args)

    assert events == ["handler_set", ("ready", 0.01), "disconnect"]


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
    for _ in range(3):
        if not sup.operation_tasks:
            break
        await asyncio.gather(*list(sup.operation_tasks))

    assert events == [
        ("stop", "target user user-1 moved to unconfigured voice channel voice-1"),
        ("start", "guild-1", "voice-1"),
        ("stop", "target user user-1 left voice"),
    ]


@pytest.mark.asyncio
async def test_supervisor_leave_handler_does_not_wait_for_slow_stop(monkeypatch):
    args = parse_args(["--target-user-ids", "user-1", "--channel-ids", "voice-1", "--guild-ids", "guild-1"])
    sup = FluxerVoiceAutoJoinSupervisor(args)
    stop_started = asyncio.Event()
    release_stop = asyncio.Event()

    async def slow_stop(reason):
        stop_started.set()
        await release_stop.wait()

    monkeypatch.setattr(sup, "stop_voice_loop", slow_stop)

    await asyncio.wait_for(
        sup.handle_voice_state_update({"user_id": "user-1", "guild_id": "guild-1", "channel_id": None}),
        timeout=0.01,
    )
    await asyncio.wait_for(stop_started.wait(), timeout=0.1)

    assert len(sup.operation_tasks) == 1
    release_stop.set()
    await asyncio.gather(*list(sup.operation_tasks))
    await asyncio.sleep(0)
    assert sup.operation_tasks == set()


@pytest.mark.asyncio
async def test_supervisor_restarts_when_target_moves_channels(monkeypatch):
    args = parse_args(
        [
            "--target-user-ids",
            "user-1",
            "--channel-ids",
            "voice-1,voice-2",
            "--guild-ids",
            "guild-1",
            "--start-cooldown-seconds",
            "0",
        ]
    )
    sup = FluxerVoiceAutoJoinSupervisor(args)
    events = []

    class FakeProcess:
        returncode = None

        def terminate(self):
            events.append(("terminate", sup.active_channel_id))

        async def wait(self):
            self.returncode = 0
            return 0

    async def fake_create(*cmd, cwd):
        events.append(("spawn", cmd[cmd.index("--channel-id") + 1]))
        return FakeProcess()

    monkeypatch.setattr("asyncio.create_subprocess_exec", fake_create)

    await sup.start_voice_loop(guild_id="guild-1", channel_id="voice-1")
    await sup.start_voice_loop(guild_id="guild-1", channel_id="voice-2")

    assert events == [("spawn", "voice-1"), ("terminate", "voice-1"), ("spawn", "voice-2")]


@pytest.mark.asyncio
async def test_supervisor_restarts_crashed_loop_while_target_remains(monkeypatch):
    args = parse_args(
        [
            "--target-user-ids",
            "user-1",
            "--channel-ids",
            "voice-1",
            "--guild-ids",
            "guild-1",
            "--start-cooldown-seconds",
            "0",
        ]
    )
    sup = FluxerVoiceAutoJoinSupervisor(args)
    events = []

    class CrashedProcess:
        returncode = 1

        async def wait(self):
            return 1

    async def fake_start(*, guild_id, channel_id):
        events.append(("restart", guild_id, channel_id))

    proc = CrashedProcess()
    sup.process = proc  # type: ignore[assignment]
    sup.active_guild_id = "guild-1"
    sup.active_channel_id = "voice-1"
    sup.desired_guild_id = "guild-1"
    sup.desired_channel_id = "voice-1"
    monkeypatch.setattr(sup, "start_voice_loop", fake_start)

    await sup._watch_process(proc)  # type: ignore[arg-type]

    assert sup.process is None
    assert events == [("restart", "guild-1", "voice-1")]


@pytest.mark.asyncio
async def test_supervisor_does_not_restart_after_target_left(monkeypatch):
    args = parse_args(
        [
            "--target-user-ids",
            "user-1",
            "--channel-ids",
            "voice-1",
            "--guild-ids",
            "guild-1",
            "--start-cooldown-seconds",
            "0",
        ]
    )
    sup = FluxerVoiceAutoJoinSupervisor(args)
    events = []

    class StoppedProcess:
        returncode = -15

        async def wait(self):
            return -15

    async def fake_start(*, guild_id, channel_id):
        events.append(("restart", guild_id, channel_id))

    proc = StoppedProcess()
    sup.process = proc  # type: ignore[assignment]
    sup.active_guild_id = "guild-1"
    sup.active_channel_id = "voice-1"
    sup.desired_guild_id = None
    sup.desired_channel_id = None
    monkeypatch.setattr(sup, "start_voice_loop", fake_start)

    await sup._watch_process(proc)  # type: ignore[arg-type]

    assert sup.process is None
    assert events == []


@pytest.mark.asyncio
async def test_supervisor_tracks_and_clears_watch_task_for_stopped_dead_process():
    args = parse_args([
        "--target-user-ids",
        "user-1",
        "--channel-ids",
        "voice-1",
    ])
    sup = FluxerVoiceAutoJoinSupervisor(args)

    class DeadProcess:
        returncode = 0

    async def never_finishes():
        await asyncio.Event().wait()

    task = asyncio.create_task(never_finishes())
    sup.process = DeadProcess()  # type: ignore[assignment]
    sup.watch_task = task
    sup.active_guild_id = "guild-1"
    sup.active_channel_id = "voice-1"

    await sup.stop_voice_loop("already dead")

    assert sup.process is None
    assert sup.watch_task is None
    assert sup.active_channel_id is None
    assert task.cancelled()


@pytest.mark.asyncio
async def test_stop_voice_loop_does_not_hang_when_killed_process_wait_times_out():
    args = parse_args([
        "--target-user-ids",
        "user-1",
        "--channel-ids",
        "voice-1",
        "--stop-timeout-seconds",
        "0.01",
    ])
    sup = FluxerVoiceAutoJoinSupervisor(args)
    events = []

    class HungProcess:
        returncode = None

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

        async def wait(self):
            await asyncio.sleep(60)

    sup.process = HungProcess()  # type: ignore[assignment]

    await sup.stop_voice_loop("test timeout")

    assert events == ["terminate", "kill"]
