from pathlib import Path
import ast
import os
from unittest.mock import AsyncMock

import pytest
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for CI matrix.
    import tomli as tomllib
import yaml

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

ROOT = Path(__file__).resolve().parents[1]


def test_plugin_manifest_is_platform_plugin():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())

    assert manifest["name"] == "fluxer-platform"
    assert manifest["kind"] == "platform"
    assert manifest["label"] == "Fluxer"
    assert {item["name"] for item in manifest["requires_env"]} == {"FLUXER_BOT_TOKEN"}
    optional = {item["name"] for item in manifest["optional_env"]}
    assert {
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_AUTO_JOIN",
        "FLUXER_VOICE_TARGET_USER_IDS",
        "FLUXER_VOICE_CHANNEL_IDS",
        "FLUXER_VOICE_BRAIN_PROVIDER",
        "FLUXER_VOICE_STT_PROVIDER",
        "FLUXER_VOICE_CONTEXT_FILE",
    }.issubset(optional)


def test_voice_env_surface_is_declared_and_documented():
    code_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("adapter.py", "scripts/fluxer_voice_auto_join.py", "scripts/fluxer_stt_voice_loop.py")
    )
    used = set(__import__("re").findall(r"FLUXER_VOICE_[A-Z0-9_]+", code_text))
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    manifest_vars = {item["name"] for item in manifest["optional_env"]}
    docs_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("README.md", "after-install.md", "docs/voice-configuration.md")
    )
    documented = set(__import__("re").findall(r"FLUXER_VOICE_[A-Z0-9_]+", docs_text))

    assert used - manifest_vars == set()
    assert used - documented == set()


def test_asyncio_wait_for_timeout_handlers_are_python310_safe():
    for path in (
        "xai_realtime.py",
        "livekit_bridge.py",
        "scripts/fluxer_xai_room_loop.py",
        "scripts/fluxer_stt_voice_loop.py",
    ):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "except TimeoutError" not in source
        assert "contextlib.suppress(TimeoutError)" not in source
    for path in ("scripts/fluxer_xai_room_loop.py", "scripts/fluxer_stt_voice_loop.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "contextlib.suppress(TimeoutError, asyncio.TimeoutError)" in source


def test_user_agent_version_matches_release_manifest():
    source = (ROOT / "adapter.py").read_text(encoding="utf-8")

    assert "Hermes-Fluxer/0.1" not in source
    assert "Hermes-Fluxer/0.2" in source


def test_fluxer_voice_yaml_config_bridge_sets_env_defaults(monkeypatch):
    for key in (
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_AUTO_JOIN",
        "FLUXER_VOICE_TARGET_USER_IDS",
        "FLUXER_VOICE_CHANNEL_IDS",
        "FLUXER_VOICE_BRAIN_PROVIDER",
        "FLUXER_VOICE_STT_PROVIDER",
        "FLUXER_VOICE_SILENCE_MS",
        "FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS",
        "FLUXER_VOICE_HERMES_URL",
        "FLUXER_VOICE_HERMES_MAX_TOKENS",
        "FLUXER_VOICE_HERMES_SESSION_ID",
        "FLUXER_VOICE_HERMES_SESSION_KEY",
        "FLUXER_VOICE_FRAME_MS",
        "FLUXER_VOICE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_START_COOLDOWN_SECONDS",
        "FLUXER_VOICE_DISABLE_BARGE_IN",
        "FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD",
        "FLUXER_VOICE_BARGE_IN_MIN_MS",
        "FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS",
        "FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY",
    ):
        monkeypatch.delenv(key, raising=False)

    fluxer_adapter._apply_yaml_config(
        {},
        {
            "voice": {
                "enabled": True,
                "auto_join": True,
                "target_user_ids": ["user-1", "user-2"],
                "channel_ids": ["voice-1"],
                "brain_provider": "auto",
                "stt_provider": "elevenlabs",
                "hermes_url": "http://127.0.0.1:8642",
                "hermes_max_tokens": 123,
                "hermes_session_id": "voice-session-test",
                "hermes_session_key": "fluxer:voice:test",
                "vad": {"silence_ms": 850, "frame_ms": 20, "energy_threshold": 300},
                "timeouts": {"capture_seconds": 90, "start_cooldown_seconds": 5},
                "barge_in": {
                    "disable": True,
                    "energy_threshold": 700,
                    "min_ms": 180,
                    "capture_timeout_seconds": 2,
                    "after_first_audio_only": False,
                },
            }
        },
    )

    assert os.environ["FLUXER_VOICE_ENABLED"] == "true"
    assert os.environ["FLUXER_VOICE_AUTO_JOIN"] == "true"
    assert os.environ["FLUXER_VOICE_TARGET_USER_IDS"] == "user-1,user-2"
    assert os.environ["FLUXER_VOICE_CHANNEL_IDS"] == "voice-1"
    assert os.environ["FLUXER_VOICE_BRAIN_PROVIDER"] == "auto"
    assert os.environ["FLUXER_VOICE_STT_PROVIDER"] == "elevenlabs"
    assert os.environ["FLUXER_VOICE_HERMES_URL"] == "http://127.0.0.1:8642"
    assert os.environ["FLUXER_VOICE_HERMES_MAX_TOKENS"] == "123"
    assert os.environ["FLUXER_VOICE_HERMES_SESSION_ID"] == "voice-session-test"
    assert os.environ["FLUXER_VOICE_HERMES_SESSION_KEY"] == "fluxer:voice:test"
    assert os.environ["FLUXER_VOICE_SILENCE_MS"] == "850"
    assert os.environ["FLUXER_VOICE_FRAME_MS"] == "20"
    assert os.environ["FLUXER_VOICE_ENERGY_THRESHOLD"] == "300"
    assert os.environ["FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS"] == "90"
    assert os.environ["FLUXER_VOICE_START_COOLDOWN_SECONDS"] == "5"
    assert os.environ["FLUXER_VOICE_DISABLE_BARGE_IN"] == "true"
    assert os.environ["FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD"] == "700"
    assert os.environ["FLUXER_VOICE_BARGE_IN_MIN_MS"] == "180"
    assert os.environ["FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS"] == "2"
    assert os.environ["FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY"] == "false"


def test_voice_supervisor_child_env_prefers_nested_vad_timeouts_over_legacy_top_level(monkeypatch, tmp_path):
    for key in (
        "FLUXER_VOICE_FRAME_MS",
        "FLUXER_VOICE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_START_COOLDOWN_SECONDS",
        "FLUXER_VOICE_STOP_TIMEOUT_SECONDS",
        "FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD",
        "FLUXER_VOICE_BARGE_IN_MIN_MS",
    ):
        monkeypatch.delenv(key, raising=False)

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        plugin_root=tmp_path,
        extra={
            "voice": {
                "frame_ms": 99,
                "energy_threshold": 999,
                "start_cooldown_seconds": 99,
                "stop_timeout_seconds": 99,
                "vad": {"frame_ms": 20, "energy_threshold": 300},
                "timeouts": {"start_cooldown_seconds": 5, "stop_timeout_seconds": 2},
                "barge_in": {"energy_threshold": 400, "min_ms": 120},
            }
        },
    )

    env = supervisor._child_env()

    assert env["FLUXER_VOICE_FRAME_MS"] == "20"
    assert env["FLUXER_VOICE_ENERGY_THRESHOLD"] == "300"
    assert env["FLUXER_VOICE_START_COOLDOWN_SECONDS"] == "5"
    assert env["FLUXER_VOICE_STOP_TIMEOUT_SECONDS"] == "2"
    assert env["FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD"] == "400"
    assert env["FLUXER_VOICE_BARGE_IN_MIN_MS"] == "120"


def test_voice_supervisor_child_env_forwards_yaml_credentials_without_overriding_env(monkeypatch, tmp_path):
    for key in ("FLUXER_BOT_TOKEN", "FLUXER_BASE_URL", "FLUXER_GATEWAY_URL"):
        monkeypatch.delenv(key, raising=False)

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        plugin_root=tmp_path,
        extra={
            "bot_token": " yaml-token ",
            "base_url": "https://fluxer.example/api",
            "gateway_url": "wss://gateway.example/ws",
            "voice": {"enabled": True},
        },
    )

    env = supervisor._child_env()

    assert env["FLUXER_BOT_TOKEN"] == "yaml-token"
    assert env["FLUXER_BASE_URL"] == "https://fluxer.example/api"
    assert env["FLUXER_GATEWAY_URL"] == "wss://gateway.example/ws"

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "env-token")
    assert supervisor._child_env()["FLUXER_BOT_TOKEN"] == "env-token"


def test_fluxer_voice_yaml_config_bridge_ignores_legacy_top_level_vad_timeouts(monkeypatch):
    for key in (
        "FLUXER_VOICE_FRAME_MS",
        "FLUXER_VOICE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_START_COOLDOWN_SECONDS",
        "FLUXER_VOICE_STOP_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    fluxer_adapter._apply_yaml_config(
        {},
        {
            "voice": {
                "frame_ms": 99,
                "energy_threshold": 999,
                "start_cooldown_seconds": 99,
                "stop_timeout_seconds": 99,
                "vad": {"frame_ms": 20, "energy_threshold": 300},
                "timeouts": {"start_cooldown_seconds": 5, "stop_timeout_seconds": 2},
            }
        },
    )

    assert os.environ["FLUXER_VOICE_FRAME_MS"] == "20"
    assert os.environ["FLUXER_VOICE_ENERGY_THRESHOLD"] == "300"
    assert os.environ["FLUXER_VOICE_START_COOLDOWN_SECONDS"] == "5"
    assert os.environ["FLUXER_VOICE_STOP_TIMEOUT_SECONDS"] == "2"


class _FakeVoiceProcess:
    pid = 4242

    def __init__(self):
        self.terminated = False
        self.killed = False
        self.wait_calls = []

    def poll(self):
        return None if not (self.terminated or self.killed) else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        return 0


def test_voice_supervisor_disabled_by_default_does_not_spawn(tmp_path, monkeypatch):
    for key in (
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_AUTO_JOIN",
        "FLUXER_VOICE_CHANNEL_IDS",
        "FLUXER_VOICE_TARGET_USER_IDS",
        "FLUXER_VOICE_SILENCE_MS",
    ):
        monkeypatch.delenv(key, raising=False)
    calls = []
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=lambda *a, **kw: calls.append((a, kw)))

    assert supervisor.start() is False
    assert calls == []


def test_voice_supervisor_spawns_when_enabled_scoped_and_auto_join(tmp_path, monkeypatch):
    for key in (
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_AUTO_JOIN",
        "FLUXER_VOICE_CHANNEL_IDS",
        "FLUXER_VOICE_TARGET_USER_IDS",
        "FLUXER_VOICE_SILENCE_MS",
    ):
        monkeypatch.delenv(key, raising=False)
    calls = []
    fake_proc = _FakeVoiceProcess()

    def fake_popen(*args, **kwargs):
        calls.append((args, kwargs))
        return fake_proc

    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        extra={
            "voice": {
                "enabled": True,
                "auto_join": True,
                "channel_ids": ["voice-1"],
                "target_user_ids": ["user-1"],
                "vad": {"silence_ms": 850},
            }
        },
        plugin_root=root,
        popen_factory=fake_popen,
    )

    assert supervisor.start() is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0][1] == "scripts/fluxer_voice_auto_join.py"
    assert kwargs["cwd"] == str(root)
    assert kwargs["env"]["FLUXER_VOICE_ENABLED"] == "true"
    assert kwargs["env"]["FLUXER_VOICE_AUTO_JOIN"] == "true"
    assert kwargs["env"]["FLUXER_VOICE_SUPERVISOR_DISABLED"] == "true"
    assert kwargs["env"]["FLUXER_VOICE_CHANNEL_IDS"] == "voice-1"
    assert kwargs["env"]["FLUXER_VOICE_TARGET_USER_IDS"] == "user-1"
    assert kwargs["env"]["FLUXER_VOICE_SILENCE_MS"] == "850"


def test_voice_supervisor_spawn_failure_is_non_fatal(tmp_path, monkeypatch, caplog):
    for key in (
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_AUTO_JOIN",
        "FLUXER_VOICE_CHANNEL_IDS",
        "FLUXER_VOICE_TARGET_USER_IDS",
        "FLUXER_VOICE_SUPERVISOR_DISABLED",
    ):
        monkeypatch.delenv(key, raising=False)

    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")

    def fail_spawn(*args, **kwargs):
        raise FileNotFoundError("missing-python")

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        extra={"voice": {"enabled": True, "auto_join": True, "channel_ids": ["voice-1"]}},
        plugin_root=root,
        popen_factory=fail_spawn,
    )

    assert supervisor.start() is False
    assert supervisor.process is None
    assert "continuing without voice supervisor" in caplog.text


def test_voice_supervisor_internal_disable_guard_prevents_recursive_spawn(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setenv("FLUXER_VOICE_CHANNEL_IDS", "voice-1")
    monkeypatch.setenv("FLUXER_VOICE_SUPERVISOR_DISABLED", "true")
    calls = []
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=lambda *a, **kw: calls.append((a, kw)))

    assert supervisor.start() is False
    assert calls == []


@pytest.mark.asyncio
async def test_voice_supervisor_stop_terminates_child(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setenv("FLUXER_VOICE_CHANNEL_IDS", "voice-1")
    fake_proc = _FakeVoiceProcess()
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=lambda *a, **kw: fake_proc)

    assert supervisor.start() is True
    await supervisor.stop()

    assert fake_proc.terminated is True
    assert fake_proc.wait_calls == [8]


@pytest.mark.asyncio
async def test_reconnect_restarts_voice_supervisor(monkeypatch):
    starts = []

    class FakeSupervisor:
        def start(self):
            starts.append("start")
            return True

    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": "app.secret"}))
    adapter._voice_supervisor = FakeSupervisor()  # type: ignore[assignment]
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", AsyncMock(return_value=None))
    adapter._connect_gateway_once = AsyncMock(return_value=None)
    adapter._mark_connected = lambda: starts.append("mark_connected")

    await adapter._reconnect_loop("test")

    assert starts == ["mark_connected", "start"]


@pytest.mark.asyncio
async def test_connect_gateway_once_clears_stale_pending_voice_joins(monkeypatch):
    import asyncio
    import contextlib
    import sys
    from types import SimpleNamespace

    class FakeWebSocket:
        async def close(self):
            pass

    async def fake_connect(*args, **kwargs):
        return FakeWebSocket()

    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": "app.secret"}))
    adapter.gateway_url = "wss://gateway.example/ws"
    adapter._pending_voice_joins["guild-1:voice-1"] = {"guild_id": "guild-1", "channel_id": "voice-1"}
    adapter._recover_backlog = AsyncMock(return_value=None)  # type: ignore[method-assign]
    monkeypatch.setitem(sys.modules, "websockets", SimpleNamespace(connect=fake_connect))

    await adapter._connect_gateway_once()

    assert adapter._pending_voice_joins == {}
    assert adapter._listener_task is not None
    adapter._listener_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await adapter._listener_task


def test_realtime_voice_code_avoids_reviewed_runtime_footguns():
    livekit_source = (ROOT / "livekit_bridge.py").read_text(encoding="utf-8")
    auto_join_source = (ROOT / "scripts" / "fluxer_voice_auto_join.py").read_text(encoding="utf-8")
    stt_loop_source = (ROOT / "scripts" / "fluxer_stt_voice_loop.py").read_text(encoding="utf-8")
    xai_room_loop_source = (ROOT / "scripts" / "fluxer_xai_room_loop.py").read_text(encoding="utf-8")
    livekit_smoke_source = (ROOT / "scripts" / "fluxer_livekit_smoke.py").read_text(encoding="utf-8")
    duplex_smoke_source = (ROOT / "scripts" / "fluxer_xai_duplex_smoke.py").read_text(encoding="utf-8")
    adapter_source = (ROOT / "adapter.py").read_text(encoding="utf-8")

    assert "asyncio.timeout" not in livekit_source
    assert "await _maybe_await(source.wait_for_playout())" not in livekit_source
    assert '"allow_all_users": True' not in auto_join_source
    assert '"allow_all_users": True' not in stt_loop_source
    assert '"allow_all_users": True' not in xai_room_loop_source
    assert '"allow_all_users": True' not in livekit_smoke_source
    assert '"allow_all_users": True' not in duplex_smoke_source
    assert "await asyncio.to_thread(_post_completion)" in stt_loop_source
    assert "stt_result = await asyncio.to_thread(" in stt_loop_source
    assert "__globals__" not in stt_loop_source
    assert "await _maybe_await(room.disconnect())" in livekit_source
    assert "logger.exception(\n                    \"Fluxer voice server update bridge handler failed" in adapter_source


def test_public_tree_contains_no_private_voice_dogfood_defaults():
    forbidden = [
        "150363" + "5769218148907",
        "151090" + "5670319210500",
        "151090" + "5670319210496",
        "/home/" + "elkim",
        "VOICE_CONTEXT" + "_CACHE.md",
    ]
    checked = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".wav"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        checked.append(path)
        for item in forbidden:
            assert item not in text, f"{item!r} leaked in {path.relative_to(ROOT)}"
    assert checked


def test_adapter_is_syntax_valid_and_registers_fluxer_platform():
    source = (ROOT / "adapter.py").read_text()
    ast.parse(source)

    assert 'ctx.register_platform(' in source
    assert 'name="fluxer"' in source
    assert 'required_env=["FLUXER_BOT_TOKEN"]' in source


def test_slash_confirm_fails_closed_when_fluxer_omits_message_id():
    source = (ROOT / "adapter.py").read_text()

    assert 'error="Fluxer slash confirm message missing id"' in source
    assert 'f"{message_id}:{_normalize_reaction_emoji(emoji)}"' in source
    assert 'f":{_normalize_reaction_emoji(emoji)}"' not in source


def test_message_and_thread_dedup_use_ordered_eviction():
    source = (ROOT / "adapter.py").read_text()

    assert "from collections import OrderedDict" in source
    assert "def _remember_message_id" in source
    assert "def _remember_mentioned_thread" in source
    assert "popitem(last=False)" in source
    assert "set(list(self._seen_message_ids)" not in source
    assert "list(self._mentioned_threads)" not in source


def test_fluxer_rest_error_logs_redact_sensitive_tokens():
    source = (ROOT / "adapter.py").read_text()

    assert "def _redact_fluxer_error_body" in source
    assert "_redact_fluxer_error_body(response.text, self.bot_token)" in source
    assert "response.text[:500]" not in source


def test_deleted_slash_confirm_prompts_are_cancelled():
    source = (ROOT / "adapter.py").read_text()

    assert "slash_cancel_action" in source
    assert "from tools.slash_confirm import resolve" in source
    assert '"cancel",' in source
    assert "deleted slash-confirm prompt cancelled" in source


def test_component_actions_are_registered_or_components_fall_back():
    source = (ROOT / "adapter.py").read_text()

    assert "def _post_message_with_optional_components" in source
    assert "Fluxer components unsupported by deployment; retrying without components" in source
    assert "status_code not in {400, 404, 415, 422}" in source
    assert "Fluxer component message send failed without safe fallback" in source
    assert "def _fluxer_action_buttons" in source
    assert "def _register_component_actions" in source
    assert 'kind="exec_approval"' in source
    assert 'kind="slash_confirm"' in source


def test_native_command_application_id_rejects_token_like_values():
    source = (ROOT / "adapter.py").read_text()

    assert "def _looks_like_fluxer_id" in source
    assert "not _looks_like_fluxer_id(application_id)" in source
    assert "no valid application id is available" in source


def test_inbound_text_messages_enforce_allowed_users():
    source = (ROOT / "adapter.py").read_text()

    handle_start = source.index("    async def _handle_message_create")
    handle_end = source.index("\n\ndef check_requirements", handle_start)
    handle_source = source[handle_start:handle_end]

    assert "if not self._interaction_user_allowed(author_id):" in handle_source
    assert "Fluxer ignoring message from non-allowed user" in handle_source
    assert handle_source.index("if not self._interaction_user_allowed(author_id):") < handle_source.index(
        "await self._extract_attachments(data)"
    )


def test_application_command_interactions_enforce_allowed_users():
    source = (ROOT / "adapter.py").read_text()

    handle_start = source.index("    async def _handle_application_command_interaction")
    handle_end = source.index("\n    async def _handle_gateway_dispatch", handle_start)
    handle_source = source[handle_start:handle_end]

    assert "user_id = str(user.get(\"id\") or \"\")" in handle_source
    assert "if not self._interaction_user_allowed(user_id):" in handle_source
    assert "Fluxer ignoring application command from non-allowed user" in handle_source
    assert "You are not allowed to use this bot." in handle_source
    assert handle_source.index("if not self._interaction_user_allowed(user_id):") < handle_source.index(
        "await self.handle_message("
    )


def test_application_command_defer_ack_is_guarded():
    source = (ROOT / "adapter.py").read_text()

    handle_start = source.index("    async def _handle_application_command_interaction")
    handle_end = source.index("\n    async def _handle_gateway_dispatch", handle_start)
    handle_source = source[handle_start:handle_end]

    defer_marker = 'json={"type": 5, "data": {"flags": 64}}'
    assert defer_marker in handle_source
    defer_index = handle_source.index(defer_marker)
    before_defer = handle_source[:defer_index]
    after_defer = handle_source[defer_index:]
    assert "try:" in before_defer
    assert "except Exception as exc:" in after_defer
    assert "Fluxer application-command defer response failed" in after_defer
    assert "await self.handle_message(" in after_defer


def test_connect_guard_only_requires_bot_token_because_base_url_has_default():
    source = (ROOT / "adapter.py").read_text()

    connect_start = source.index("    async def connect")
    connect_end = source.index("\n    async def disconnect", connect_start)
    connect_source = source[connect_start:connect_end]

    assert "if not self.bot_token:" in connect_source
    assert "if not self.base_url or not self.bot_token:" not in connect_source


@pytest.mark.asyncio
async def test_inbound_message_from_non_allowed_user_is_not_dispatched(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allowed_users": "owner-user",
                "free_response_channels": ["chan-1"],
            },
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_message_create(
        {
            "id": "msg-intruder",
            "channel_id": "chan-1",
            "channel_type": "channel",
            "content": "hello from outside",
            "author": {"id": "intruder", "username": "Mallory", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert seen == []


@pytest.mark.asyncio
async def test_allowed_inbound_message_dispatches_normalized_event(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allowed_users": "owner-user",
                "free_response_channels": ["chan-1"],
            },
        )
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_message_create(
        {
            "id": "msg-owner",
            "channel_id": "chan-1",
            "channel_type": "channel",
            "content": "hello from owner",
            "author": {"id": "owner-user", "username": "Alice", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert len(seen) == 1
    assert seen[0].text == "hello from owner"
    assert seen[0].source.user_id == "owner-user"
    assert seen[0].source.chat_id == "chan-1"


@pytest.mark.asyncio
async def test_message_create_with_null_channel_and_missing_channel_id_returns_cleanly(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allowed_users": "owner-user"})
    )
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_message_create(
        {
            "id": "msg-null-channel",
            "channel": None,
            "content": "channel is missing",
            "author": {"id": "owner-user", "username": "Alice", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert seen == []


@pytest.mark.asyncio
async def test_application_command_from_non_allowed_user_gets_ephemeral_rejection(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allowed_users": "owner-user"})
    )
    adapter._request = AsyncMock(return_value={})
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_application_command_interaction(
        {
            "id": "interaction-1",
            "token": "tok-1",
            "type": 2,
            "channel_id": "chan-1",
            "member": {"user": {"id": "intruder", "username": "Mallory", "bot": False}},
            "data": {"name": "model"},
        }
    )

    assert seen == []
    adapter._request.assert_awaited_once_with(
        "POST",
        "/interactions/interaction-1/tok-1/callback",
        json={"type": 4, "data": {"content": "You are not allowed to use this bot.", "flags": 64}},
        warn_on_error=False,
    )


@pytest.mark.asyncio
async def test_fluxer_voice_attachment_dispatches_as_voice_for_stt(monkeypatch):
    """Voice-shaped Fluxer attachments should trigger Hermes STT, not generic audio handling."""
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allowed_users": "owner-user",
                "free_response_channels": ["chan-1"],
            },
        )
    )
    adapter._cache_attachment = AsyncMock(return_value=("/tmp/hermes-voice.ogg", "audio/ogg"))
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle

    await adapter._handle_message_create(
        {
            "id": "msg-voice",
            "channel_id": "chan-1",
            "channel_type": "channel",
            "content": "",
            "author": {"id": "owner-user", "username": "Alice", "bot": False},
            "attachments": [
                {
                    "id": "att-1",
                    "filename": "voice-message.ogg",
                    "url": "https://cdn.fluxer.example/voice-message.ogg",
                    "content_type": "audio/ogg",
                    "duration": 4.2,
                    "waveform": "AAAA",
                }
            ],
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert len(seen) == 1
    assert seen[0].message_type is fluxer_adapter.MessageType.VOICE
    assert seen[0].media_urls == ["/tmp/hermes-voice.ogg"]
    assert seen[0].media_types == ["audio/ogg"]
    assert seen[0].raw_message["fluxer_voice_message"] == {
        "is_voice_message": True,
        "attachment_id": "att-1",
        "filename": "voice-message.ogg",
        "content_type": "audio/ogg",
        "duration_seconds": 4.2,
        "has_waveform": True,
    }



def test_fluxer_action_buttons_generate_native_control_rows():
    buttons, actions = fluxer_adapter._fluxer_action_buttons(
        prefix="fluxer_test",
        specs=(("✅", "ok", "approve"), ("❌", "no", "deny")),
        danger_choice="no",
    )

    assert [button["label"] for button in buttons] == ["approve", "deny"]
    assert buttons[0]["style"] == 3
    assert buttons[1]["style"] == 4
    assert actions[0][1] == "ok"
    assert actions[1][1] == "no"
    assert all(action_id.startswith("fluxer_test:") for action_id, _choice in actions)


def test_fluxer_voice_metadata_is_safe_and_normalized():
    metadata = fluxer_adapter._voice_attachment_metadata(
        {
            "attachments": [
                {
                    "id": "att-voice",
                    "filename": "note.webm",
                    "duration_seconds": "2.5",
                    "waveform": "large-waveform-blob",
                    "is_voice_message": True,
                }
            ]
        }
    )

    assert metadata == {
        "is_voice_message": True,
        "attachment_id": "att-voice",
        "filename": "note.webm",
        "content_type": "audio/webm",
        "duration_seconds": 2.5,
        "has_waveform": True,
    }
    assert "large-waveform-blob" not in repr(metadata)

def test_fluxer_voice_metadata_skips_non_voice_attachment_before_voice_file():
    metadata = fluxer_adapter._voice_attachment_metadata(
        {
            "type": "VOICE_MESSAGE",
            "attachments": [
                {"id": "thumb", "filename": "thumb.jpg", "content_type": "image/jpeg"},
                {
                    "id": "voice",
                    "filename": "clip.webm",
                    "content_type": "video/webm",
                    "is_voice_message": True,
                    "duration": 3.0,
                },
            ],
        }
    )

    assert metadata == {
        "is_voice_message": True,
        "attachment_id": "voice",
        "filename": "clip.webm",
        "content_type": "audio/webm",
        "duration_seconds": 3.0,
    }


def test_fluxer_voice_metadata_preserves_zero_duration_over_fallback_keys():
    metadata = fluxer_adapter._voice_attachment_metadata(
        {
            "attachments": [
                {
                    "id": "voice-zero",
                    "filename": "zero.ogg",
                    "is_voice_message": True,
                    "duration": 0,
                    "duration_secs": 5.0,
                }
            ]
        }
    )

    assert metadata is not None
    assert metadata["duration_seconds"] == 0.0


def test_fluxer_voice_attachment_without_content_type_infers_audio_mime():
    att = {
        "filename": "voice-message.ogg",
        "url": "https://cdn.fluxer.example/voice-message.ogg",
        "duration": 3,
        "waveform": "AAAA",
    }

    assert fluxer_adapter._attachment_content_type(att) == "audio/ogg"


def test_zero_duration_attachment_without_waveform_is_not_voice_message():
    data = {
        "attachments": [
            {
                "filename": "empty-audio.ogg",
                "url": "https://cdn.fluxer.example/empty-audio.ogg",
                "duration": 0,
            }
        ]
    }

    assert fluxer_adapter._is_voice_message(data) is False


@pytest.mark.asyncio
async def test_send_voice_uploads_fluxer_voice_message_payload(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    audio_path = tmp_path / "reply.ogg"
    audio_path.write_bytes(b"fake-ogg")
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-voice-out"})
    adapter._verify_delivery = AsyncMock(return_value={"id": "msg-voice-out", "attachments": [{"id": "0"}]})

    result = await adapter.send_voice("chan-1", str(audio_path), duration=5, waveform="BBBB")

    assert result.success is True
    assert result.message_id == "msg-voice-out"
    adapter._multipart_request.assert_awaited_once()
    kwargs = adapter._multipart_request.await_args.kwargs
    assert kwargs["payload"]["flags"] == fluxer_adapter._VOICE_MESSAGE_FLAG
    assert kwargs["payload"]["attachments"] == [
        {"id": 0, "filename": "reply.ogg", "title": "reply.ogg", "duration": 5, "waveform": "BBBB"}
    ]
    assert kwargs["files"][0][0] == "files[0]"


def test_pyproject_has_runtime_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    deps = pyproject["project"]["dependencies"]

    assert any(dep.startswith("httpx") for dep in deps)
    assert any(dep.startswith("websockets") for dep in deps)


def test_realtime_voice_spike_doc_records_fluxer_livekit_flow():
    doc = (ROOT / "REALTIME_VOICE.md").read_text()

    assert "opcode 4" in doc
    assert "VOICE_SERVER_UPDATE" in doc
    assert "LiveKit" in doc
    assert "xAI Realtime" in doc
    assert "standalone plugin" in doc

@pytest.mark.asyncio
async def test_voice_server_update_bridge_handler_receives_raw_token_safely(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    received = []

    adapter._pending_voice_joins["guild-1:voice-1"] = {"guild_id": "guild-1", "channel_id": "voice-1"}
    adapter.set_voice_server_update_handler(lambda raw, safe: received.append((raw, safe)))

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "VOICE_SERVER_UPDATE",
            "d": {
                "guild_id": "guild-1",
                "channel_id": "voice-1",
                "connection_id": "conn-1",
                "endpoint": "wss://voice.example.test",
                "token": "livekit-secret-token",
            },
        }
    )

    assert len(received) == 1
    raw, safe = received[0]
    assert raw["token"] == "livekit-secret-token"
    assert safe == {
        "guild_id": "guild-1",
        "channel_id": "voice-1",
        "connection_id": "conn-1",
        "endpoint": "wss://voice.example.test",
        "has_token": True,
        "matched_pending_join": True,
    }
    assert adapter._last_voice_server_update == safe
    assert adapter._last_voice_server_update is not None
    assert "token" not in adapter._last_voice_server_update


@pytest.mark.asyncio
async def test_voice_state_update_handler_receives_user_join_and_leave(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    received = []

    adapter.set_voice_state_update_handler(lambda raw: received.append(raw))

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "VOICE_STATE_UPDATE",
            "d": {"guild_id": "guild-1", "channel_id": "voice-1", "user_id": "user-1"},
        }
    )
    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "VOICE_STATE_UPDATE",
            "d": {"guild_id": "guild-1", "channel_id": None, "user_id": "user-1"},
        }
    )

    assert received == [
        {"guild_id": "guild-1", "channel_id": "voice-1", "user_id": "user-1"},
        {"guild_id": "guild-1", "channel_id": None, "user_id": "user-1"},
    ]


@pytest.mark.asyncio
async def test_gateway_ready_event_is_set_on_ready_dispatch(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )

    assert await adapter.wait_until_gateway_ready(timeout=0.001) is False
    await adapter._handle_gateway_dispatch(
        {"op": 0, "t": "READY", "d": {"user": {"id": "bot-user"}}}
    )

    assert adapter.bot_user_id == "bot-user"
    assert await adapter.wait_until_gateway_ready(timeout=0.001) is True


def test_xai_realtime_defaults_are_generic_and_concise():
    source = (ROOT / "xai_realtime.py").read_text()

    assert "configured assistant" in source
    assert "default to English" in source
    assert "Do not answer in Spanish" not in source


def test_continuous_room_loop_script_has_noise_and_language_guardrails():
    source = (ROOT / "scripts" / "fluxer_xai_room_loop.py").read_text()

    assert "default to English" in source
    assert "Ignore background music" in source
    assert "clearly directed at the assistant" in source
    assert "WAKE_GATE_INSTRUCTIONS" in source
    assert "--disable-wake-gate" in source
    assert "RESPOND" in source
    assert "IGNORE" in source
    assert "def _speech_segments" in source
    assert "iter_remote_audio_pcm16" in source
    assert "audio_response_from_pcm16_to_sink" in source
    assert "pcm16_publisher" in source
    assert "first_audio_seconds" in source
    assert "BargeInInterrupt" in source
    assert "--disable-barge-in" in source
    assert "barge_in_min_ms" in source
    assert "--diagnose-barge-in" in source
    assert "barge probe chunk" in source
    assert "xAI response/publish failed for turn %s: %s: %s" in source


def test_livekit_bridge_exposes_streaming_and_pcm_publish_helpers():
    source = (ROOT / "livekit_bridge.py").read_text()

    assert "def iter_remote_audio_pcm16" in source
    assert "async def publish_pcm16" in source
    assert "def pcm16_publisher" in source
    assert "AsyncIterator[bytes]" in source
