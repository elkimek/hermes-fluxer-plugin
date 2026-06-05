from pathlib import Path
import ast
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
    assert 'self._pending_component_actions[custom_id] = {' in source
    assert '"kind": "exec_approval"' in source
    assert '"kind": "slash_confirm"' in source


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


def test_voice_state_update_payload_matches_fluxer_livekit_handshake_shape():
    payload = fluxer_adapter._build_voice_state_update_payload(
        channel_id="voice-chan",
        guild_id="guild-1",
        connection_id="conn-1",
        self_mute=True,
        self_deaf=False,
    )

    assert payload == {
        "op": 4,
        "d": {
            "guild_id": "guild-1",
            "channel_id": "voice-chan",
            "self_mute": True,
            "self_deaf": False,
            "self_video": False,
            "self_stream": False,
            "viewer_stream_keys": [],
            "connection_id": "conn-1",
        },
    }


@pytest.mark.asyncio
async def test_fluxer_adapter_can_send_voice_state_update_over_gateway(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, raw):
            self.sent.append(raw)

    fake_ws = FakeWebSocket()
    adapter._ws = fake_ws

    result = await adapter.send_voice_state_update("voice-chan", guild_id="guild-1", connection_id="conn-1")

    assert result is True
    assert fake_ws.sent == [
        '{"op": 4, "d": {"guild_id": "guild-1", "channel_id": "voice-chan", "self_mute": false, "self_deaf": true, "self_video": false, "self_stream": false, "viewer_stream_keys": [], "connection_id": "conn-1"}}'
    ]


@pytest.mark.asyncio
async def test_fluxer_adapter_voice_state_update_fails_closed_without_gateway(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )

    result = await adapter.send_voice_state_update("voice-chan")

    assert result is False


@pytest.mark.asyncio
async def test_send_voice_state_update_tracks_pending_voice_join_without_token(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )

    class FakeWebSocket:
        async def send(self, raw):
            pass

    adapter._ws = FakeWebSocket()

    result = await adapter.send_voice_state_update("voice-chan", guild_id="guild-1", connection_id="conn-1")

    assert result is True
    assert adapter._pending_voice_joins == {
        "guild-1:voice-chan": {"guild_id": "guild-1", "channel_id": "voice-chan", "connection_id": "conn-1"}
    }


@pytest.mark.asyncio
async def test_voice_server_update_is_captured_without_persisting_livekit_token(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    adapter._pending_voice_joins["guild-1:voice-chan"] = {
        "guild_id": "guild-1",
        "channel_id": "voice-chan",
        "connection_id": "conn-1",
    }

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "VOICE_SERVER_UPDATE",
            "d": {
                "guild_id": "guild-1",
                "channel_id": "voice-chan",
                "connection_id": "conn-1",
                "endpoint": "wss://livekit.fluxer.example",
                "token": "secret-livekit-token",
            },
        }
    )

    assert adapter._pending_voice_joins == {}
    assert adapter._last_voice_server_update == {
        "guild_id": "guild-1",
        "channel_id": "voice-chan",
        "connection_id": "conn-1",
        "endpoint": "wss://livekit.fluxer.example",
        "has_token": True,
        "matched_pending_join": True,
    }
    assert "secret-livekit-token" not in repr(adapter._last_voice_server_update)


@pytest.mark.asyncio
async def test_unmatched_voice_server_update_is_still_sanitized(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )

    await adapter._handle_gateway_dispatch(
        {
            "op": 0,
            "t": "VOICE_SERVER_UPDATE",
            "d": {
                "guild_id": None,
                "channel_id": "dm-call",
                "connection_id": "conn-2",
                "endpoint": "wss://livekit.fluxer.example",
                "token": "another-secret-token",
            },
        }
    )

    assert adapter._last_voice_server_update["has_token"] is True
    assert adapter._last_voice_server_update["matched_pending_join"] is False
    assert "token" not in adapter._last_voice_server_update
    assert "another-secret-token" not in repr(adapter._last_voice_server_update)


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
