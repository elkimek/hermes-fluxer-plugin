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


def test_xai_realtime_defaults_pin_english_not_spanish():
    source = (ROOT / "xai_realtime.py").read_text()

    assert "Always answer in English" in source
    assert "Do not answer in Spanish" in source


def test_continuous_room_loop_script_has_noise_and_language_guardrails():
    source = (ROOT / "scripts" / "fluxer_xai_room_loop.py").read_text()

    assert "Always answer in English" in source
    assert "Do not answer in Spanish" in source
    assert "Ignore background music" in source
    assert "Jefka" in source
    assert "obvious ASR confusion" in source
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
