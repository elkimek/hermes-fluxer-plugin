from pathlib import Path
import ast
import asyncio
import inspect
import os
import time
from unittest.mock import AsyncMock, call

import pytest
try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility for CI matrix.
    import tomli as tomllib
import yaml

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def restore_fluxer_env_after_test():
    original = {key: value for key, value in os.environ.items() if key.startswith("FLUXER_")}
    yield
    for key in list(os.environ):
        if key.startswith("FLUXER_"):
            os.environ.pop(key, None)
    os.environ.update(original)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media_name", "method_name"),
    [
        ("photo.png", "send_image_file"),
        ("clip.mp4", "send_video"),
        ("report.pdf", "send_document"),
    ],
)
async def test_standalone_single_media_uses_message_as_native_caption(
    monkeypatch, tmp_path, media_name, method_name
):
    media_path = str(tmp_path / media_name)
    fake = AsyncMock()
    fake.send.return_value = fluxer_adapter.SendResult(success=True, message_id="text-id")
    getattr(fake, method_name).return_value = fluxer_adapter.SendResult(
        success=True, message_id="media-id"
    )
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "Native caption",
        media_files=[media_path],
    )

    assert result["success"] is True
    assert result["message_id"] == "media-id"
    fake.send.assert_not_awaited()
    getattr(fake, method_name).assert_awaited_once_with(
        "chan-1",
        media_path,
        caption="Native caption",
        metadata=None,
    )


@pytest.mark.asyncio
async def test_standalone_voice_keeps_text_separate_from_native_voice_message(
    monkeypatch, tmp_path
):
    media_path = str(tmp_path / "voice.ogg")
    fake = AsyncMock()
    fake.send.return_value = fluxer_adapter.SendResult(success=True, message_id="text-id")
    fake.send_voice.return_value = fluxer_adapter.SendResult(
        success=True, message_id="voice-id"
    )
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "Voice introduction",
        media_files=[(media_path, True)],
    )

    assert result["success"] is True
    assert result["message_id"] == "voice-id"
    fake.send.assert_awaited_once_with("chan-1", "Voice introduction", metadata=None)
    fake.send_voice.assert_awaited_once_with(
        "chan-1", media_path, metadata=None
    )


@pytest.mark.asyncio
async def test_standalone_plain_audio_is_a_document_with_native_caption(
    monkeypatch, tmp_path
):
    media_path = str(tmp_path / "audio.ogg")
    fake = AsyncMock()
    fake.send_document.return_value = fluxer_adapter.SendResult(
        success=True, message_id="audio-file-id"
    )
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "Audio attachment",
        media_files=[media_path],
    )

    assert result["success"] is True
    fake.send.assert_not_awaited()
    fake.send_voice.assert_not_awaited()
    fake.send_document.assert_awaited_once_with(
        "chan-1",
        media_path,
        caption="Audio attachment",
        metadata=None,
    )


@pytest.mark.asyncio
async def test_standalone_rolls_back_text_after_media_failure(monkeypatch, tmp_path):
    media_path = str(tmp_path / "voice.ogg")
    fake = AsyncMock()
    fake.send.return_value = fluxer_adapter.SendResult(
        success=True, message_id="text-id"
    )
    fake.send_voice.return_value = fluxer_adapter.SendResult(
        success=False, error="upload failed"
    )
    fake.delete_message.return_value = True
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "Voice introduction",
        media_files=[(media_path, True)],
    )

    assert result == {
        "error": "Fluxer media send failed; partial delivery rolled back",
        "rollback_complete": True,
    }
    fake.delete_message.assert_awaited_once_with("chan-1", "text-id")


@pytest.mark.asyncio
async def test_standalone_media_only_failure_does_not_claim_a_rollback(
    monkeypatch, tmp_path
):
    media_path = str(tmp_path / "report.pdf")
    fake = AsyncMock()
    fake.send_document.return_value = fluxer_adapter.SendResult(
        success=False, error="upload failed"
    )
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "",
        media_files=[media_path],
    )

    assert result == {"error": "Fluxer media send failed"}
    fake.delete_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_standalone_marks_incomplete_rollback_success_to_prevent_retry(
    monkeypatch, tmp_path
):
    media_path = str(tmp_path / "voice.ogg")
    fake = AsyncMock()
    fake.send.return_value = fluxer_adapter.SendResult(
        success=True, message_id="text-id"
    )
    fake.send_voice.return_value = fluxer_adapter.SendResult(
        success=False, error="upload failed"
    )
    fake.delete_message.return_value = False
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        "Voice introduction",
        media_files=[(media_path, True)],
    )

    assert result == {
        "success": True,
        "platform": "fluxer",
        "chat_id": "chan-1",
        "message_id": "text-id",
        "partial_delivery": True,
        "warnings": [
            "Fluxer media delivery was incomplete; existing messages were kept "
            "to prevent duplicate retries"
        ],
    }


@pytest.mark.asyncio
async def test_standalone_uses_utf16_length_for_native_caption(monkeypatch, tmp_path):
    media_path = str(tmp_path / "photo.png")
    fake = AsyncMock()
    fake.send.return_value = fluxer_adapter.SendResult(
        success=True, message_id="text-id"
    )
    fake.send_image_file.return_value = fluxer_adapter.SendResult(
        success=True, message_id="image-id"
    )
    monkeypatch.setattr(fluxer_adapter, "FluxerAdapter", lambda _config: fake)
    caption = "😀" * 2001

    result = await fluxer_adapter._standalone_send(
        PlatformConfig(enabled=True),
        "chan-1",
        caption,
        media_files=[media_path],
    )

    assert result["success"] is True
    fake.send.assert_awaited_once_with("chan-1", caption, metadata=None)
    fake.send_image_file.assert_awaited_once_with(
        "chan-1", media_path, caption=None, metadata=None
    )


@pytest.mark.asyncio
async def test_live_send_handler_preserves_media_and_force_document(tmp_path, monkeypatch):
    image = tmp_path / "original.png"
    image.write_bytes(b"not-a-real-png")
    standalone = AsyncMock(return_value={"success": True, "message_id": "media-id"})
    monkeypatch.setattr(fluxer_adapter, "_standalone_send", standalone)
    config = PlatformConfig(enabled=True)

    result = await fluxer_adapter._send_message_handler(
        {
            "message": f"[[as_document]]\nMEDIA:{image}\nNative caption",
            "thread_id": "thread-1",
        },
        "chan-1",
        "fluxer",
        config,
    )

    assert result == {"success": True, "message_id": "media-id"}
    standalone.assert_awaited_once_with(
        config,
        "chan-1",
        "Native caption",
        thread_id="thread-1",
        media_files=[(str(image.resolve()), False)],
        force_document=True,
    )


@pytest.mark.asyncio
async def test_live_send_handler_preserves_plain_text_exactly(monkeypatch):
    standalone = AsyncMock(return_value={"success": True, "message_id": "text-id"})
    monkeypatch.setattr(fluxer_adapter, "_standalone_send", standalone)
    config = PlatformConfig(enabled=True)

    result = await fluxer_adapter._send_message_handler(
        {"message": "  intentional spacing  "},
        "chan-1",
        "fluxer",
        config,
    )

    assert result == {"success": True, "message_id": "text-id"}
    standalone.assert_awaited_once_with(
        config,
        "chan-1",
        "  intentional spacing  ",
        thread_id=None,
        media_files=[],
        force_document=False,
    )


@pytest.mark.asyncio
async def test_send_handler_prefers_host_normalized_cron_context(monkeypatch, tmp_path):
    standalone = AsyncMock(return_value={"success": True, "message_id": "cron-id"})
    monkeypatch.setattr(fluxer_adapter, "_standalone_send", standalone)
    config = PlatformConfig(enabled=True)
    media_path = str(tmp_path / "report.pdf")
    media_files = [(media_path, False)]

    result = await fluxer_adapter._send_message_handler(
        {},
        "chan-1",
        "fluxer",
        config,
        normalized={
            "message": "Scheduled report",
            "thread_id": "thread-1",
            "media_files": media_files,
            "force_document": True,
        },
    )

    assert result == {"success": True, "message_id": "cron-id"}
    standalone.assert_awaited_once_with(
        config,
        "chan-1",
        "Scheduled report",
        thread_id="thread-1",
        media_files=media_files,
        force_document=True,
    )


def test_register_exposes_full_request_send_handler(monkeypatch):
    monkeypatch.setattr(
        fluxer_adapter, "_supports_normalized_send_handler_context", lambda: True
    )

    class Context:
        kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    ctx = Context()
    fluxer_adapter.register(ctx)

    assert ctx.kwargs is not None
    assert ctx.kwargs["send_message_handler"] is fluxer_adapter._send_message_handler
    assert ctx.kwargs["standalone_sender_fn"] is fluxer_adapter._standalone_send


def test_register_omits_full_request_handler_on_older_hermes(monkeypatch):
    monkeypatch.setattr(
        fluxer_adapter, "_supports_normalized_send_handler_context", lambda: False
    )

    class Context:
        kwargs = None

        def register_platform(self, **kwargs):
            self.kwargs = kwargs

    ctx = Context()
    fluxer_adapter.register(ctx)

    assert ctx.kwargs is not None
    assert ctx.kwargs["send_message_handler"] is None
    assert ctx.kwargs["standalone_sender_fn"] is fluxer_adapter._standalone_send


@pytest.mark.asyncio
async def test_agent_reaction_defaults_to_last_processed_inbound_message(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "require_mention": False,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_create(
        {
            "id": "msg-latest",
            "channel_id": "chan-1",
            "channel_type": "channel",
            "content": "React to this",
            "author": {"id": "owner-user", "username": "Alice", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )
    adapter._request = AsyncMock(return_value={})

    result = await adapter.add_reaction("chan-1", "👍")

    assert result == {"success": True, "message_id": "msg-latest"}
    adapter._request.assert_awaited_once_with(
        "PUT",
        "/channels/chan-1/messages/msg-latest/reactions/%F0%9F%91%8D/@me",
    )


@pytest.mark.asyncio
async def test_agent_reaction_requires_a_target_message(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(return_value={})

    result = await adapter.add_reaction("chan-1", "👍")

    assert result == {
        "success": False,
        "error": "no message to react to — pass message_id",
    }
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_reaction_failure_does_not_expose_api_error(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(side_effect=RuntimeError("private upstream body"))

    result = await adapter.add_reaction("chan-1", "👍", message_id="msg-explicit")

    assert result == {
        "success": False,
        "error": "reaction failed (see gateway debug log)",
    }
    assert "private upstream body" not in str(result)


@pytest.mark.asyncio
async def test_agent_reaction_rejects_an_empty_emoji(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(return_value={})

    result = await adapter.add_reaction("chan-1", "", message_id="msg-explicit")

    assert result == {"success": False, "error": "emoji is required"}
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_unreact_removes_every_reaction_owned_by_the_bot(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(
        side_effect=[
            {
                "id": "msg-explicit",
                "reactions": [
                    {"emoji": {"id": None, "name": "👍"}, "count": 2, "me": True},
                    {"emoji": {"id": "123", "name": "party"}, "count": 1, "me": True},
                    {"emoji": {"id": None, "name": "❤️"}, "count": 1, "me": False},
                ],
            },
            {},
            {},
        ]
    )

    result = await adapter.remove_reaction("chan-1", message_id="msg-explicit")

    assert result == {"success": True, "message_id": "msg-explicit", "removed": 2}
    assert adapter._request.await_args_list == [
        call("GET", "/channels/chan-1/messages/msg-explicit"),
        call(
            "DELETE",
            "/channels/chan-1/messages/msg-explicit/reactions/%F0%9F%91%8D/@me",
        ),
        call(
            "DELETE",
            "/channels/chan-1/messages/msg-explicit/reactions/party%3A123/@me",
        ),
    ]


@pytest.mark.asyncio
async def test_agent_unreact_requires_a_target_message(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(return_value={})

    result = await adapter.remove_reaction("chan-1")

    assert result == {
        "success": False,
        "error": "no message to unreact — pass message_id",
    }
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_agent_unreact_failure_does_not_expose_api_error(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._request = AsyncMock(side_effect=RuntimeError("private upstream body"))

    result = await adapter.remove_reaction("chan-1", message_id="msg-explicit")

    assert result == {
        "success": False,
        "error": "unreact failed (see gateway debug log)",
    }
    assert "private upstream body" not in str(result)


@pytest.mark.asyncio
async def test_deleted_latest_inbound_is_not_used_as_reaction_target(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "require_mention": False,
            },
        )
    )
    adapter.handle_message = AsyncMock()
    await adapter._handle_message_create(
        {
            "id": "msg-deleted",
            "channel_id": "chan-1",
            "channel_type": "channel",
            "content": "This will be deleted",
            "author": {"id": "owner-user", "username": "Alice", "bot": False},
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )
    await adapter._handle_message_delete(
        {"id": "msg-deleted", "channel_id": "chan-1"}
    )
    adapter._request = AsyncMock(return_value={})

    result = await adapter.add_reaction("chan-1", "👍")

    assert result == {
        "success": False,
        "error": "no message to react to — pass message_id",
    }
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_target_history_is_bounded(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "require_mention": False,
            },
        )
    )
    adapter.handle_message = AsyncMock()

    for index in range(1001):
        await adapter._handle_message_create(
            {
                "id": f"msg-{index}",
                "channel_id": f"chan-{index}",
                "channel_type": "channel",
                "content": "remember me",
                "author": {"id": "owner-user", "username": "Alice", "bot": False},
            },
            {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
        )

    assert len(adapter._last_inbound_by_chat) == 1000
    assert "chan-0" not in adapter._last_inbound_by_chat
    assert adapter._last_inbound_by_chat["chan-1000"] == "msg-1000"


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


def test_release_metadata_matches_v040_changelog():
    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text())
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert manifest["version"] == "0.4.0"
    assert project["version"] == "0.4.0"
    assert "## [0.4.0] - 2026-10-09" in changelog


def test_fluxer_adapter_advertises_markdown_code_blocks():
    assert fluxer_adapter.FluxerAdapter.supports_code_blocks is True


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

    assert "Hermes-Fluxer/0.2" not in source
    assert "Hermes-Fluxer/0.3" in source


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
        "FLUXER_BOT_TOKEN",
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
        "FLUXER_VOICE_BARGE_IN_WINDOW_MS",
        "FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS",
        "FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY",
    ):
        monkeypatch.delenv(key, raising=False)

    fluxer_adapter._apply_yaml_config(
        {},
        {
            "bot_token": "yaml-token",
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
                    "window_ms": 1200,
                    "capture_timeout_seconds": 2,
                    "stop_phrase_energy_threshold": 450,
                    "stop_phrase_min_ms": 120,
                    "stop_phrase_silence_ms": 180,
                    "stop_phrase_max_seconds": 2.0,
                    "after_first_audio_only": False,
                },
            }
        },
    )

    assert os.environ["FLUXER_BOT_TOKEN"] == "yaml-token"
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
    assert os.environ["FLUXER_VOICE_BARGE_IN_WINDOW_MS"] == "1200"
    assert os.environ["FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS"] == "2"
    assert os.environ["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD"] == "450"
    assert os.environ["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS"] == "120"
    assert os.environ["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS"] == "180"
    assert os.environ["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS"] == "2.0"
    assert os.environ["FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY"] == "false"


def test_check_requirements_is_a_passive_dependency_probe(monkeypatch):
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)

    assert fluxer_adapter.check_requirements() is True


def test_profile_scoped_yaml_config_stays_in_platform_extras(monkeypatch):
    for key in (
        "FLUXER_BOT_TOKEN",
        "FLUXER_HOME_CHANNEL",
        "FLUXER_HOME_GUILDS",
        "FLUXER_ALLOWED_USERS",
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_CHANNEL_IDS",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(fluxer_adapter, "_profile_scoped_config_load", lambda: True)

    extra = fluxer_adapter._apply_yaml_config(
        {},
        {
            "bot_token": "profile-token",
            "allowed_users": ["user-1"],
            "home_channel": "channel-1",
            "home_channel_name": "Home",
            "home_guilds": ["guild-1"],
            "voice": {"enabled": True, "channel_ids": ["voice-1"]},
        },
    )

    assert extra == {
        "bot_token": "profile-token",
        "allowed_users": ["user-1"],
        "allow_from": ["user-1"],
        "group_allow_from": ["user-1"],
        "home_channel": {"chat_id": "channel-1", "name": "Home"},
        "home_channel_name": "Home",
        "home_guilds": ["guild-1"],
        "home_guild_ids": ["guild-1"],
        "voice": {"enabled": True, "channel_ids": ["voice-1"]},
    }
    assert not any(key in os.environ for key in (
        "FLUXER_BOT_TOKEN",
        "FLUXER_HOME_CHANNEL",
        "FLUXER_HOME_GUILDS",
        "FLUXER_ALLOWED_USERS",
        "FLUXER_VOICE_ENABLED",
        "FLUXER_VOICE_CHANNEL_IDS",
    ))

    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=extra))
    assert adapter.bot_token == "profile-token"
    assert adapter._allowed_user_ids == {"user-1"}
    assert adapter._home_channel_ids == {"channel-1"}
    assert adapter._home_guild_ids == {"guild-1"}
    assert adapter._voice_supervisor.enabled is True
    assert adapter._voice_supervisor.configured_channel_ids == "voice-1"


def test_multiplex_yaml_without_active_scope_never_mutates_process_env(monkeypatch):
    secret_scope = pytest.importorskip("agent.secret_scope")
    monkeypatch.delenv("FLUXER_BOT_TOKEN", raising=False)
    secret_scope.set_multiplex_active(True)
    try:
        seed = fluxer_adapter._apply_yaml_config(
            {},
            {"bot_token": "profile-token", "allowed_users": "profile-user"},
        )
        open_seed = fluxer_adapter._apply_yaml_config(
            {},
            {"bot_token": "profile-token", "allow_all_users": True},
        )
    finally:
        secret_scope.set_multiplex_active(False)

    assert "FLUXER_BOT_TOKEN" not in os.environ
    assert seed is not None
    assert seed["bot_token"] == "profile-token"
    assert seed["allowed_users"] == "profile-user"
    assert seed["allow_from"] == "profile-user"
    assert seed["group_allow_from"] == "profile-user"
    assert open_seed is not None
    assert open_seed["allow_from"] == ["*"]
    assert open_seed["group_allow_from"] == ["*"]


def test_multiplexed_adapter_reads_active_profile_scope_not_process_env(monkeypatch):
    secret_scope = pytest.importorskip("agent.secret_scope")

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "primary-token")
    monkeypatch.setenv("FLUXER_ALLOWED_USERS", "primary-user")
    monkeypatch.setenv("XAI_API_KEY", "primary-xai")
    secret_scope.set_multiplex_active(True)
    scope_marker = secret_scope.set_secret_scope(
        {
            "FLUXER_BOT_TOKEN": "secondary-token",
            "FLUXER_ALLOWED_USERS": "secondary-user",
            "XAI_API_KEY": "secondary-xai",
        }
    )
    try:
        adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={}))
        child_env = adapter._voice_supervisor._child_env()
    finally:
        secret_scope.reset_secret_scope(scope_marker)
        secret_scope.set_multiplex_active(False)

    assert adapter.bot_token == "secondary-token"
    assert adapter._allowed_user_ids == {"secondary-user"}
    assert child_env["FLUXER_BOT_TOKEN"] == "secondary-token"
    assert child_env["XAI_API_KEY"] == "secondary-xai"


def test_multiplexed_voice_child_env_drops_primary_fluxer_values_and_uses_profile_extras(monkeypatch):
    secret_scope = pytest.importorskip("agent.secret_scope")

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "primary-token")
    monkeypatch.setenv("FLUXER_BASE_URL", "https://primary.example/api")
    monkeypatch.setenv("FLUXER_ALLOW_ALL_USERS", "true")
    secret_scope.set_multiplex_active(True)
    scope_marker = secret_scope.set_secret_scope({"XAI_API_KEY": "secondary-xai"})
    try:
        supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
            extra={
                "bot_token": "secondary-token",
                "base_url": "https://secondary.example/api",
                "allowed_users": "secondary-user",
                "allow_all_users": False,
            }
        )
        child_env = supervisor._child_env()
    finally:
        secret_scope.reset_secret_scope(scope_marker)
        secret_scope.set_multiplex_active(False)

    assert child_env["FLUXER_BOT_TOKEN"] == "secondary-token"
    assert child_env["FLUXER_BASE_URL"] == "https://secondary.example/api"
    assert child_env["FLUXER_ALLOWED_USERS"] == "secondary-user"
    assert child_env["FLUXER_ALLOW_ALL_USERS"] == "false"
    assert child_env["XAI_API_KEY"] == "secondary-xai"


def test_voice_supervisor_treats_null_yaml_channel_ids_as_unscoped(monkeypatch):
    monkeypatch.delenv("FLUXER_VOICE_CHANNEL_IDS", raising=False)
    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        extra={"voice": {"enabled": True, "auto_join": True, "channel_ids": None}},
    )

    assert supervisor.configured_channel_ids == ""
    assert supervisor.should_start() is False


def test_voice_supervisor_child_env_prefers_nested_vad_timeouts_over_legacy_top_level(monkeypatch, tmp_path):
    for key in (
        "FLUXER_VOICE_FRAME_MS",
        "FLUXER_VOICE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_START_COOLDOWN_SECONDS",
        "FLUXER_VOICE_STOP_TIMEOUT_SECONDS",
        "FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD",
        "FLUXER_VOICE_BARGE_IN_MIN_MS",
        "FLUXER_VOICE_BARGE_IN_WINDOW_MS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS",
        "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS",
        "FLUXER_VOICE_BRAIN_PROVIDER",
        "FLUXER_VOICE_MAX_SEGMENT_SECONDS",
    ):
        monkeypatch.delenv(key, raising=False)

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(
        plugin_root=tmp_path,
        extra={
            "voice": {
                "frame_ms": 99,
                "energy_threshold": 999,
                "brain_provider": None,
                "start_cooldown_seconds": 99,
                "stop_timeout_seconds": 99,
                "vad": {"frame_ms": 20, "energy_threshold": 300, "max_segment_seconds": None},
                "timeouts": {"start_cooldown_seconds": 5, "stop_timeout_seconds": 2},
                "barge_in": {
                    "energy_threshold": 400,
                    "min_ms": 120,
                    "window_ms": 900,
                    "stop_phrase_energy_threshold": 350,
                    "stop_phrase_min_ms": 100,
                    "stop_phrase_silence_ms": 160,
                    "stop_phrase_max_seconds": 1.5,
                },
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
    assert env["FLUXER_VOICE_BARGE_IN_WINDOW_MS"] == "900"
    assert env["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD"] == "350"
    assert env["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS"] == "100"
    assert env["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS"] == "160"
    assert env["FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS"] == "1.5"
    assert "FLUXER_VOICE_BRAIN_PROVIDER" not in env
    assert "FLUXER_VOICE_MAX_SEGMENT_SECONDS" not in env


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

    def wait(self, timeout=None) -> int:
        self.wait_calls.append(timeout)
        if timeout is None:
            deadline = time.monotonic() + 1.0
            while not (self.terminated or self.killed) and time.monotonic() < deadline:
                time.sleep(0.001)
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
    assert args[0][1] == str(root / "scripts" / "fluxer_voice_auto_join.py")
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
    assert 8 in fake_proc.wait_calls


@pytest.mark.asyncio
async def test_voice_supervisor_watcher_restarts_exited_child(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setenv("FLUXER_VOICE_CHANNEL_IDS", "voice-1")
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    processes = []

    class ExitingProcess(_FakeVoiceProcess):
        def wait(self, timeout=None) -> int:
            self.wait_calls.append(timeout)
            self.terminated = True
            return 17

    class RunningProcess(_FakeVoiceProcess):
        pass

    def fake_popen(*args, **kwargs):
        proc = ExitingProcess() if not processes else RunningProcess()
        processes.append(proc)
        return proc

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=fake_popen)
    supervisor._restart_delay_seconds = 0

    assert supervisor.start() is True
    deadline = time.monotonic() + 1.0
    while len(processes) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert len(processes) == 2
    await supervisor.stop()


@pytest.mark.asyncio
async def test_voice_supervisor_stop_cancels_pending_restart(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setenv("FLUXER_VOICE_CHANNEL_IDS", "voice-1")
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    processes = []

    class ExitingProcess(_FakeVoiceProcess):
        def wait(self, timeout=None) -> int:
            self.wait_calls.append(timeout)
            self.terminated = True
            return 17

    def fake_popen(*args, **kwargs):
        proc = ExitingProcess()
        processes.append(proc)
        return proc

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=fake_popen)
    supervisor._restart_delay_seconds = 0.2

    assert supervisor.start() is True
    deadline = time.monotonic() + 1.0
    while supervisor.process is not None and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert len(processes) == 1
    assert supervisor.process is None
    assert supervisor._watch_task is not None
    await supervisor.stop()
    await asyncio.sleep(0.25)

    assert len(processes) == 1
    assert supervisor._watch_task is None


@pytest.mark.asyncio
async def test_voice_supervisor_reconnect_start_replaces_sleeping_old_watcher(tmp_path, monkeypatch):
    monkeypatch.setenv("FLUXER_VOICE_ENABLED", "true")
    monkeypatch.setenv("FLUXER_VOICE_AUTO_JOIN", "true")
    monkeypatch.setenv("FLUXER_VOICE_CHANNEL_IDS", "voice-1")
    root = tmp_path
    (root / "scripts").mkdir()
    (root / "scripts" / "fluxer_voice_auto_join.py").write_text("", encoding="utf-8")
    processes = []

    class ExitingProcess(_FakeVoiceProcess):
        def wait(self, timeout=None) -> int:
            self.wait_calls.append(timeout)
            self.terminated = True
            return 17

    class RunningProcess(_FakeVoiceProcess):
        pass

    def fake_popen(*args, **kwargs):
        proc = ExitingProcess() if not processes else RunningProcess()
        processes.append(proc)
        return proc

    supervisor = fluxer_adapter.FluxerVoiceSupervisorProcess(extra={}, plugin_root=root, popen_factory=fake_popen)
    supervisor._restart_delay_seconds = 0.2

    assert supervisor.start() is True
    deadline = time.monotonic() + 1.0
    while supervisor.process is not None and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert len(processes) == 1
    assert supervisor.process is None
    old_watch = supervisor._watch_task
    assert old_watch is not None and not old_watch.done()

    assert supervisor.start() is True
    new_proc = processes[1]
    assert supervisor.process is new_proc
    assert supervisor._watch_process_ref is new_proc
    assert supervisor._watch_task is not old_watch
    assert supervisor._watch_task is not None and not supervisor._watch_task.done()

    new_proc.terminated = True
    deadline = time.monotonic() + 1.0
    while supervisor.process is new_proc and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert supervisor.process is None
    await supervisor.stop()


def test_voice_supervisor_signal_fallback_suppresses_missing_process(monkeypatch):
    class GoneProcess:
        pid = 999999

        def terminate(self):
            raise ProcessLookupError("gone")

    monkeypatch.setattr(fluxer_adapter.os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError("gone")))

    fluxer_adapter.FluxerVoiceSupervisorProcess._signal_process_group(
        GoneProcess(),  # type: ignore[arg-type]
        fluxer_adapter.signal.SIGTERM,
        fallback=GoneProcess().terminate,
    )


@pytest.mark.asyncio
async def test_sidecar_adapter_can_disable_gateway_state_updates(monkeypatch):
    marks = []
    monkeypatch.setattr(
        fluxer_adapter.BasePlatformAdapter,
        "_mark_connected",
        lambda self: marks.append("connected"),
        raising=False,
    )
    monkeypatch.setattr(
        fluxer_adapter.BasePlatformAdapter,
        "_mark_disconnected",
        lambda self: marks.append("disconnected"),
        raising=False,
    )

    main_adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={"bot_token": "app.secret"}))
    main_adapter._mark_connected()
    main_adapter._mark_disconnected()

    sidecar_adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "gateway_state_updates": False})
    )
    sidecar_adapter._mark_connected()
    sidecar_adapter._mark_disconnected()

    assert marks == ["connected", "disconnected"]


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
async def test_connect_gateway_once_preserves_pending_voice_joins(monkeypatch):
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

    assert adapter._pending_voice_joins == {"guild-1:voice-1": {"guild_id": "guild-1", "channel_id": "voice-1"}}
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
    assert "self._pending_voice_joins.clear()" not in adapter_source
    assert "if self._voice_supervisor:" not in adapter_source
    assert "await asyncio.wait_for(task, timeout=" in livekit_source
    assert "await asyncio.wait({task}, timeout=timeout)" in xai_room_loop_source
    assert 'getattr(current, "cancelling", None)' not in xai_room_loop_source
    assert "contextlib.suppress(asyncio.CancelledError, Exception)" in xai_room_loop_source
    assert "await asyncio.wait_for(task, timeout=timeout)" not in xai_room_loop_source
    assert 'getattr(exc_type, "_fluxer_fast_close", False)' in livekit_source
    assert 'exc_type.__name__ == "BargeInInterrupt"' not in livekit_source
    assert "not issubclass(exc_type, Exception)" in livekit_source
    assert "except asyncio.CancelledError:\n                    logger.info(\"Cancelling STT-backed voice turn %s; interrupting publisher\"" in stt_loop_source
    assert "await publisher.interrupt()" in stt_loop_source
    assert "elif voice_update_task is not None and not voice_update_task.done():" in stt_loop_source
    assert "shutdown_requested.set()\n        finished.set()" not in stt_loop_source


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
async def test_connect_accepts_gateway_reconnect_keyword(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "gateway_state_updates": False})
    )
    adapter._connect_gateway_once = AsyncMock()
    adapter._maybe_register_native_commands = AsyncMock()
    adapter._voice_supervisor.start = lambda: None

    signature = inspect.signature(adapter.connect)
    assert signature.parameters["is_reconnect"].kind is inspect.Parameter.KEYWORD_ONLY

    assert await adapter.connect(is_reconnect=True) is True
    adapter._connect_gateway_once.assert_awaited_once()
    adapter._maybe_register_native_commands.assert_not_awaited()


@pytest.mark.asyncio
async def test_initial_connect_still_registers_native_commands(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "gateway_state_updates": False})
    )
    adapter._connect_gateway_once = AsyncMock()
    adapter._maybe_register_native_commands = AsyncMock()
    adapter._voice_supervisor.start = lambda: None

    assert await adapter.connect() is True
    adapter._connect_gateway_once.assert_awaited_once()
    adapter._maybe_register_native_commands.assert_awaited_once()


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


def test_fluxer_advertises_native_long_message_chunking():
    assert fluxer_adapter.FluxerAdapter.splits_long_messages is True


@pytest.mark.asyncio
async def test_long_cron_report_chunks_below_fluxer_utf16_limit(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "delivery_verification": False,
            },
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "chunk-1"}, {"id": "chunk-2"}])
    report = ("Monday infra line 😀\n" * 300).strip()

    result = await adapter.send("chan-1", report)
    payloads = [call.kwargs["json"]["content"] for call in adapter._request.await_args_list]

    assert result.success is True
    assert len(payloads) == 2
    assert all(len(chunk.encode("utf-16-le")) // 2 <= fluxer_adapter.MAX_MESSAGE_LENGTH for chunk in payloads)


@pytest.mark.asyncio
async def test_send_sanitizes_mentions_before_utf16_chunking(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_MENTION_EVERYONE", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "allow_mention_everyone": False,
                "delivery_verification": False,
            },
        )
    )
    adapter._request = AsyncMock(side_effect=[{"id": "chunk-1"}, {"id": "chunk-2"}])

    result = await adapter.send("chan-1", ("x" * 3991) + "@everyone")
    payloads = [call.kwargs["json"]["content"] for call in adapter._request.await_args_list]

    assert result.success is True
    assert len(payloads) == 2
    assert all(fluxer_adapter.utf16_len(chunk) <= fluxer_adapter.MAX_MESSAGE_LENGTH for chunk in payloads)


@pytest.mark.asyncio
async def test_edit_sanitizes_mentions_before_utf16_truncation(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_MENTION_EVERYONE", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": "app.secret",
                "allow_all_users": True,
                "allow_mention_everyone": False,
                "delivery_verification": False,
            },
        )
    )
    adapter._request = AsyncMock(return_value={"id": "msg-edit"})

    result = await adapter.edit_message(
        "chan-1",
        "msg-edit",
        ("x" * 3991) + "@everyone",
        finalize=True,
    )
    payload = adapter._request.await_args.kwargs["json"]["content"]

    assert result.success is True
    assert fluxer_adapter.utf16_len(payload) <= fluxer_adapter.MAX_MESSAGE_LENGTH


@pytest.mark.asyncio
async def test_intermediate_stream_edit_does_not_exact_verify_racy_content(monkeypatch):
    """A newer stream edit can win before GET read-back, so only final content is exact-checked."""
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    adapter._request = AsyncMock(return_value={"id": "msg-stream"})
    adapter._verify_delivery = AsyncMock(return_value={"id": "msg-stream", "content": "newer chunk"})

    result = await adapter.edit_message("chan-1", "msg-stream", "older chunk", finalize=False)

    assert result.success is True
    adapter._verify_delivery.assert_awaited_once_with(
        "chan-1",
        "msg-stream",
        expected_content=None,
    )


@pytest.mark.asyncio
async def test_final_stream_edit_exact_verifies_visible_content(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    adapter._request = AsyncMock(return_value={"id": "msg-stream"})
    adapter._verify_delivery = AsyncMock(return_value={"id": "msg-stream", "content": "final answer"})

    result = await adapter.edit_message("chan-1", "msg-stream", "final answer", finalize=True)

    assert result.success is True
    adapter._verify_delivery.assert_awaited_once_with(
        "chan-1",
        "msg-stream",
        expected_content="final answer",
    )


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


@pytest.mark.asyncio
async def test_file_caption_is_included_in_delivery_verification(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    image_path = tmp_path / "caption.png"
    image_path.write_bytes(b"fake-png")
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"bot_token": "app.secret", "allow_all_users": True},
        )
    )
    adapter._multipart_request = AsyncMock(return_value={"id": "msg-caption"})
    adapter._verify_delivery = AsyncMock(
        return_value={
            "id": "msg-caption",
            "content": "Caption @\u200beveryone",
            "attachments": [{"id": "0"}],
        }
    )

    result = await adapter.send_image_file(
        "chan-1", str(image_path), caption="Caption @everyone"
    )

    assert result.success is True
    adapter._verify_delivery.assert_awaited_once_with(
        "chan-1",
        "msg-caption",
        expected_content="Caption @\u200beveryone",
        expected_attachment_count=1,
    )


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
async def test_voice_leave_clears_pending_join_before_delayed_voice_server_update(monkeypatch):
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    monkeypatch.delenv("FLUXER_ALLOWED_USERS", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    payloads = []

    async def fake_send(payload):
        payloads.append(payload)
        return True

    adapter._send_gateway_payload = fake_send  # type: ignore[method-assign]

    assert await adapter.send_voice_state_update("voice-1", guild_id="guild-1", connection_id="conn-1") is True
    assert adapter._pending_voice_joins == {
        "guild-1:voice-1": {"guild_id": "guild-1", "channel_id": "voice-1", "connection_id": "conn-1"}
    }

    assert await adapter.send_voice_state_update(None, guild_id="guild-1", connection_id="conn-1") is True
    assert adapter._pending_voice_joins == {}

    received = []
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
                "token": "late-livekit-token",
            },
        }
    )

    assert received == []
    assert adapter._last_voice_server_update is not None
    assert adapter._last_voice_server_update["matched_pending_join"] is False
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


@pytest.mark.asyncio
async def test_rest_request_retries_fluxer_429_using_retry_after(monkeypatch):
    import httpx

    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    request = httpx.Request("POST", "https://api.fluxer.app/v1/channels/1/messages")
    responses = [
        httpx.Response(429, headers={"Retry-After": "0.25"}, json={"retry_after": 1}, request=request),
        httpx.Response(200, json={"id": "message-1"}, request=request),
    ]

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, *args, **kwargs):
            response = responses[self.calls]
            self.calls += 1
            return response

    client = FakeClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    sleep = AsyncMock()
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", sleep)

    result = await adapter._request("POST", "/channels/1/messages", json={"content": "hello"})

    assert result == {"id": "message-1"}
    assert client.calls == 2
    sleep.assert_awaited_once_with(0.25)


@pytest.mark.asyncio
async def test_rest_request_logs_do_not_expose_paths_or_tokens(monkeypatch, caplog):
    import httpx

    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    request = httpx.Request("POST", "https://api.fluxer.app/v1/channels/private-route/messages")
    responses = [
        httpx.Response(429, headers={"Retry-After": "0"}, request=request),
        httpx.Response(400, text='{"token":"app.secret"}', request=request),
    ]

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, *args, **kwargs):
            response = responses[self.calls]
            self.calls += 1
            return response

    client = FakeClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", AsyncMock())
    caplog.set_level("INFO")

    with pytest.raises(httpx.HTTPStatusError):
        await adapter._request("POST", "/channels/private-route/messages", json={"content": "hello"})

    assert "private-route" not in caplog.text
    assert "app.secret" not in caplog.text


@pytest.mark.asyncio
async def test_multipart_retry_rewinds_file_handles(monkeypatch, tmp_path):
    import httpx

    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret", "allow_all_users": True})
    )
    upload = tmp_path / "voice.ogg"
    upload.write_bytes(b"audio-payload")
    request = httpx.Request("POST", "https://api.fluxer.app/v1/channels/1/messages")
    positions = []

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def request(self, *args, **kwargs):
            handle = kwargs["files"][0][1][1]
            positions.append(handle.tell())
            handle.read()
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(429, json={"retry_after": 0}, request=request)
            return httpx.Response(200, json={"id": "message-2"}, request=request)

    client = FakeClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", AsyncMock())

    result = await adapter._multipart_request(
        "POST",
        "/channels/1/messages",
        payload={"content": ""},
        files=[("files[0]", upload, "voice.ogg")],
    )

    assert result == {"id": "message-2"}
    assert positions == [0, 0]


def test_xai_realtime_defaults_are_generic_and_concise():
    source = (ROOT / "xai_realtime.py").read_text()

    assert "configured assistant" in source
    assert "default to English" in source
    assert "Do not answer in Spanish" not in source


def test_continuous_room_loop_script_has_noise_and_language_guardrails():
    source = (ROOT / "scripts" / "fluxer_xai_room_loop.py").read_text()
    realtime_source = (ROOT / "xai_realtime.py").read_text()

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
    assert "_fluxer_fast_close = True" in realtime_source
    assert "--disable-barge-in" in source
    assert "barge_in_min_ms" in source
    assert "--diagnose-barge-in" in source
    assert "barge probe chunk" in source
    assert "xAI response/publish failed for turn %s: %s: %s" in source


def test_voice_auto_join_default_barge_in_is_not_hair_trigger():
    source = (ROOT / "scripts" / "fluxer_voice_auto_join.py").read_text()

    assert 'FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD", "180"' in source
    assert 'FLUXER_VOICE_BARGE_IN_MIN_MS", "1200"' in source


def test_livekit_bridge_exposes_streaming_and_pcm_publish_helpers():
    source = (ROOT / "livekit_bridge.py").read_text()

    assert "def iter_remote_audio_pcm16" in source
    assert "async def publish_pcm16" in source
    assert "def pcm16_publisher" in source
    assert "AsyncIterator[bytes]" in source


# ── Outbound reply/thread contract ───────────────────────────────────────────
# Mirrors the Discord adapter's reply_to_mode (off/first/all) and the shared
# BasePlatformAdapter.create_handoff_thread contract, so Hermes-side
# expectations hold the same way on Fluxer as on the built-in platforms.


def _send_adapter(**extra_overrides):
    extra = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "delivery_verification": False,
    }
    extra.update(extra_overrides)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=extra))


def test_reply_to_mode_is_exposed_for_the_shared_contract(monkeypatch):
    monkeypatch.delenv("FLUXER_REPLY_TO_MODE", raising=False)

    assert _send_adapter()._reply_to_mode == "first"
    assert _send_adapter(reply_to_mode="all")._reply_to_mode == "all"
    assert _send_adapter(reply_to_mode="off")._reply_to_mode == "off"
    # Unknown values fall back to the default rather than disabling references.
    assert _send_adapter(reply_to_mode="every")._reply_to_mode == "first"


def test_reply_to_mode_reads_the_environment(monkeypatch):
    monkeypatch.setenv("FLUXER_REPLY_TO_MODE", "all")

    assert _send_adapter()._reply_to_mode == "all"


def test_reply_to_mode_reads_the_shared_platform_config_field(monkeypatch):
    """The canonical source is the shared PlatformConfig field, as on Discord.

    ``tests/conftest.py`` substitutes a minimal ``gateway`` stub when the Hermes
    tree is absent (as in CI), and that stub predates ``reply_to_mode``, so the
    attribute is set directly. Both the stub and the real ``PlatformConfig``
    resolve it the same way.
    """
    monkeypatch.delenv("FLUXER_REPLY_TO_MODE", raising=False)
    config = PlatformConfig(enabled=True, extra={"bot_token": "app.secret"})
    config.reply_to_mode = "all"

    assert fluxer_adapter.FluxerAdapter(config)._reply_to_mode == "all"


def test_reply_to_mode_precedence_favours_env_over_platform_config(monkeypatch):
    monkeypatch.setenv("FLUXER_REPLY_TO_MODE", "off")
    config = PlatformConfig(enabled=True, extra={"bot_token": "app.secret"})
    config.reply_to_mode = "all"

    assert fluxer_adapter.FluxerAdapter(config)._reply_to_mode == "off"


@pytest.mark.asyncio
async def test_reply_to_mode_off_suppresses_the_reply_reference(monkeypatch):
    monkeypatch.delenv("FLUXER_REPLY_TO_MODE", raising=False)
    adapter = _send_adapter(reply_to_mode="off")
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "hello", reply_to="user-msg-1")

    assert result.success is True
    payloads = [call.kwargs["json"] for call in adapter._request.await_args_list]
    assert all("message_reference" not in payload for payload in payloads)


@pytest.mark.asyncio
async def test_reply_to_mode_first_references_only_the_first_chunk(monkeypatch):
    monkeypatch.delenv("FLUXER_REPLY_TO_MODE", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "line\n" * 2000, reply_to="user-msg-1")

    assert result.success is True
    payloads = [call.kwargs["json"] for call in adapter._request.await_args_list]
    assert len(payloads) > 1
    assert payloads[0]["message_reference"] == {"message_id": "user-msg-1"}
    assert all("message_reference" not in payload for payload in payloads[1:])


@pytest.mark.asyncio
async def test_reply_to_mode_all_references_every_chunk(monkeypatch):
    monkeypatch.delenv("FLUXER_REPLY_TO_MODE", raising=False)
    adapter = _send_adapter(reply_to_mode="all")
    adapter._request = AsyncMock(return_value={"id": "msg-1"})

    result = await adapter.send("chan-1", "line\n" * 2000, reply_to="user-msg-1")

    assert result.success is True
    payloads = [call.kwargs["json"] for call in adapter._request.await_args_list]
    assert len(payloads) > 1
    assert all(
        payload["message_reference"] == {"message_id": "user-msg-1"} for payload in payloads
    )


@pytest.mark.asyncio
async def test_explicit_thread_id_targets_the_thread_channel(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(return_value={"id": "reply-1"})

    result = await adapter.send(
        "chan-1", "in-thread", reply_to="user-msg-1", metadata={"thread_id": "thread-7"}
    )

    assert result.success is True
    call = adapter._request.await_args_list[0]
    assert call.args == ("POST", "/channels/thread-7/messages")


@pytest.mark.asyncio
async def test_thread_reply_channels_start_a_thread_from_the_message(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._request = AsyncMock(side_effect=[{"id": "thread-9"}, {"id": "reply-1"}])

    result = await adapter.send("chan-1", "threaded reply", reply_to="user-msg-1")

    assert result.success is True
    calls = adapter._request.await_args_list
    assert calls[0].args == ("POST", "/channels/chan-1/messages/user-msg-1/threads")
    assert calls[1].args == ("POST", "/channels/thread-9/messages")
    # The thread-starter message already carries the parent context.
    assert "message_reference" not in calls[1].kwargs["json"]


@pytest.mark.asyncio
async def test_thread_replies_wrap_every_channel_when_enabled_globally(monkeypatch):
    monkeypatch.setenv("FLUXER_THREAD_REPLIES", "true")
    adapter = _send_adapter()
    adapter._request = AsyncMock(side_effect=[{"id": "thread-9"}, {"id": "reply-1"}])

    result = await adapter.send("chan-any", "threaded reply", reply_to="user-msg-1")

    assert result.success is True
    calls = adapter._request.await_args_list
    assert calls[0].args == ("POST", "/channels/chan-any/messages/user-msg-1/threads")
    assert calls[1].args == ("POST", "/channels/thread-9/messages")


@pytest.mark.asyncio
async def test_thread_reply_falls_back_to_the_channel_when_thread_creation_fails(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])

    async def _request(method, path, **kwargs):
        if path.endswith("/threads"):
            raise RuntimeError("threads unavailable")
        return {"id": "reply-1"}

    adapter._request = AsyncMock(side_effect=_request)

    result = await adapter.send("chan-1", "still delivered", reply_to="user-msg-1")

    assert result.success is True
    call = adapter._request.await_args_list[-1]
    assert call.args == ("POST", "/channels/chan-1/messages")
    assert call.kwargs["json"]["message_reference"] == {"message_id": "user-msg-1"}


@pytest.mark.asyncio
async def test_create_handoff_thread_returns_the_new_thread_id(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(return_value={"id": "thread-42"})

    thread_id = await adapter.create_handoff_thread("chan-1", "CLI session")

    assert thread_id == "thread-42"
    call = adapter._request.await_args_list[0]
    assert call.args == ("POST", "/channels/chan-1/threads")
    assert call.kwargs["json"]["name"] == "CLI session"
    assert "thread-42" in adapter._known_channel_ids


@pytest.mark.asyncio
async def test_create_handoff_thread_returns_none_when_creation_fails(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(side_effect=RuntimeError("unsupported"))

    assert await adapter.create_handoff_thread("chan-1", "CLI session") is None


@pytest.mark.asyncio
async def test_create_handoff_thread_falls_back_to_a_default_name(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(return_value={"id": "thread-42"})

    assert await adapter.create_handoff_thread("chan-1", "") == "thread-42"
    call = adapter._request.await_args_list[0]
    assert call.kwargs["json"]["name"] == fluxer_adapter._DEFAULT_THREAD_NAME


# ── Thread follow-ups, edits, and uploads must stay inside the thread ────────


@pytest.mark.asyncio
async def test_edits_and_deletes_route_to_the_thread_that_received_the_message(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(side_effect=[{"id": "reply-1"}, {"id": "reply-1"}, {}])

    await adapter.send("chan-1", "in-thread", metadata={"thread_id": "thread-7"})
    edited = await adapter.edit_message("chan-1", "reply-1", "edited")
    deleted = await adapter.delete_message("chan-1", "reply-1")

    assert edited.success is True
    assert deleted is True
    calls = adapter._request.await_args_list
    assert calls[0].args == ("POST", "/channels/thread-7/messages")
    assert calls[1].args == ("PATCH", "/channels/thread-7/messages/reply-1")
    assert calls[2].args == ("DELETE", "/channels/thread-7/messages/reply-1")


@pytest.mark.asyncio
async def test_edit_and_delete_fall_back_to_the_given_channel_for_unknown_messages(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(side_effect=[{"id": "some-id"}, {}])

    await adapter.edit_message("chan-1", "restored-msg", "edited")
    deleted = await adapter.delete_message("chan-1", "restored-msg")

    assert deleted is True
    calls = adapter._request.await_args_list
    assert calls[0].args == ("PATCH", "/channels/chan-1/messages/restored-msg")
    assert calls[1].args == ("DELETE", "/channels/chan-1/messages/restored-msg")


@pytest.mark.asyncio
async def test_thread_reply_uses_the_last_inbound_message_when_reply_to_is_absent(monkeypatch):
    """The gateway supplies no reply_to and no thread id for Fluxer, so the
    adapter anchors the thread on the most recent inbound message itself."""
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-7"
    adapter._request = AsyncMock(side_effect=[{"id": "thread-9"}, {"id": "reply-1"}])

    result = await adapter.send("chan-1", "threaded reply")

    assert result.success is True
    calls = adapter._request.await_args_list
    assert calls[0].args == ("POST", "/channels/chan-1/messages/user-msg-7/threads")
    assert calls[1].args == ("POST", "/channels/thread-9/messages")


@pytest.mark.asyncio
async def test_thread_reply_without_reply_to_or_inbound_history_stays_in_the_channel(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._request = AsyncMock(return_value={"id": "reply-1"})

    result = await adapter.send("chan-1", "no anchor available")

    assert result.success is True
    calls = adapter._request.await_args_list
    assert len(calls) == 1
    assert calls[0].args == ("POST", "/channels/chan-1/messages")


@pytest.mark.asyncio
async def test_progress_and_final_reply_share_one_thread(monkeypatch):
    """Two sends for the same triggering message must not open two threads."""
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-7"
    adapter._request = AsyncMock(side_effect=[{"id": "thread-9"}, {"id": "p-1"}, {"id": "f-1"}])

    await adapter.send("chan-1", "working…")
    await adapter.send("chan-1", "done")

    posts = [c.args for c in adapter._request.await_args_list]
    # Exactly one thread creation, then both messages inside it.
    assert posts.count(("POST", "/channels/chan-1/messages/user-msg-7/threads")) == 1
    assert posts[-2:] == [
        ("POST", "/channels/thread-9/messages"),
        ("POST", "/channels/thread-9/messages"),
    ]


@pytest.mark.asyncio
async def test_thread_anchor_is_reused_from_an_explicit_reply_to(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "older-msg"
    adapter._request = AsyncMock(side_effect=[{"id": "thread-9"}, {"id": "reply-1"}])

    await adapter.send("chan-1", "threaded", reply_to="user-msg-1")

    # The explicit reply_to wins over the remembered inbound message.
    assert adapter._request.await_args_list[0].args == (
        "POST",
        "/channels/chan-1/messages/user-msg-1/threads",
    )


class _FakeHTTPResponse:
    def __init__(self, status_code, text=""):
        self.status_code = status_code
        self.text = text


class _RestStatusError(Exception):
    """Stands in for the httpx error the REST helpers raise on a bad status."""

    def __init__(self, status, text=""):
        super().__init__(f"HTTP {status}")
        self.response = _FakeHTTPResponse(status, text)


async def _noop() -> None:
    return None


def _completed_typing_task():
    """A task that finishes at once, so send_typing pins the anchor without
    starting a real typing loop against the mocked request layer."""
    return asyncio.get_running_loop().create_task(_noop())


@pytest.mark.asyncio
async def test_typing_pins_the_turn_anchor_before_the_turn_can_drift(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._typing_tasks["chan-1"] = _completed_typing_task()

    await adapter.send_typing("chan-1")
    # Another accepted message arrives while this reply is still being produced.
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-2"
    adapter._request = AsyncMock(side_effect=[{"id": "thread-1"}, {"id": "reply-1"}])

    await adapter.send("chan-1", "the reply")

    # The thread is anchored on the message that started the turn, not the newest.
    assert adapter._request.await_args_list[0].args == (
        "POST",
        "/channels/chan-1/messages/user-msg-1/threads",
    )


@pytest.mark.asyncio
async def test_typing_refreshes_do_not_move_the_turn_anchor(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._typing_tasks["chan-1"] = _completed_typing_task()
    await adapter.send_typing("chan-1")

    # A mid-turn resume (e.g. re-arming typing after a clarify prompt) must not
    # re-pin against a message that arrived in the meantime.
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-2"
    await adapter.send_typing("chan-1")

    assert adapter._turn_anchor["chan-1"][0] == "user-msg-1"


@pytest.mark.asyncio
async def test_stop_typing_releases_the_anchor_for_the_next_turn(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._typing_tasks["chan-1"] = _completed_typing_task()
    await adapter.send_typing("chan-1")
    await adapter.stop_typing("chan-1")

    assert "chan-1" not in adapter._turn_anchor
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-2"
    adapter._request = AsyncMock(side_effect=[{"id": "thread-2"}, {"id": "reply-2"}])

    await adapter.send("chan-1", "next turn")

    assert adapter._request.await_args_list[0].args == (
        "POST",
        "/channels/chan-1/messages/user-msg-2/threads",
    )


@pytest.mark.asyncio
async def test_a_deleted_thread_falls_back_to_the_parent_channel(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._request = AsyncMock(
        side_effect=[
            {"id": "thread-1"},
            _RestStatusError(404),
            {"id": "fallback-1"},
        ]
    )

    result = await adapter.send("chan-1", "the reply", reply_to="user-msg-1")

    assert result.success is True
    assert result.message_id == "fallback-1"
    calls = adapter._request.await_args_list
    assert calls[1].args == ("POST", "/channels/thread-1/messages")
    assert calls[2].args == ("POST", "/channels/chan-1/messages")
    # The reference was suppressed because the thread was chosen, so it is
    # restored on the fallback rather than losing the reply link entirely.
    assert calls[2].kwargs["json"]["message_reference"] == {"message_id": "user-msg-1"}
    # The dead thread is gone from every cache, so it cannot be chosen again.
    assert ("chan-1", "user-msg-1") not in adapter._threads_by_anchor
    assert "thread-1" not in adapter._known_channel_ids
    assert "thread-1" not in adapter._thread_parents


@pytest.mark.asyncio
async def test_a_dead_thread_is_not_reopened_within_the_same_turn(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._request = AsyncMock(
        side_effect=[
            {"id": "thread-1"},
            _RestStatusError(404),
            {"id": "fallback-1"},
            {"id": "fallback-2"},
        ]
    )

    await adapter.send("chan-1", "the reply", reply_to="user-msg-1")
    await adapter.send("chan-1", "a trailing note")

    posts = [c.args for c in adapter._request.await_args_list]
    # A second thread is never created: the rest of the turn stays in the parent.
    assert posts.count(("POST", "/channels/chan-1/messages/user-msg-1/threads")) == 1
    assert posts[-1] == ("POST", "/channels/chan-1/messages")


@pytest.mark.asyncio
async def test_a_non_missing_failure_is_not_redirected_to_the_channel(monkeypatch):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._request = AsyncMock(
        side_effect=[
            {"id": "thread-1"},
            _RestStatusError(500),
        ]
    )

    result = await adapter.send("chan-1", "the reply", reply_to="user-msg-1")

    # A genuine delivery failure must surface, not silently land in the channel.
    assert result.success is False
    assert len(adapter._request.await_args_list) == 2
    assert adapter._threads_by_anchor[("chan-1", "user-msg-1")] == "thread-1"


@pytest.mark.asyncio
async def test_a_deleted_thread_during_an_upload_falls_back_to_the_channel(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._last_inbound_by_chat["chan-1"] = "user-msg-1"
    adapter._request = AsyncMock(return_value={"id": "thread-1"})
    adapter._multipart_request = AsyncMock(
        side_effect=[
            _RestStatusError(404),
            {"id": "file-1"},
        ]
    )

    result = await adapter._send_file_message("chan-1", str(document), reply_to="user-msg-1")

    assert result.success is True
    uploads = [c.args[1] for c in adapter._multipart_request.await_args_list]
    assert uploads == ["/channels/thread-1/messages", "/channels/chan-1/messages"]


@pytest.mark.asyncio
async def test_upload_targets_an_explicit_thread(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")
    adapter = _send_adapter()
    adapter._multipart_request = AsyncMock(return_value={"id": "file-1"})
    adapter._request = AsyncMock(return_value={})

    result = await adapter._send_file_message(
        "chan-1", str(document), metadata={"thread_id": "thread-7"}
    )

    assert result.success is True
    assert adapter._multipart_request.await_args.args[1] == "/channels/thread-7/messages"
    # The upload is recorded so a later edit/delete stays in the thread.
    assert await adapter.delete_message("chan-1", "file-1") is True
    assert adapter._request.await_args_list[0].args == (
        "DELETE",
        "/channels/thread-7/messages/file-1",
    )


@pytest.mark.asyncio
async def test_upload_honours_reply_mode_off(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")
    adapter = _send_adapter(reply_to_mode="off")
    adapter._multipart_request = AsyncMock(return_value={"id": "file-1"})

    await adapter._send_file_message("chan-1", str(document), reply_to="user-msg-1")

    payload = adapter._multipart_request.await_args.kwargs["payload"]
    assert "message_reference" not in payload


@pytest.mark.asyncio
async def test_upload_starts_a_thread_when_thread_replies_are_enabled(monkeypatch, tmp_path):
    monkeypatch.delenv("FLUXER_THREAD_REPLIES", raising=False)
    document = tmp_path / "report.pdf"
    document.write_bytes(b"%PDF-1.4")
    adapter = _send_adapter(thread_reply_channels=["chan-1"])
    adapter._request = AsyncMock(return_value={"id": "thread-9"})
    adapter._multipart_request = AsyncMock(return_value={"id": "file-1"})

    result = await adapter._send_file_message("chan-1", str(document), reply_to="user-msg-1")

    assert result.success is True
    assert adapter._request.await_args_list[0].args == (
        "POST",
        "/channels/chan-1/messages/user-msg-1/threads",
    )
    assert adapter._multipart_request.await_args.args[1] == "/channels/thread-9/messages"


@pytest.mark.asyncio
async def test_created_thread_records_its_parent_and_seeds_mention_memory(monkeypatch):
    monkeypatch.delenv("FLUXER_STRICT_MENTION", raising=False)
    adapter = _send_adapter()
    adapter._request = AsyncMock(return_value={"id": "thread-42"})

    await adapter.create_handoff_thread("chan-1", "CLI session")

    assert adapter._thread_parents["thread-42"] == "chan-1"
    # Follow-ups in a thread the bot opened are addressed to the bot.
    assert "thread-42" in adapter._mentioned_threads


@pytest.mark.asyncio
async def test_strict_mention_does_not_seed_a_created_thread(monkeypatch):
    monkeypatch.delenv("FLUXER_STRICT_MENTION", raising=False)
    adapter = _send_adapter(strict_mention=True)
    adapter._request = AsyncMock(return_value={"id": "thread-42"})

    await adapter.create_handoff_thread("chan-1", "CLI session")

    assert adapter._thread_parents["thread-42"] == "chan-1"
    assert "thread-42" not in adapter._mentioned_threads


def test_thread_inherits_the_parent_channel_allowlist(monkeypatch):
    monkeypatch.delenv("FLUXER_STRICT_MENTION", raising=False)
    adapter = _send_adapter(allowed_channels=["chan-1"], require_mention=True)
    adapter._thread_parents["thread-7"] = "chan-1"
    adapter._mentioned_threads["thread-7"] = None

    allowed, _ = adapter._should_process_message(
        channel_id="thread-7",
        chat_type="thread",
        text="follow up",
        data={"channel_id": "thread-7"},
        reply_to_message_id=None,
    )

    assert allowed is True


def test_thread_outside_the_allowlist_is_still_rejected(monkeypatch):
    monkeypatch.delenv("FLUXER_STRICT_MENTION", raising=False)
    adapter = _send_adapter(allowed_channels=["chan-1"], require_mention=True)

    allowed, _ = adapter._should_process_message(
        channel_id="thread-9",
        chat_type="thread",
        text="follow up",
        data={"channel_id": "thread-9"},
        reply_to_message_id=None,
    )

    assert allowed is False


def test_thread_inherits_free_response_from_its_parent(monkeypatch):
    monkeypatch.delenv("FLUXER_STRICT_MENTION", raising=False)
    adapter = _send_adapter(free_response_channels=["chan-1"], require_mention=True)
    adapter._thread_parents["thread-7"] = "chan-1"

    allowed, _ = adapter._should_process_message(
        channel_id="thread-7",
        chat_type="thread",
        text="no mention here",
        data={"channel_id": "thread-7"},
        reply_to_message_id=None,
    )

    assert allowed is True
