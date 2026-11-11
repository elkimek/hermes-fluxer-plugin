"""Persistent inbound watermark: a gateway restart must not re-action messages backlog recovery would redeliver."""
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


def make_adapter(**options):
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "app.secret", "allow_all_users": True, "require_mention": False,
        "thread_replies": True, "delivery_verification": False, **options,
    }))


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # A prior test may have left a context-local secret scope behind; the
    # env var would otherwise outrank the extra= setting below.
    monkeypatch.delenv("FLUXER_ALLOW_ALL_USERS", raising=False)
    return home


@pytest.mark.asyncio
async def test_watermark_persists_across_restart(hermes_home):
    first = make_adapter()
    first._mark_inbound_seen("chan-1", "111")
    first._mark_inbound_seen("chan-1", "222")

    # A new adapter simulates the restarted gateway process.
    second = make_adapter()
    assert second._inbound_watermarks.get("chan-1") == 222
    assert second._backlog_already_handled("chan-1", "222") is True
    assert second._backlog_already_handled("chan-1", "111") is True
    assert second._backlog_already_handled("chan-1", "223") is False
    # Unknown channels are untouched by other channels' watermarks.
    assert second._backlog_already_handled("chan-2", "999") is False


@pytest.mark.asyncio
async def test_recover_backlog_skips_watermarked_messages(hermes_home):
    first = make_adapter()
    first._mark_inbound_seen("chan-1", "222")

    second = make_adapter()
    second._known_channel_ids.add("chan-1")
    second._handle_message_create = AsyncMock()

    async def request(method, path, **kwargs):
        assert method == "GET" and path == "/channels/chan-1/messages"
        return [
            {"id": "100", "channel_id": "chan-1", "timestamp": "2026-11-09T10:00:00Z", "content": "old"},
            {"id": "222", "channel_id": "chan-1", "timestamp": "2026-11-09T10:01:00Z", "content": "restart"},
            {"id": "300", "channel_id": "chan-1", "timestamp": "2026-11-09T10:02:00Z", "content": "fresh"},
        ]

    second._request = request
    cutoff = datetime(2026, 11, 1, tzinfo=timezone.utc)  # before every message above
    await second._recover_backlog(cutoff=cutoff)

    # Only the message above the watermark reaches the handler.
    assert second._handle_message_create.await_count == 1
    (call,) = second._handle_message_create.await_args_list
    assert call.args[0]["id"] == "300"


@pytest.mark.asyncio
async def test_in_process_dedup_still_gates_backlog(hermes_home):
    adapter = make_adapter()
    adapter._known_channel_ids.add("chan-1")
    adapter._handle_message_create = AsyncMock()
    adapter._remember_message_id("555")  # seen this process, never persisted-relevant

    async def request(method, path, **kwargs):
        return [{"id": "555", "channel_id": "chan-1", "content": "dup"}]

    adapter._request = request
    cutoff = datetime(2000, 1, 1, tzinfo=timezone.utc)
    await adapter._recover_backlog(cutoff=cutoff)
    adapter._handle_message_create.assert_not_awaited()


def test_missing_watermark_file_is_an_empty_watermark(hermes_home):
    adapter = make_adapter()
    assert adapter._inbound_watermarks == {}
    assert adapter._backlog_already_handled("chan-1", "123") is False


def test_malformed_watermark_file_degrades_to_empty(hermes_home):
    (hermes_home / fluxer_adapter._INBOUND_WATERMARK_FILE).write_text("{not json", encoding="utf-8")
    adapter = make_adapter()
    assert adapter._inbound_watermarks == {}

    (hermes_home / fluxer_adapter._INBOUND_WATERMARK_FILE).write_text(
        '{"version": 1, "channels": {"chan-1": "not-an-int", "chan-2": 42}}', encoding="utf-8"
    )
    adapter = make_adapter()
    assert adapter._inbound_watermarks == {"chan-2": 42}


def test_non_numeric_ids_leave_the_watermark_alone(hermes_home):
    adapter = make_adapter()
    adapter._mark_inbound_seen("chan-1", "not-a-number")
    adapter._mark_inbound_seen("", "123")
    assert adapter._inbound_watermarks == {}


@pytest.mark.asyncio
async def test_handle_message_create_marks_the_watermark(hermes_home):
    """A message admitted through the normal path advances the persisted watermark."""
    adapter = make_adapter()
    adapter.bot_user_id = "bot"

    async def extract_attachments(data):
        return [], []

    async def resolve_reply_context(data, channel_id):
        return None, None

    adapter._extract_attachments = extract_attachments
    adapter._resolve_reply_context = resolve_reply_context
    adapter._should_process_message = lambda **kwargs: (True, kwargs.get("text", ""))
    adapter._resolve_thread_source = AsyncMock(return_value=("chan-1", "channel", None, None))
    # Everything after admission is irrelevant to this test.
    adapter.handle_message = AsyncMock()

    data = {
        "id": "777",
        "channel_id": "chan-1",
        "author": {"id": "human-1", "bot": False},
        "content": "/restart",
        "timestamp": "2026-11-09T10:00:00Z",
    }
    await adapter._handle_message_create(data, {"op": 0, "t": "MESSAGE_CREATE", "d": data, "backfill": True})

    assert adapter._inbound_watermarks.get("chan-1") == 777
    second = make_adapter()
    assert second._backlog_already_handled("chan-1", "777") is True
