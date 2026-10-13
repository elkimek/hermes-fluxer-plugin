"""Regression coverage for the gateway setters and delivery lifetime."""

import asyncio
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def make_adapter():
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "allow_all_users": True,
        "require_mention": False,
    }))


class Titles:
    def __init__(self, title):
        self.title = title
        self.closed = False

    def find_session_by_origin(self, **kwargs):
        return "test-session"

    def get_session_title_source(self, session_id):
        return "user"

    def get_session_title(self, session_id):
        return self.title

    def close(self):
        self.closed = True


def test_real_host_setter_does_not_shadow_title_reader(monkeypatch):
    owned_reader = Titles("owned title")
    injected = Titles("injected title")
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda *args: owned_reader)
    adapter = make_adapter()
    BasePlatformAdapter.set_session_store(adapter, injected)
    assert adapter._session_title_for_chat("channel", "alice") == "owned title"
    assert adapter._session_store is injected
    adapter._close_session_store()
    assert not injected.closed


def test_owner_profile_title_is_read_outside_handler_scope(monkeypatch, tmp_path):
    default_home, secondary_home = tmp_path / "default", tmp_path / "secondary"
    default_home.mkdir()
    secondary_home.mkdir()
    default_reader, secondary_reader = Titles("default private title"), Titles("secondary title")
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    opened = []

    def open_reader(home=None):
        selected = Path(home or __import__("os").environ["HERMES_HOME"])
        opened.append(selected)
        return secondary_reader if selected == secondary_home else default_reader

    profiles = ModuleType("hermes_cli.profiles")
    profiles.get_profile_dir = lambda name: secondary_home
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", open_reader)
    adapter = make_adapter()
    adapter._owner_profile = "secondary"
    assert adapter._session_title_for_chat("channel", "alice") == "secondary title"
    assert opened == [secondary_home]


def event(anchor, user="alice", profile=None):
    return SimpleNamespace(message_id=anchor, source=SimpleNamespace(
        chat_id="channel", user_id=user, profile=profile, thread_id=None,
    ))


@pytest.mark.asyncio
async def test_concurrent_gateway_turns_keep_distinct_thread_targets(monkeypatch):
    adapter = make_adapter()
    targets = []
    both_started = asyncio.Event()
    started = []
    adapter._last_inbound_by_chat["channel"] = "message-b"
    adapter._request = AsyncMock(return_value={"id": "outbound"})
    adapter.create_thread = AsyncMock(side_effect=lambda chat, name, **kw: {"id": "thread-" + kw["message_id"]})

    async def process(self, incoming, session_key):
        started.append(incoming.message_id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        await self.send_typing("channel")
        target, _, _ = await self._resolve_outbound_target("channel", None, None)
        targets.append((incoming.message_id, target))
        await self.stop_typing("channel")

    monkeypatch.setattr(BasePlatformAdapter, "_process_message_background", process, raising=False)
    await asyncio.gather(
        adapter._process_message_background(event("message-a"), "session-a"),
        adapter._process_message_background(event("message-b", "bob"), "session-b"),
    )
    assert sorted(targets) == [("message-a", "thread-message-a"), ("message-b", "thread-message-b")]
    assert not adapter._typing_tasks


@pytest.mark.asyncio
async def test_trailing_text_and_media_keep_anchor_after_typing_stop(monkeypatch, tmp_path):
    adapter = make_adapter()
    targets = []
    adapter._last_inbound_by_chat["channel"] = "message-a"
    adapter._request = AsyncMock(return_value={"id": "outbound"})
    adapter.create_thread = AsyncMock(side_effect=lambda chat, name, **kw: {"id": "thread-" + kw["message_id"]})
    attachment = tmp_path / "note.txt"
    adapter._multipart_request = AsyncMock(return_value={"id": "attachment"})
    attachment.write_text("synthetic attachment")

    async def process(self, incoming, session_key):
        await self.send_typing("channel")
        first, _, _ = await self._resolve_outbound_target("channel", None, None)
        self._last_inbound_by_chat["channel"] = "message-b"
        await self.stop_typing("channel")
        footer, _, _ = await self._resolve_outbound_target("channel", None, None)
        assert (await self.send_document("channel", str(attachment))).success
        targets.extend([first, footer])

    monkeypatch.setattr(BasePlatformAdapter, "_process_message_background", process, raising=False)
    await adapter._process_message_background(event("message-a"), "session-a")
    assert targets == ["thread-message-a", "thread-message-a"]
    upload_paths = [call.args[1] for call in adapter._multipart_request.await_args_list]
    assert upload_paths == ["/channels/thread-message-a/messages"]


@pytest.mark.asyncio
async def test_unscoped_outbound_does_not_guess_latest_inbound():
    adapter = make_adapter()
    adapter._last_inbound_by_chat["channel"] = "someone-elses-message"
    adapter.create_thread = AsyncMock(return_value={"id": "wrong-thread"})
    target, _, _ = await adapter._resolve_outbound_target("channel", None, None)
    assert target == "channel"
    adapter.create_thread.assert_not_awaited()
