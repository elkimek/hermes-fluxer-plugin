"""Optional session naming must not prevent replies or leak lookup details."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


class BrokenString:
    def __str__(self):
        raise RuntimeError("synthetic-private-detail")


@pytest.fixture
def adapter():
    instance = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only",
        "allow_all_users": True,
        "require_mention": False,
        "thread_replies": True,
        "delivery_verification": False,
    }))
    # Exercise the same setter as the host, not only direct construction.
    instance.set_session_store(object())
    return instance


def _store():
    return SimpleNamespace(
        find_session_by_origin=Mock(return_value="session-1"),
        get_session_title_source=Mock(return_value="user"),
        get_session_title=Mock(return_value="Existing title"),
        list_sessions_rich=Mock(return_value=[{"chat_id": "channel-1", "user_id": "user-1"}]),
    )


def _install_failure(adapter, monkeypatch, fault):
    store = _store()
    getter = Mock(return_value=store)
    if fault == "acquisition":
        getter.side_effect = RuntimeError("synthetic-private-detail")
    elif fault == "title-text":
        store.get_session_title.return_value = BrokenString()
    elif fault in {"chat-text", "user-text"}:
        field = "chat_id" if fault == "chat-text" else "user_id"
        store.list_sessions_rich.return_value[0][field] = BrokenString()
    monkeypatch.setattr(adapter, "_get_session_store", getter)
    return store


@pytest.mark.parametrize("fault", ["acquisition", "chat-text", "title-text"])
def test_title_lookup_returns_none_for_optional_naming_failure(adapter, monkeypatch, fault):
    _install_failure(adapter, monkeypatch, fault)
    chat_id = BrokenString() if fault == "chat-text" else "channel-1"

    assert adapter._session_title_for_chat(chat_id, "user-1") is None


@pytest.mark.parametrize("fault", ["acquisition", "key-text", "chat-text", "user-text"])
def test_origin_lookup_returns_none_for_optional_naming_failure(adapter, monkeypatch, fault):
    _install_failure(adapter, monkeypatch, fault)
    session_key = BrokenString() if fault == "key-text" else "agent:main:fluxer:channel-1:user-1"

    assert adapter._session_origin_for_key(session_key) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["healthy", "acquisition", "title-text"])
@pytest.mark.parametrize("thread_available", [True, False])
async def test_naming_failure_does_not_block_outbound_reply(
    adapter, monkeypatch, fault, thread_available
):
    _install_failure(adapter, monkeypatch, fault)
    adapter._inbound_text_by_message["trigger-1"] = "Synthetic trigger"
    adapter._inbound_author_by_message["trigger-1"] = "user-1"
    adapter.create_thread = AsyncMock(return_value={"id": "thread-1"})
    if not thread_available:
        adapter.create_thread.side_effect = RuntimeError("synthetic thread unavailable")
    adapter._request = AsyncMock(return_value={"id": "reply-1"})
    try:
        result = await adapter.send("channel-1", "Reply still arrives", reply_to="trigger-1")

        assert result.success is True
        assert result.message_id == "reply-1"
        name = "Existing title" if fault == "healthy" else "Synthetic trigger"
        adapter.create_thread.assert_awaited_once_with("channel-1", name, message_id="trigger-1")
        adapter._request.assert_awaited_once()
        call = adapter._request.await_args
        assert call is not None
        destination = "thread-1" if thread_available else "channel-1"
        assert call.args == ("POST", f"/channels/{destination}/messages")
        assert call.kwargs["json"]["content"] == "Reply still arrives"
    finally:
        # Keep the real scheduling path, but never leave a poller past this test.
        tasks = list(adapter._title_tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["acquisition", "title-text"])
async def test_background_title_poll_keeps_placeholder_on_lookup_failure(
    adapter, monkeypatch, fault
):
    _install_failure(adapter, monkeypatch, fault)
    monkeypatch.setattr(fluxer_adapter, "_TITLE_RENAME_WINDOW_SECONDS", 0)
    adapter.rename_thread = AsyncMock()

    await adapter._rename_thread_when_titled("channel-1", "thread-1", "Placeholder", "user-1")

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["acquisition", "chat-text", "user-text"])
async def test_title_command_declines_thread_rename_on_origin_failure(
    adapter, monkeypatch, fault
):
    _install_failure(adapter, monkeypatch, fault)
    adapter._record_auto_thread("channel-1", "thread-1", "Placeholder", "user-1")
    adapter.rename_thread = AsyncMock()

    await adapter._rename_thread_for_session("agent:main:fluxer:channel-1:user-1", "New title")

    adapter.rename_thread.assert_not_awaited()
    assert adapter.auto_thread_info_for_chat("channel-1") == ("thread-1", "Placeholder")


@pytest.mark.parametrize("helper", ["title", "origin"])
def test_lookup_diagnostics_omit_identifiers_and_exception_text(
    adapter, monkeypatch, caplog, helper
):
    store = _store()
    monkeypatch.setattr(adapter, "_get_session_store", Mock(return_value=store))
    caplog.set_level(logging.DEBUG, logger=fluxer_adapter.logger.name)
    error = RuntimeError("synthetic-private-detail")
    if helper == "title":
        store.find_session_by_origin.side_effect = error
        result = adapter._session_title_for_chat("synthetic-private-channel")
    else:
        store.list_sessions_rich.side_effect = error
        result = adapter._session_origin_for_key("agent:main:fluxer:synthetic-private-channel")

    assert result is None
    assert "RuntimeError" in caplog.text
    assert "synthetic-private" not in caplog.text
