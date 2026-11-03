"""Cancellation of optional membership must not abandon thread-title setup."""
import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

import adapter
from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_cancelled_membership_preserves_title_setup_for_cached_thread(monkeypatch):
    instance = adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "gateway_state_updates": False,
    }))
    instance._inbound_author_by_message["trigger"] = "admitted-author"
    instance._inbound_text_by_message["trigger"] = "Synthetic placeholder"
    monkeypatch.setattr(instance, "_session_title_for_chat", lambda *_args: None)
    instance.create_thread = AsyncMock(return_value={"id": "created-thread", "name": "Synthetic placeholder"})
    instance._request = AsyncMock(return_value={"id": "final-message"})
    schedule = Mock()
    monkeypatch.setattr(instance, "_schedule_title_rename", schedule)
    entered = asyncio.Event()

    async def membership(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(instance, "_add_thread_member", membership)
    pending = asyncio.create_task(instance.send("parent", "progress", reply_to="trigger"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert instance._threads_by_anchor[("parent", "trigger")] == "created-thread"
    assert instance._thread_rename_guard("created-thread") == "Synthetic placeholder"
    schedule.assert_called_once_with("trigger", "created-thread", "Synthetic placeholder", "admitted-author")
    final = await instance.send("parent", "final", reply_to="trigger")
    assert final.success
    instance.create_thread.assert_awaited_once()
    schedule.assert_called_once()
