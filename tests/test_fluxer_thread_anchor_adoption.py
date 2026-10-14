"""A thread Fluxer already has for the anchor is adopted, not abandoned.

Fluxer anchors a thread to the message it was started from and the thread carries
that message's id, which the API states outright when it refuses a second start
(``400 THREAD_ALREADY_CREATED_FOR_MESSAGE``). The adapter's anchor map is in
memory only, so that refusal is reachable after a restart, after an evicted entry,
and whenever two sends for one turn interleave. Reading it as "no thread" demoted
the whole turn to the parent channel -- the reply that lands outside its thread.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

# Verbatim from the live API; the adapter matches on the code, not the message.
THREAD_EXISTS_BODY = (
    '{"code":"THREAD_ALREADY_CREATED_FOR_MESSAGE",'
    '"message":"A thread has already been created for this message."}'
)


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {"bot_token": "app.secret", "allow_all_users": True, "require_mention": False}
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeHttpError(Exception):
    """Shaped like the ``httpx.HTTPStatusError`` the REST helpers raise."""

    def __init__(self, status_code: int, text: str = "") -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = _FakeResponse(status_code, text)


def test_the_refusal_is_recognised_only_for_a_thread_that_already_exists():
    assert fluxer_adapter._thread_already_exists_for_message(
        _FakeHttpError(400, THREAD_EXISTS_BODY)
    )
    # Any other 4xx keeps the parent-channel fallback: those are real failures.
    assert not fluxer_adapter._thread_already_exists_for_message(
        _FakeHttpError(400, '{"code":"MISSING_PERMISSIONS"}')
    )
    assert not fluxer_adapter._thread_already_exists_for_message(
        _FakeHttpError(403, THREAD_EXISTS_BODY)
    )
    assert not fluxer_adapter._thread_already_exists_for_message(RuntimeError("boom"))


@pytest.mark.asyncio
async def test_an_existing_thread_is_adopted_instead_of_demoting_to_the_channel():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    adapter._schedule_title_rename = lambda *args: None

    thread_id = await adapter._start_thread_from_message("chan-1", "msg-1")

    assert thread_id == "msg-1"  # Fluxer: the thread carries its anchor's id
    assert adapter.auto_thread_info_for_chat("chan-1") == ("msg-1", "deploy the thing")


@pytest.mark.asyncio
async def test_an_adopted_thread_is_remembered_for_the_rest_of_the_turn():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    adapter._schedule_title_rename = lambda *args: None

    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "msg-1"
    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "msg-1"
    adapter.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_adopted_thread_keeps_the_name_the_server_already_has():
    """Adoption must not claim a rename that never happened, or the placeholder
    guard stops protecting a thread someone renamed by hand."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="Gwyn's own thread name")
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert adapter.auto_thread_info_for_chat("chan-1") == ("msg-1", "Gwyn's own thread name")
    # The poll is seeded with the real name, so a later rename is declined unless
    # the thread still carries it.
    assert watched == [("chan-1", "msg-1", "Gwyn's own thread name", None)]


@pytest.mark.asyncio
async def test_an_unreadable_server_name_still_adopts_the_thread():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value=None)
    adapter._schedule_title_rename = lambda *args: None

    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "msg-1"


@pytest.mark.asyncio
async def test_a_real_creation_failure_still_falls_back_to_the_channel():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(500, "internal error"))

    assert await adapter._start_thread_from_message("chan-1", "msg-1") is None


@pytest.mark.asyncio
async def test_a_restart_re_adopts_a_thread_this_process_never_created():
    """Nothing survives a gateway restart, so the only way back into the thread is
    the server's own refusal."""
    adapter = _adapter()
    assert adapter._threads_by_anchor == {}
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    adapter._schedule_title_rename = lambda *args: None

    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "msg-1"


@pytest.mark.asyncio
async def test_two_interleaved_sends_open_one_thread():
    """The live failure: one POST wins, the other gets the refusal, and before this
    the loser's turn -- often the final reply -- went to the parent channel."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    inside = asyncio.Event()
    release = asyncio.Event()
    calls: list = []

    async def slow_create(chat_id, name, *, message_id=None, **kwargs):
        calls.append((chat_id, name, message_id))
        inside.set()
        await release.wait()
        return {"id": "th-1", "name": "deploy the thing"}

    adapter.create_thread = slow_create
    adapter._schedule_title_rename = lambda *args: None

    first = asyncio.create_task(adapter._start_thread_from_message("chan-1", "msg-1"))
    await inside.wait()  # the first send is inside the REST call
    second = asyncio.create_task(adapter._start_thread_from_message("chan-1", "msg-1"))
    for _ in range(5):  # let the second reach the per-anchor lock
        await asyncio.sleep(0)
    release.set()

    assert await first == "th-1"
    assert await second == "th-1"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_the_creation_locks_stay_bounded():
    adapter = _adapter()
    adapter._threads_by_anchor_max = 3

    for index in range(6):
        adapter._thread_create_lock(("chan-1", f"msg-{index}"))

    assert len(adapter._thread_create_locks) == 3
    # Re-asking for a key must hand back the same lock, not a fresh one.
    assert adapter._thread_create_lock(("chan-1", "msg-5")) is adapter._thread_create_lock(
        ("chan-1", "msg-5")
    )
