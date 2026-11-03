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
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))
    # Name and ownership are independent reads; do not let adoption tests reach
    # a real provider when server ownership is irrelevant to their assertion.
    adapter._fetch_thread_info = AsyncMock(return_value={})
    return adapter


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
    """Adoption must not claim a rename that never happened. The cached label is the
    real name, so a human's choice stays visible, but the rename *guard* stays the
    name this adapter would have given the thread: the guard is what authorises an
    overwrite, so seeding it with the server's current name would authorise
    replacing that very name."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="Gwyn's own thread name")
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert adapter.auto_thread_info_for_chat("chan-1") == ("msg-1", "Gwyn's own thread name")
    assert watched == [("msg-1", "msg-1", "deploy the thing", None)]


@pytest.mark.asyncio
async def test_a_human_thread_name_survives_the_session_title_rename():
    """The guard declines the rename once the thread no longer carries the name this
    adapter gave it — which is exactly the case for a thread a human renamed."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="Gwyn's own thread name")
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")
    guard = watched[0][2]

    # The real rename path, against the name the server actually holds.
    adapter._request = AsyncMock(return_value={})
    applied = await adapter.rename_thread("msg-1", "Deploying the thing", only_if_current_name=guard)

    assert applied is False
    adapter._request.assert_not_awaited()  # no PATCH: the human's name stands


@pytest.mark.asyncio
async def test_a_thread_still_carrying_the_placeholder_is_renamed_to_the_title():
    """The other direction: the guard still lets the session title through when the
    thread is untouched, which is the restart-mid-turn case this lane exists for."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")
    guard = watched[0][2]

    assert guard == "deploy the thing"
    calls: list = []

    async def fake_request(method, path, **kwargs):
        calls.append(method)
        return {}

    adapter._request = fake_request
    applied = await adapter.rename_thread("msg-1", "Deploying the thing", only_if_current_name=guard)

    assert applied is True
    assert calls == ["PATCH"]


@pytest.mark.asyncio
async def test_slash_title_is_unguarded_rather_than_label_guarded_after_guard_eviction():
    """The guard map is bounded independently of the label maps, so a live label can
    outlive its guard. Falling back to the label would reinstate exactly the bug: an
    adopted thread's label is the server's own name. With no guard remembered the
    rename goes out unguarded, because an explicit /title must still land."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="Gwyn's own thread name")
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message("chan-1", "msg-1")
    # Evict the guard, keep the label -- exactly what an independent bound allows.
    adapter._auto_thread_guards.clear()
    assert adapter.auto_thread_info_for_chat("chan-1") == ("msg-1", "Gwyn's own thread name")
    adapter._session_origin_for_key = lambda _key: ("chan-1", None)
    adapter._auto_thread_for_session = lambda *_a: ("msg-1", "Gwyn's own thread name")
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session("agent:main:fluxer:thread:msg-1:u1", "Deploying")

    # Not label-guarded: the server's own name is not the guard. Unguarded is the
    # only safe answer once the written name is forgotten, so /title still lands.
    adapter.rename_thread.assert_awaited_once_with(
        "msg-1", "Deploying", only_if_current_name=None
    )


@pytest.mark.asyncio
async def test_slash_title_cannot_overwrite_a_human_thread_name():
    """/title runs after adoption and used the cached label as its guard -- which on
    the adoption path is the server's own name, so the guard passed by construction
    and a human's name was replaced."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    adapter._fetch_channel_name = AsyncMock(return_value="Gwyn's own thread name")
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message("chan-1", "msg-1")
    # The session is the thread this turn owns, so /title resolves to them.
    adapter._session_origin_for_key = lambda _key: ("chan-1", None)
    adapter._auto_thread_for_session = lambda *_a: ("msg-1", "Gwyn's own thread name")
    adapter._request = AsyncMock(return_value={})

    await adapter._rename_thread_for_session("agent:main:fluxer:thread:msg-1:u1", "Deploying")

    adapter._request.assert_not_awaited()  # no PATCH: the human's name stands


@pytest.mark.asyncio
async def test_slash_title_still_renames_a_thread_this_adapter_named():
    """The other direction: an untouched thread keeps taking /title."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: None
    adapter.create_thread = AsyncMock(return_value={"id": "msg-1", "name": "deploy the thing"})
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message("chan-1", "msg-1")
    adapter._session_origin_for_key = lambda _key: ("chan-1", None)
    adapter._auto_thread_for_session = lambda *_a: ("msg-1", "deploy the thing")
    calls: list = []

    async def fake_request(method, path, **kwargs):
        calls.append(method)
        return {}

    adapter._request = fake_request

    await adapter._rename_thread_for_session("agent:main:fluxer:thread:msg-1:u1", "Deploying")

    assert calls == ["PATCH"]


@pytest.mark.asyncio
async def test_an_adopted_thread_with_a_stale_placeholder_is_renamed():
    """A restart can land after the placeholder was written but before the rename, so
    the session title already exists when adoption runs. The failed create applies
    nothing, and later sends only reuse the cached thread without re-checking -- so
    adoption has to run the guarded rename even when a title is available."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: "Deploying the thing"
    adapter.create_thread = AsyncMock(side_effect=_FakeHttpError(400, THREAD_EXISTS_BODY))
    # The thread still carries the pre-restart placeholder, so the guard passes.
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert watched == [("msg-1", "msg-1", "deploy the thing", None)]


@pytest.mark.asyncio
async def test_a_created_thread_does_not_schedule_a_second_rename():
    """Only adoption needs the rename when a title already exists: a thread created
    here was born with the title, so scheduling again would be redundant work."""
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._session_title_for_chat = lambda *args, **kwargs: "Deploying the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "msg-1", "name": "Deploying the thing"})
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert watched == []


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


@pytest.mark.asyncio
async def test_the_lock_map_returns_to_its_bound_as_sends_finish():
    """While every candidate is busy the map overshoots its bound. Releasing the last
    mark must trim it back, without waiting for some later anchor to arrive."""
    adapter = _adapter()
    adapter._threads_by_anchor_max = 2
    keys = [("chan-1", f"msg-{index}") for index in range(5)]
    release = asyncio.Event()
    all_inside = [asyncio.Event() for _ in keys]

    async def hold(key, inside):
        async with adapter._thread_creation_slot(key):
            inside.set()
            await release.wait()

    tasks = [
        asyncio.create_task(hold(key, inside))
        for key, inside in zip(keys, all_inside)
    ]
    for inside in all_inside:  # a slot that never opens must fail, not hang
        await asyncio.wait_for(inside.wait(), timeout=2)

    # Every anchor held a slot at once, so nothing was evictable and the map
    # legitimately overshot its bound.
    assert len(adapter._thread_create_locks) == 5
    assert len(adapter._thread_create_locks) > adapter._threads_by_anchor_max

    release.set()
    await asyncio.gather(*tasks)

    assert adapter._thread_create_lock_users == {}
    assert len(adapter._thread_create_locks) <= adapter._threads_by_anchor_max


@pytest.mark.asyncio
async def test_a_lock_a_send_is_holding_is_never_evicted():
    """Eviction on age alone can drop the lock a send still holds, and the next send
    for that anchor then builds a second lock and POSTs a second thread -- the
    duplicate the lock exists to prevent."""
    adapter = _adapter()
    adapter._threads_by_anchor_max = 2
    held = ("chan-1", "msg-held")
    inside = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with adapter._thread_creation_slot(held):
            inside.set()
            await release.wait()

    holder = asyncio.create_task(hold())
    # A slot that never opens must fail the test, not hang it.
    await asyncio.wait_for(inside.wait(), timeout=2)
    original = adapter._thread_create_locks[held]

    for index in range(6):  # inserts well past the bound
        adapter._thread_create_lock(("chan-2", f"msg-{index}"))

    # The held anchor keeps its lock object, so a concurrent send for it queues on
    # the same lock instead of becoming a second creator.
    assert adapter._thread_create_lock(held) is original

    release.set()
    await holder
    assert adapter._thread_create_lock_users == {}  # the mark is released
    for index in range(6):  # now idle, the anchor is evictable again
        adapter._thread_create_lock(("chan-3", f"msg-{index}"))
    assert len(adapter._thread_create_locks) == adapter._threads_by_anchor_max


@pytest.mark.asyncio
async def test_a_lock_a_send_is_only_waiting_on_is_never_evicted():
    """A queued send pins its anchor's entry as well: it holds the old lock object,
    so replacing the entry would put two creators on one anchor."""
    adapter = _adapter()
    adapter._threads_by_anchor_max = 1
    key = ("chan-1", "msg-1")
    first_inside = asyncio.Event()
    release = asyncio.Event()

    async def send(marker):
        async with adapter._thread_creation_slot(key):
            if marker == 1:
                first_inside.set()
                await release.wait()

    first = asyncio.create_task(send(1))
    await asyncio.wait_for(first_inside.wait(), timeout=2)
    second = asyncio.create_task(send(2))
    for _ in range(5):  # let the second reach the lock
        await asyncio.sleep(0)
    original = adapter._thread_create_locks[key]
    assert adapter._thread_create_lock_users[key] == 2

    for index in range(4):
        adapter._thread_create_lock(("chan-9", f"msg-{index}"))

    assert adapter._thread_create_locks.get(key) is original
    assert adapter._thread_create_lock(key) is original

    release.set()
    await asyncio.gather(first, second)
    assert adapter._thread_create_lock_users == {}
