"""Thread routing without the host runtime: the transport half of the contract.

The companion module ``test_fluxer_thread_routing_contract.py`` runs the *host's*
``build_session_key`` to pin session-key equality, and therefore skips wholesale
in the standalone plugin CI (no ``gateway.session``). That left the routing
behaviour itself — which container a typing indicator refreshes, where a reply is
delivered, what a deleted thread falls back to, and how a raw event is classified
— with no coverage in this repository's own CI.

This module needs only the gateway config/platform stubs ``conftest.py`` already
provides, so it runs everywhere. Each test drives the adapter's real inbound entry
point (``_handle_message_create``) or its real public send surface, and asserts on
the REST calls the adapter actually makes.

The failures these pin down, all seen on the live deployment:

* Fluxer's own ``MESSAGE_CREATE`` payload carries no ``channel_type`` and no
  ``channel`` object — only ``channel_id``. A follow-up typed inside a thread was
  therefore classified as a *channel* message, so the thread-keyed session the
  fix exists to produce never happened and the indicator refreshed the parent.
* A plain-text command (``/stop``) in a thread-enabled channel had a thread
  opened on the command's own message id, by the typing indicator at turn start
  and again by the reply. Live proof: an active thread named ``/stop`` under the
  channel it was typed in.
* Once the typing indicator opened the prospective thread, the outbound resolver
  lost the pending entry, so a thread deleted mid-turn fell through to a failed
  send instead of the parent channel the changelog promises.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

PARENT = "chan-1"
ANCHOR = "msg-anchor-1"  # Fluxer: the thread's id IS its anchor message's id
GUILD = "guild-1"
OWNER = "owner-user"
OTHER = "other-user"
BOT = "bot-1"

THREAD_KEY = f"agent:main:fluxer:thread:{ANCHOR}:{OWNER}"


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "require_mention": False,
        "delivery_verification": False,
    }
    options.update(extra)
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))
    adapter.bot_user_id = BOT
    return adapter


def _message(message_id: str, channel_id: str, content: str = "carry on", **overrides):
    """A Fluxer MESSAGE_CREATE payload, shaped like the real one.

    The real payload has no ``channel_type`` and no ``channel`` object (verified
    against the live API), so tests must not need either unless they are
    explicitly exercising an event that carries one.
    """
    data = {
        "id": message_id,
        "channel_id": channel_id,
        "content": content,
        "author": {"id": OWNER, "username": "Gwyn", "bot": False},
    }
    data.update(overrides)
    return data


async def _drive_turn(adapter, data, turn):
    """Run one inbound message through the adapter, with ``turn`` as the turn body.

    ``handle_message`` is replaced by the adapter's own six-line turn-context
    wrapper, so the tests exercise the real ``_handle_message_create`` (and thus
    the real resolver and pending registration) without needing the host's base
    class, which the standalone CI stubs.
    """
    async def fake_handle(event):
        marker = adapter._turn_context.set((event, ""))
        try:
            await turn(event)
        finally:
            adapter._turn_context.reset(marker)

    adapter.handle_message = fake_handle
    await adapter._handle_message_create(data, {"op": 0, "t": "MESSAGE_CREATE", "d": data})


async def _drain_loop(ticks: int = 20) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


def _thread_starts(calls) -> list[tuple]:
    return [c.args for c in calls if str(c.args[1]).endswith("/threads")]


def _not_found(path: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"https://example.invalid{path}")
    response = httpx.Response(404, request=request, text='{"code":"UNKNOWN_CHANNEL"}')
    return httpx.HTTPStatusError("404 Not Found", request=request, response=response)


# ── a raw Fluxer event must still be classified ─────────────────────────────────


@pytest.mark.asyncio
async def test_an_untyped_event_inside_a_known_thread_is_still_the_thread():
    """Fluxer sends ``channel_id`` with no ``channel_type``; the thread must survive it.

    Without this the follow-up is keyed as a *channel* session, which is exactly
    the split the routing fix exists to remove: the thread the bot opened and the
    follow-up typed inside it resolve to different sessions.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._thread_parents[ANCHOR] = PARENT
    adapter._remember_thread_owner(ANCHOR, True)
    seen = {}

    async def turn(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type
        seen["parent"] = getattr(event.source, "parent_chat_id", None)

    # No channel_type, no channel object: the real payload shape.
    await _drive_turn(adapter, _message("msg-2", ANCHOR, "carry on"), turn)

    assert seen["chat_id"] == ANCHOR
    assert seen["chat_type"] == "thread"
    assert seen["parent"] == PARENT


@pytest.mark.asyncio
async def test_an_untyped_event_in_an_unknown_thread_is_classified_once():
    """A cold process must learn a thread's kind from the API, not guess "channel"."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._fetch_thread_info = AsyncMock(
        return_value={"type": 11, "parent_id": PARENT, "owner_id": BOT}
    )
    seen = {}

    async def turn(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type
        seen["parent"] = getattr(event.source, "parent_chat_id", None)

    await _drive_turn(adapter, _message("msg-2", ANCHOR, "carry on"), turn)
    await _drive_turn(adapter, _message("msg-3", ANCHOR, "again"), turn)

    assert seen["chat_type"] == "thread"
    assert seen["parent"] == PARENT
    adapter._fetch_thread_info.assert_awaited_once()


@pytest.mark.asyncio
async def test_an_untyped_event_in_a_text_channel_stays_a_channel():
    """A text channel must not be mistaken for a thread just because it is untyped."""
    adapter = _adapter(thread_replies=True)
    adapter._fetch_thread_info = AsyncMock(return_value={"type": 0, "parent_id": "category-1"})
    seen = {}

    async def turn(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type
        seen["parent"] = getattr(event.source, "parent_chat_id", None)

    await _drive_turn(adapter, _message(ANCHOR, PARENT, "deploy the thing"), turn)

    # Thread replies apply, so the channel message becomes the prospective thread
    # named by its own anchor -- the same result an explicitly typed channel event
    # gets. What matters is that the kind was not read as "thread" for PARENT.
    assert seen["chat_id"] == ANCHOR
    assert seen["chat_type"] == "thread"
    assert seen["parent"] == PARENT
    assert adapter._threads_pending == {ANCHOR: PARENT}
    # A category is not a parent channel: nothing was inherited from it.
    assert adapter._thread_parents == {}
    assert adapter._channel_kinds[PARENT] == "channel"


@pytest.mark.asyncio
async def test_an_untyped_event_in_a_thread_enabled_channel_keeps_its_channel_session():
    """A text command in a thread-enabled channel stays the channel's session."""
    adapter = _adapter(thread_replies=True)
    adapter._fetch_thread_info = AsyncMock(return_value={"type": 0, "parent_id": "category-1"})
    seen = {}

    async def turn(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type

    await _drive_turn(adapter, _message("msg-stop", PARENT, "/stop"), turn)

    assert seen["chat_id"] == PARENT
    assert seen["chat_type"] == "channel"
    assert adapter._threads_pending == {}


@pytest.mark.asyncio
async def test_a_failed_kind_lookup_leaves_the_event_as_it_arrived():
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._fetch_thread_info = AsyncMock(side_effect=RuntimeError("offline"))
    seen = {}

    async def turn(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type

    await _drive_turn(adapter, _message("msg-2", ANCHOR, "carry on"), turn)

    assert seen["chat_id"] == ANCHOR
    assert seen["chat_type"] == "channel"  # unchanged behaviour, no invented parent


# ── a plain-text command must not open a thread of its own ──────────────────────


@pytest.mark.asyncio
async def test_a_command_turn_never_opens_a_thread_with_its_typing_indicator():
    """``/stop`` in a thread-enabled channel opened a thread on its own message.

    The live deployment carries the artefact: an active thread named ``/stop``
    under the channel it was typed in, created by the typing indicator the turn
    start fires.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._schedule_title_rename = lambda *args: None
    calls: list[tuple] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"id": ANCHOR, "name": "stop"}

    adapter._request = fake_request

    async def turn(event):
        await adapter.send_typing(event.source.chat_id)
        await _drain_loop()

    await _drive_turn(adapter, _message("msg-stop", PARENT, "/stop"), turn)

    assert ("POST", f"/channels/{PARENT}/typing") in calls
    assert not any(str(path).endswith("/threads") for _, path in calls)


@pytest.mark.asyncio
async def test_a_command_turn_reply_stays_in_the_channel():
    """Auto-threading a command's reply keys it on the command's own message id."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._schedule_title_rename = lambda *args: None
    calls: list[tuple] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    async def turn(event):
        result = await adapter.send(event.source.chat_id, "Stopped.")
        assert result.success is True

    await _drive_turn(adapter, _message("msg-stop", PARENT, "/stop"), turn)

    assert ("POST", f"/channels/{PARENT}/messages") in calls
    assert not any(str(path).endswith("/threads") for _, path in calls)


@pytest.mark.asyncio
async def test_a_command_typed_inside_a_thread_stays_in_that_thread():
    """A command in a thread addresses that thread's turn; it must not leak to the parent."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._thread_parents[ANCHOR] = PARENT
    adapter._remember_thread_owner(ANCHOR, True)
    adapter._schedule_title_rename = lambda *args: None
    calls: list[tuple] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    async def turn(event):
        await adapter.send_typing(event.source.chat_id)
        await _drain_loop()
        await adapter.send(event.source.chat_id, "Stopped.")

    await _drive_turn(adapter, _message("msg-stop", ANCHOR, "/stop"), turn)

    assert ("POST", f"/channels/{ANCHOR}/messages") in calls
    assert ("POST", f"/channels/{ANCHOR}/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls


# ── typing honours an explicit container ────────────────────────────────────────


@pytest.mark.asyncio
async def test_typing_honours_an_explicit_thread_id():
    """An explicit ``metadata['thread_id']`` wins over the caller's chat id.

    The approval and delivery paths already read it first; the indicator did not,
    so a caller that names the container got the channel refreshed instead.
    """
    adapter = _adapter()
    calls: list[tuple] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request

    await adapter.send_typing(PARENT, metadata={"thread_id": "thread-explicit"})
    await _drain_loop()

    assert ("POST", "/channels/thread-explicit/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls

    await adapter.stop_typing(PARENT, metadata={"thread_id": "thread-explicit"})


# ── a deleted thread must still fall back to the parent channel ─────────────────


@pytest.mark.asyncio
async def test_a_thread_opened_by_typing_can_still_fall_back_when_deleted():
    """The typing indicator opens the prospective thread; a later 404 must still recover.

    Opening it consumes the pending entry, so the outbound resolver no longer
    recognises the container as a thread it chose and the deleted-thread fallback
    never fires: the reply is dropped instead of landing in the parent channel.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._schedule_title_rename = lambda *args: None
    posts: list[str] = []

    async def fake_request(method, path, **kwargs):
        if str(path).endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        posts.append(path)
        if path.startswith(f"/channels/{ANCHOR}/"):
            raise _not_found(path)
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    await adapter.send_typing(ANCHOR)
    await _drain_loop()
    result = await adapter.send(ANCHOR, "the answer")

    assert result.success is True, result.error
    assert posts[-1] == f"/channels/{PARENT}/messages"


@pytest.mark.asyncio
async def test_the_deleted_thread_fallback_sticks_for_the_rest_of_the_turn():
    """A second send in the same turn must not re-address the dead thread."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._schedule_title_rename = lambda *args: None
    posts: list[str] = []

    async def fake_request(method, path, **kwargs):
        if str(path).endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        posts.append(path)
        if path.startswith(f"/channels/{ANCHOR}/"):
            raise _not_found(path)
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    await adapter.send_typing(ANCHOR)
    await _drain_loop()
    await adapter.send(ANCHOR, "the answer", reply_to=ANCHOR)
    posts.clear()
    await adapter.send(ANCHOR, "a trailing note", reply_to="someone-else")

    assert posts == [f"/channels/{PARENT}/messages"]


@pytest.mark.asyncio
async def test_an_upload_falls_back_to_the_parent_of_a_deleted_thread():
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._schedule_title_rename = lambda *args: None
    uploads: list[str] = []

    async def fake_request(method, path, **kwargs):
        if str(path).endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        return {"id": "reply-1"}

    async def fake_multipart(method, path, **kwargs):
        uploads.append(path)
        if path.startswith(f"/channels/{ANCHOR}/"):
            raise _not_found(path)
        return {"id": "upload-1"}

    adapter._request = fake_request
    adapter._multipart_request = fake_multipart
    adapter._verify_delivery = AsyncMock(return_value=None)
    document = _temp_file("notes.txt", "hello")

    await adapter.send_typing(ANCHOR)
    await _drain_loop()
    result = await adapter.send_document(ANCHOR, str(document))

    assert result.success is True, result.error
    assert uploads[-1] == f"/channels/{PARENT}/messages"


# ── approvals must not be posted into a thread that was never created ───────────


@pytest.mark.asyncio
async def test_an_approval_for_an_uncreated_thread_goes_to_its_parent():
    """A failed thread creation must not aim the prompt at a channel that is not there.

    The resolver returned the prospective thread id regardless, so the approval
    prompt 404'd and the waiting turn could never be answered.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._request = AsyncMock(return_value={})  # thread creation fails, no id
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "approval-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", THREAD_KEY, description="dangerous")

    assert adapter._post_message_with_optional_components.await_args.args[0] == PARENT


@pytest.mark.asyncio
async def test_an_approval_for_an_existing_thread_stays_in_it():
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._thread_parents[ANCHOR] = PARENT
    adapter._remember_thread_owner(ANCHOR, True)
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "approval-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", THREAD_KEY, description="dangerous")

    assert adapter._post_message_with_optional_components.await_args.args[0] == ANCHOR


# ── an unknown owner must not inherit the parent's allowlist ────────────────────


@pytest.mark.asyncio
async def test_a_thread_whose_owner_cannot_be_established_inherits_nothing():
    """Ownership unknown means unknown: the parent's allowlist must not leak in.

    The parent is still recorded -- the host reads ``parent_chat_id`` for profile
    routing and thread branching -- but the *inheritance* of the parent's
    allowlist and free-response status is a separate decision that only a
    confirmed bot-owned thread gets.
    """
    adapter = _adapter(allow_all_users=True, require_mention=True)
    adapter.bot_user_id = None  # ownership cannot be decided yet
    adapter._fetch_thread_info = AsyncMock(return_value={"type": 11, "parent_id": PARENT})
    adapter._allowed_channel_ids = {PARENT}

    await adapter._resolve_thread_info(ANCHOR, expect_thread=True)

    # The parent is named for routing...
    assert adapter._thread_parents.get(ANCHOR) == PARENT
    # ...but the gate still refuses the thread, because ownership is unknown.
    allowed, _ = adapter._should_process_message(
        channel_id=ANCHOR,
        chat_type="thread",
        text="hello there",
        data={"channel_id": ANCHOR},
        reply_to_message_id=None,
    )
    assert allowed is False


@pytest.mark.asyncio
async def test_an_unknown_owner_thread_is_rejected_by_the_parents_allowlist():
    """A parent recorded without a confirmed owner must not admit the thread.

    The parent map can hold an entry whose ownership was never established (no
    bot id yet, or the owner field absent from the API read). Treating that as
    inheritance let any thread under an allowlisted channel through the gate.
    """
    adapter = _adapter(allow_all_users=True, require_mention=True)
    adapter.bot_user_id = None
    adapter._allowed_channel_ids = {PARENT}
    # The parent is present but ownership was never established.
    adapter._thread_parents[ANCHOR] = PARENT

    allowed, _ = adapter._should_process_message(
        channel_id=ANCHOR,
        chat_type="thread",
        text="hello there",
        data={"channel_id": ANCHOR},
        reply_to_message_id=None,
    )

    assert allowed is False


# ── the native interaction path is normalized like any other message ────────────


@pytest.mark.asyncio
async def test_a_native_command_is_routed_through_the_same_resolver():
    """A native command must not key a different container than the message path.

    Fluxer registers no native application commands on this deployment, so the
    path is cold — but it built its own source with no parent and no normalization,
    which is how a second, divergent routing contract grows.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    seen = {}

    async def fake_handle(event):
        seen["chat_id"] = event.source.chat_id
        seen["chat_type"] = event.source.chat_type

    adapter.handle_message = fake_handle
    adapter._request = AsyncMock(return_value={})

    await adapter._handle_application_command_interaction(
        {
            "id": "interaction-1",
            "token": "tok",
            "channel_id": PARENT,
            "user": {"id": OWNER, "username": "Gwyn"},
            "data": {"name": "model", "options": []},
        }
    )

    assert seen["chat_id"] == PARENT
    assert seen["chat_type"] == "channel"
    # A command never registers a prospective thread for its own reply.
    assert adapter._threads_pending == {}


def _temp_file(name: str, content: str):
    import pathlib
    import tempfile

    path = pathlib.Path(tempfile.mkdtemp(prefix="fluxer-routing-")) / name
    path.write_text(content)
    return path
