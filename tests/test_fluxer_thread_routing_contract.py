"""One routing contract for a thread: session, admission, typing and delivery.

The live failure this pins down: a turn that starts in a parent channel and
delivers into a thread it opened was keyed as the *channel* session, while a
follow-up typed inside that thread was keyed as the *thread* session. ``/stop``
typed in the thread therefore cancelled a session with no run, and the typing
indicator refreshed the parent channel the whole time.

Fluxer anchors a thread to the message it was started from and gives the thread
that message's id, so the message that opens a thread already names the container
it will deliver into. The host has a contract for exactly this shape
(``build_session_key`` keys a thread on ``chat_id`` with no thread slot when the
platform models a thread as its own chat container — Discord's in-thread sources
carry ``chat_id == thread_id``). Fluxer's own already-persisted thread sessions
look like ``agent:main:fluxer:thread:<anchor-message-id>:<user-id>``, i.e. the
thread id in the ``chat_id`` slot and NO thread slot. This module pins that shape
because it is the one that keeps per-user isolation: the host drops the
participant slot from a key that carries a real ``thread_id`` unless
``thread_sessions_per_user`` is on, and this deployment runs the default (off).

The regression below runs the *host's own* ``build_session_key`` so the equality
is a property of the real key derivation, not of a local reimplementation.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

# The host's real key derivation is the point of the session tests. The standalone
# plugin harness only stubs ``gateway.platforms.base``, so this import succeeds
# against a Hermes checkout and skips in the bare plugin CI.
session_module = pytest.importorskip("gateway.session")
build_session_key = session_module.build_session_key

PARENT = "chan-1"
ANCHOR = "msg-anchor-1"  # Fluxer: the thread's id IS its anchor message's id
GUILD = "guild-1"
OWNER = "owner-user"
OTHER = "other-user"
BOT = "bot-1"

THREAD_KEY = "agent:main:fluxer:thread:msg-anchor-1:owner-user"


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
    data = {
        "id": message_id,
        "channel_id": channel_id,
        "content": content,
        "author": {"id": OWNER, "username": "Gwyn", "bot": False},
    }
    data.update(overrides)
    return data


def _key(source, *, thread_sessions_per_user: bool = False) -> str:
    """The host's key for a source, at the deployment's default isolation settings."""
    return build_session_key(
        source,
        group_sessions_per_user=True,
        thread_sessions_per_user=thread_sessions_per_user,
    )


async def _capture_source(adapter, data) -> object:
    """Run one inbound message through the adapter and return the source it built."""
    adapter.handle_message = AsyncMock()
    await adapter._handle_message_create(data, {"op": 0, "t": "MESSAGE_CREATE", "d": data})
    return adapter.handle_message.await_args.args[0].source


def _bot_thread(adapter) -> None:
    """The API says this thread is the bot's own, under PARENT."""
    adapter._fetch_thread_info = AsyncMock(
        return_value={"parent_id": PARENT, "owner_id": BOT}
    )


# ── session continuity: /stop in the thread must target the running turn ────────


@pytest.mark.asyncio
@pytest.mark.parametrize("thread_sessions_per_user", [False, True])
async def test_an_in_thread_turn_shares_the_session_key_of_the_turn_that_opened_it(
    thread_sessions_per_user,
):
    """The reported bug, at the level of the host's key derivation.

    A channel message that will open a thread, and the follow-up typed inside
    that thread, must resolve to ONE session key — otherwise ``/stop`` typed in
    the thread cancels a session that has no run while the real turn keeps going.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    initiating = await _capture_source(adapter, _message(ANCHOR, PARENT, "deploy the thing"))
    follow_up = await _capture_source(
        adapter,
        _message("msg-2", ANCHOR, "carry on", channel_type=11),
    )

    assert _key(initiating, thread_sessions_per_user=thread_sessions_per_user) == _key(
        follow_up, thread_sessions_per_user=thread_sessions_per_user
    )


@pytest.mark.asyncio
async def test_a_command_typed_in_the_thread_targets_the_running_turn():
    """/stop is the reason the two keys have to agree."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    initiating = await _capture_source(adapter, _message(ANCHOR, PARENT, "deploy the thing"))
    stop = await _capture_source(adapter, _message("msg-2", ANCHOR, "/stop", channel_type=11))

    assert _key(stop) == _key(initiating)


@pytest.mark.asyncio
async def test_a_command_typed_in_the_channel_keeps_the_channel_session():
    """A command addresses the running turn; it must not open a thread of its own.

    Auto-threading a ``/stop`` would key it on the command's own message id — a
    brand-new session with no run — so the stop would never reach the turn.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])

    source = await _capture_source(adapter, _message("msg-2", PARENT, "/stop"))

    assert source.chat_id == PARENT
    assert source.chat_type == "channel"
    assert _key(source) == "agent:main:fluxer:channel:chan-1:owner-user"
    assert adapter._threads_pending == {}


# ── session normalization ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_an_in_thread_message_carries_its_parent_and_thread_id():
    """The follow-up must present itself as the thread it arrived in."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    source = await _capture_source(adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11))

    assert source.chat_id == ANCHOR
    assert source.chat_type == "thread"
    assert source.thread_id is None
    assert source.parent_chat_id == PARENT
    assert _key(source) == THREAD_KEY


@pytest.mark.asyncio
async def test_the_initiating_message_names_the_thread_it_will_open():
    """The channel message already names the container its reply will live in."""
    adapter = _adapter(thread_reply_channels=[PARENT])

    source = await _capture_source(adapter, _message(ANCHOR, PARENT, "deploy the thing"))

    assert source.chat_id == ANCHOR
    assert source.chat_type == "thread"
    assert source.parent_chat_id == PARENT
    assert _key(source) == THREAD_KEY
    # The thread does not exist on the server yet; delivery creates it first.
    assert adapter._threads_pending[ANCHOR] == PARENT


@pytest.mark.asyncio
async def test_a_channel_without_thread_replies_keeps_its_channel_session():
    adapter = _adapter()

    source = await _capture_source(adapter, _message(ANCHOR, PARENT, "deploy the thing"))

    assert source.chat_id == PARENT
    assert source.chat_type == "channel"
    assert _key(source) == "agent:main:fluxer:channel:chan-1:owner-user"
    assert adapter._threads_pending == {}


@pytest.mark.asyncio
async def test_a_direct_message_never_opens_a_thread():
    """DMs have no threads; a prospective id there would only split the DM session."""
    adapter = _adapter(thread_replies=True)

    source = await _capture_source(adapter, _message(ANCHOR, PARENT, "hello", channel_type=1))

    assert source.chat_type == "dm"
    assert source.chat_id == PARENT
    assert adapter._threads_pending == {}


@pytest.mark.asyncio
async def test_two_users_in_one_thread_keep_separate_sessions():
    """Per-user isolation must survive the fix: no silent shared-thread session."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    first = await _capture_source(adapter, _message("msg-2", ANCHOR, "hi", channel_type=11))
    second = await _capture_source(
        adapter,
        _message(
            "msg-3", ANCHOR, "hi", channel_type=11,
            author={"id": OTHER, "username": "Rowan", "bot": False},
        ),
    )

    assert _key(first) == THREAD_KEY
    assert _key(second) == "agent:main:fluxer:thread:msg-anchor-1:other-user"


@pytest.mark.asyncio
async def test_a_foreign_thread_keeps_its_own_session():
    """A thread a human created must not be merged into the parent channel's session."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._fetch_thread_info = AsyncMock(
        return_value={"parent_id": PARENT, "owner_id": "someone-else"}
    )

    source = await _capture_source(adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11))

    assert source.chat_id == ANCHOR
    assert source.chat_type == "thread"
    assert source.parent_chat_id == PARENT
    assert _key(source) == THREAD_KEY


@pytest.mark.asyncio
async def test_the_resolver_reads_a_cold_thread_once_and_remembers_it():
    """After a restart nothing is in memory, so the parent comes from the API — once."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    await _capture_source(adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11))
    await _capture_source(adapter, _message("msg-3", ANCHOR, "again", channel_type=11))

    adapter._fetch_thread_info.assert_awaited_once()
    assert adapter._thread_parents[ANCHOR] == PARENT


@pytest.mark.asyncio
async def test_an_unresolvable_thread_is_left_exactly_as_it_was():
    """A lookup failure must not invent a parent, and must not fail the message."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._fetch_thread_info = AsyncMock(side_effect=RuntimeError("offline"))

    source = await _capture_source(adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11))

    assert source.chat_id == ANCHOR
    assert source.chat_type == "thread"
    assert source.parent_chat_id is None


# ── admission: a bot-owned thread stays reachable after a restart ───────────────


@pytest.mark.asyncio
async def test_a_follow_up_in_the_bots_thread_is_admitted_after_a_restart():
    """Cold maps must not turn the bot's own thread into a silent drop zone."""
    adapter = _adapter(thread_reply_channels=[PARENT], require_mention=True)
    _bot_thread(adapter)
    adapter._free_response_channels = {PARENT}

    source = await _capture_source(adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11))

    assert source is not None  # admitted, not dropped by the mention gate


def test_a_foreign_thread_does_not_inherit_the_parent_channel():
    """Inheritance is for containers the adapter owns, not for any thread."""
    adapter = _adapter(require_mention=True, free_response_channels=[PARENT])
    adapter._thread_parents[ANCHOR] = PARENT
    adapter._thread_owners[ANCHOR] = False  # a human created it

    allowed, _ = adapter._should_process_message(
        channel_id=ANCHOR,
        chat_type="thread",
        text="no mention here",
        data={"channel_id": ANCHOR},
        reply_to_message_id=None,
    )

    assert allowed is False


# ── typing: the indicator belongs to the container the turn delivers into ──────


async def _drain_loop(ticks: int = 20) -> None:
    for _ in range(ticks):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_typing_lands_in_the_thread_the_turn_will_use():
    """The channel-origin turn's indicator must refresh the thread, not the channel."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._last_inbound_by_chat[PARENT] = ANCHOR
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if path.endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        return {}

    adapter._request = fake_request
    adapter._schedule_title_rename = lambda *args: None

    await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
    await _drain_loop()

    assert ("POST", f"/channels/{ANCHOR}/typing") in calls
    assert ("POST", f"/channels/{PARENT}/typing") not in calls

    await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})


@pytest.mark.asyncio
async def test_typing_stays_in_the_channel_without_thread_replies():
    adapter = _adapter()
    adapter._last_inbound_by_chat[PARENT] = ANCHOR
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request

    await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
    await _drain_loop()

    assert ("POST", f"/channels/{PARENT}/typing") in calls
    assert not any(path.endswith("/threads") for _, path in calls)

    await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})


@pytest.mark.asyncio
async def test_typing_uses_a_thread_the_turn_already_adopted():
    """A thread opened earlier in the turn (e.g. by a progress send) is reused."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._last_inbound_by_chat[PARENT] = ANCHOR
    adapter._threads_by_anchor[(PARENT, ANCHOR)] = "thread-existing"
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        return {}

    adapter._request = fake_request

    await adapter.send_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})
    await _drain_loop()

    assert ("POST", "/channels/thread-existing/typing") in calls
    assert not any(path.endswith("/threads") for _, path in calls)

    await adapter.stop_typing(PARENT, metadata={"reply_to_message_id": ANCHOR})


@pytest.mark.asyncio
async def test_typing_in_a_pending_thread_creates_it_under_the_parent():
    """The thread does not exist yet, so the indicator must open it first."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    calls: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if path.endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        return {}

    adapter._request = fake_request
    adapter._schedule_title_rename = lambda *args: None

    await adapter.send_typing(ANCHOR)
    await _drain_loop()

    assert ("POST", f"/channels/{PARENT}/messages/{ANCHOR}/threads") in calls
    assert ("POST", f"/channels/{ANCHOR}/typing") in calls

    await adapter.stop_typing(ANCHOR)


# ── delivery rides the same resolver ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_pending_thread_is_created_before_the_first_delivery():
    """The reply must land in the thread the session was keyed on."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._schedule_title_rename = lambda *args: None
    posts: list[tuple[str, str]] = []

    async def fake_request(method, path, **kwargs):
        posts.append((method, path))
        if path.endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    result = await adapter.send(ANCHOR, "the answer")

    assert result.success is True
    assert ("POST", f"/channels/{PARENT}/messages/{ANCHOR}/threads") in posts
    assert ("POST", f"/channels/{ANCHOR}/messages") in posts
    assert ("POST", f"/channels/{PARENT}/messages") not in posts


@pytest.mark.asyncio
async def test_a_deleted_pending_thread_falls_back_to_the_parent_channel():
    """A thread removed between keying and delivery must not lose the reply.

    The caller's chat_id IS the prospective thread, so the fallback has to name the
    parent channel recorded when the message was admitted -- posting to chat_id
    would re-address the missing thread and drop the reply.
    """
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._threads_pending[ANCHOR] = PARENT
    adapter._schedule_title_rename = lambda *args: None
    posts: list[str] = []

    async def fake_request(method, path, **kwargs):
        if path.endswith("/threads"):
            return {"id": ANCHOR, "name": "deploy the thing"}
        posts.append(path)
        if path.startswith(f"/channels/{ANCHOR}/"):
            request = httpx.Request("POST", f"https://example.invalid{path}")
            response = httpx.Response(404, request=request, text='{"code":"UNKNOWN_CHANNEL"}')
            raise httpx.HTTPStatusError("404 Not Found", request=request, response=response)
        return {"id": "reply-1"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    result = await adapter.send(ANCHOR, "the answer")

    assert result.success is True
    assert posts[-1] == f"/channels/{PARENT}/messages"


# ── approvals ride the same resolver ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_a_slash_confirm_routes_to_the_turns_thread():
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "confirm-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    result = await adapter.send_slash_confirm(
        PARENT, "Reset?", "Reset the session?", THREAD_KEY, "confirm-xyz",
    )

    assert result.success is True
    assert adapter._post_message_with_optional_components.await_args.args[0] == ANCHOR


@pytest.mark.asyncio
async def test_a_channel_keyed_approval_stays_in_the_channel():
    """A non-thread session must not be redirected into some thread."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "approval-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(
        PARENT, "ls", "agent:main:fluxer:channel:chan-1:owner-user",
    )

    assert adapter._post_message_with_optional_components.await_args.args[0] == PARENT


def test_a_thread_session_key_names_its_container():
    assert (
        fluxer_adapter._thread_id_from_session_key(THREAD_KEY, "fluxer") == ANCHOR
    )
    assert (
        fluxer_adapter._thread_id_from_session_key(
            "agent:main:fluxer:channel:chan-1:owner-user", "fluxer"
        )
        is None
    )
    assert (
        fluxer_adapter._thread_id_from_session_key(
            "agent:main:discord:thread:999:user", "fluxer"
        )
        is None
    )
    assert fluxer_adapter._thread_id_from_session_key(None, "fluxer") is None


# ── a deleted thread must not wedge the routing state ───────────────────────────


@pytest.mark.asyncio
async def test_forgetting_a_thread_clears_its_resolved_identity():
    """A reused anchor must re-resolve rather than inherit a stale parent."""
    adapter = _adapter(thread_reply_channels=[PARENT])
    _bot_thread(adapter)

    await _capture_source(adapter, _message("msg-2", ANCHOR, "hi", channel_type=11))
    assert adapter._thread_parents[ANCHOR] == PARENT

    adapter._forget_thread(ANCHOR)

    assert ANCHOR not in adapter._thread_parents
    assert ANCHOR not in adapter._thread_owners
    assert ANCHOR not in adapter._thread_info_cache


@pytest.mark.asyncio
async def test_a_thread_message_without_a_guild_id_is_still_admitted():
    """A thread event may omit the guild; that must not drop a bot-owned follow-up."""
    adapter = _adapter(thread_reply_channels=[PARENT], require_mention=True)
    _bot_thread(adapter)

    source = await _capture_source(
        adapter, _message("msg-2", ANCHOR, "carry on", channel_type=11)
    )

    assert source is not None
    assert _key(source) == THREAD_KEY


@pytest.mark.asyncio
async def test_two_channel_users_opening_threads_get_separate_sessions():
    """Two channel messages must key on their OWN anchors, not one shared thread."""
    adapter = _adapter(thread_reply_channels=[PARENT])

    first = await _capture_source(adapter, _message("anchor-a", PARENT, "one"))
    second = await _capture_source(
        adapter,
        _message(
            "anchor-b", PARENT, "two",
            author={"id": OTHER, "username": "Rowan", "bot": False},
        ),
    )

    assert first.chat_id == "anchor-a"
    assert second.chat_id == "anchor-b"
    assert _key(first) == "agent:main:fluxer:thread:anchor-a:owner-user"
    assert _key(second) == "agent:main:fluxer:thread:anchor-b:other-user"


@pytest.mark.asyncio
async def test_a_rejected_message_never_leaves_a_pending_thread():
    """A mention-gated channel message must not register a thread that never runs."""
    adapter = _adapter(thread_reply_channels=[PARENT], require_mention=True)

    adapter.handle_message = AsyncMock()
    await adapter._handle_message_create(
        _message("anchor-gated", PARENT, "no mention here"),
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert adapter.handle_message.await_count == 0
    assert adapter._threads_pending == {}


@pytest.mark.asyncio
async def test_an_approval_prompt_routes_to_the_turns_thread():
    adapter = _adapter(thread_reply_channels=[PARENT])
    adapter._record_auto_thread(PARENT, ANCHOR, "deploy the thing", OWNER)

    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "approval-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    result = await adapter.send_exec_approval(
        PARENT, "rm -rf /", THREAD_KEY,
        description="dangerous",
    )

    assert result.success is True
    assert adapter._post_message_with_optional_components.await_args.args[0] == ANCHOR
    assert adapter._add_reaction.await_args.args[0] == ANCHOR
