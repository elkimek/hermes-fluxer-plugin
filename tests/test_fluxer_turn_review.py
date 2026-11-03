"""Boundary regressions from the stacked thread/session lifecycle review."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType

PARENT = "review-parent"
THREAD = "review-anchor"
USER = "review-user"
KEY = f"agent:main:fluxer:thread:{THREAD}:{USER}"


def make_adapter():
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "allow_all_users": True,
        "require_mention": False, "gateway_state_updates": False,
    }))
    adapter._request = AsyncMock(return_value={"id": "review-output"})
    adapter._session_title_for_chat = lambda *args: None
    adapter._schedule_title_rename = lambda *args: None
    return adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("control", ["status", "pause", "clear"])
async def test_saved_goal_controls_do_not_target_an_unrelated_live_turn(control):
    adapter = make_adapter()
    adapter._remember_goal_thread(PARENT, USER, THREAD)
    adapter._active_auto_threads[(PARENT, USER)] = {"unrelated-thread": 1}
    target = await adapter._resolve_thread_source({
        "id": "control-anchor", "content": f"/goal {control}", "author": {"id": USER},
    }, PARENT, "channel")
    assert target[:2] == (THREAD, "thread")
    if control != "clear":
        assert adapter._goal_thread_for(PARENT, USER) == THREAD


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["\t", "\n", "\u2003"])
@pytest.mark.parametrize("argument", ["pause", "repair the watcher"])
async def test_goal_argument_whitespace_uses_the_host_command_grammar(separator, argument):
    adapter = make_adapter()
    adapter._remember_goal_thread(PARENT, USER, THREAD)
    content = f"/goal{separator}{argument}"
    target = await adapter._resolve_thread_source({
        "id": "new-goal-anchor", "content": content, "author": {"id": USER},
    }, PARENT, "channel")
    expected = THREAD if argument == "pause" else "new-goal-anchor"
    assert target[:2] == (expected, "thread")


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", ["pause the watcher safely", "status of the build", "clear stale tasks", "done with cleanup after tests"])
async def test_goal_specs_beginning_with_control_words_still_start_their_own_session(spec):
    adapter = make_adapter()
    adapter._remember_goal_thread(PARENT, USER, THREAD)
    target = await adapter._resolve_thread_source({
        "id": "new-goal-anchor", "content": f"/goal {spec}", "author": {"id": USER},
    }, PARENT, "channel")
    assert target == ("new-goal-anchor", "thread", PARENT, PARENT)
    assert not fluxer_adapter._goal_control_ends_goal(f"/goal {spec}")


@pytest.mark.asyncio
async def test_explicit_thread_overrides_a_previous_chat_redirect_for_text_and_typing():
    adapter = make_adapter()
    adapter._remember_dead_container(THREAD, PARENT)
    metadata = {"thread_id": "explicit-thread"}
    assert await adapter._typing_target(THREAD, metadata) == "explicit-thread"
    result = await adapter.send(THREAD, "explicit delivery", metadata=metadata)
    assert result.success
    assert adapter._request.await_args.args[1] == "/channels/explicit-thread/messages"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("temporary failure"), None, {}])
async def test_confirmation_keeps_the_parent_after_delivery_abandons_creation(failure):
    adapter = make_adapter()
    adapter._threads_pending[THREAD] = PARENT
    adapter.create_thread = AsyncMock(side_effect=[failure, {"id": THREAD}])
    first = await adapter.send(THREAD, "progress")
    assert first.success
    # Both confirmation lanes (session-keyed and chat-keyed) must obey delivery's
    # decision, not retry creation and move the rest of the turn into a thread.
    assert await adapter._resolve_thread_container(THREAD, session_key=KEY) == PARENT
    assert await adapter._resolve_thread_container(THREAD) == PARENT
    assert (await adapter.send(THREAD, "final")).success
    assert adapter.create_thread.await_count == 1
    assert {call.args[1] for call in adapter._request.await_args_list} == {
        f"/channels/{PARENT}/messages",
    }


@pytest.mark.asyncio
async def test_concurrent_typing_and_progress_open_the_prospective_thread_once():
    adapter = make_adapter()
    adapter._threads_pending[THREAD] = PARENT
    creating = asyncio.Event()
    release = asyncio.Event()
    attempts = []

    async def create(chat, name, **kwargs):
        attempts.append(kwargs["message_id"])
        if len(attempts) > 1:
            raise RuntimeError("thread already created for message")
        creating.set()
        await release.wait()
        return {"id": THREAD, "name": name}

    adapter.create_thread = AsyncMock(side_effect=create)
    typing = asyncio.create_task(adapter._typing_target(THREAD))
    await asyncio.wait_for(creating.wait(), timeout=2)
    progress = asyncio.create_task(adapter.send(THREAD, "progress"))
    try:
        await asyncio.sleep(0.02)
    finally:
        release.set()
    typing_target, result = await asyncio.gather(typing, progress)
    assert typing_target == THREAD
    assert result.success
    assert adapter._request.await_args.args[1] == f"/channels/{THREAD}/messages"
    assert adapter.create_thread.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("first_operation", ["typing", "confirmation"])
@pytest.mark.parametrize("failure", [RuntimeError("temporary failure"), None, {}])
async def test_a_first_operation_creation_failure_sticks_through_final(first_operation, failure):
    adapter = make_adapter()
    adapter._threads_pending[THREAD] = PARENT
    adapter.create_thread = AsyncMock(side_effect=[failure, {"id": THREAD}])
    if first_operation == "typing":
        assert await adapter._typing_target(THREAD) == PARENT
    else:
        assert await adapter._resolve_thread_container(THREAD, session_key=KEY) == PARENT
    assert (await adapter.send(THREAD, "final")).success
    assert adapter._request.await_args.args[1] == f"/channels/{PARENT}/messages"
    assert adapter.create_thread.await_count == 1


@pytest.mark.asyncio
async def test_cancelled_creation_waiter_does_not_release_another_creators_lock():
    adapter = make_adapter()
    creating = asyncio.Event()
    release = asyncio.Event()

    async def create(chat, name, **kwargs):
        creating.set()
        await release.wait()
        return {"id": THREAD, "name": name}

    adapter.create_thread = AsyncMock(side_effect=create)
    first = asyncio.create_task(adapter._start_thread_from_message(PARENT, THREAD))
    await asyncio.wait_for(creating.wait(), timeout=2)
    waiter = asyncio.create_task(adapter._start_thread_from_message(PARENT, THREAD))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    third = asyncio.create_task(adapter._start_thread_from_message(PARENT, THREAD))
    await asyncio.sleep(0)
    release.set()
    assert await first == THREAD
    assert await third == THREAD
    assert adapter.create_thread.await_count == 1
    assert not adapter._thread_create_lock_users
    assert len(adapter._thread_create_locks) <= adapter._threads_by_anchor_max


@pytest.mark.asyncio
async def test_real_host_parent_stop_cancels_the_first_prospective_thread_turn():
    pytest.importorskip("gateway.session")
    adapter = make_adapter()
    adapter.config.typing_indicator = False
    adapter.create_thread = AsyncMock(return_value={"id": THREAD, "name": "review"})
    adapter._channel_kinds[PARENT] = "channel"
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    controls = []

    async def handler(event):
        if event.get_command() == "stop":
            controls.append(event.source.chat_id)
            return "Stopped."
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter.set_message_handler(handler)

    async def incoming(anchor, text):
        data = {"id": anchor, "channel_id": PARENT, "content": text,
                "author": {"id": USER, "username": "Reviewer", "bot": False}}
        await adapter._handle_message_create(data, {"t": "MESSAGE_CREATE", "d": data})

    try:
        await incoming(THREAD, "synthetic review turn")
        await asyncio.wait_for(entered.wait(), timeout=2)
        await incoming("stop-anchor", "/stop")
        await asyncio.wait_for(cancelled.wait(), timeout=2)
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(adapter._background_tasks),
                                                  return_exceptions=True), timeout=5)
        assert controls == [THREAD]
        assert not adapter._active_auto_threads
        assert not adapter._active_sessions
        assert adapter._request.await_args.args[1] == f"/channels/{THREAD}/messages"
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_real_host_goal_pause_resume_and_clear_keep_the_thread_session():
    session = pytest.importorskip("gateway.session")
    commands = pytest.importorskip("gateway.slash_commands")
    goals = pytest.importorskip("hermes_cli.goals")
    adapter = make_adapter()
    managers = {}
    queued = []

    class Runner(commands.GatewaySlashCommandsMixin):
        adapters = {adapter.platform: adapter}

        def _session_key_for_source(self, source):
            return session.build_session_key(source)

        async def _get_goal_manager_for_event(self, event):
            key = self._session_key_for_source(event.source)
            manager = managers.setdefault(key, goals.GoalManager("review-goal-" + event.source.chat_id))
            return manager, None

        def _enqueue_fifo(self, key, event, target_adapter):
            queued.append((key, event))

        def _clear_goal_pending_continuations(self, key, target_adapter):
            queued[:] = [(k, event) for k, event in queued if k != key]

    runner = Runner()

    async def dispatch(text, anchor):
        chat, kind, parent, pending = await adapter._resolve_thread_source({
            "id": anchor, "content": text, "author": {"id": USER},
        }, PARENT, "channel")
        if pending:
            adapter._threads_pending[chat] = pending
        event = MessageEvent(text=text, message_id=anchor, source=adapter.build_source(
            chat_id=chat, chat_type=kind, parent_chat_id=parent, user_id=USER,
            message_id=anchor,
        ))
        await runner._handle_goal_command(event)
        return runner._session_key_for_source(event.source)

    key = await dispatch("/goal repair the watcher", THREAD)
    assert key == KEY
    assert queued[-1][0] == key
    assert queued[-1][1].source.chat_id == THREAD
    manager = managers[key]
    assert manager.has_goal()
    await dispatch("/goal pause", "pause-anchor")
    assert manager.state.status == "paused"
    # The user can work in a separate thread while the goal is parked.
    adapter._active_auto_threads[(PARENT, USER)] = {"unrelated-thread": 1}
    assert await dispatch("/goal resume", "resume-anchor") == key
    assert queued[-1][0] == key
    assert queued[-1][1].message_id is None
    assert await dispatch("/goal clear", "clear-anchor") == key
    assert not manager.has_goal()
    assert adapter._goal_thread_for(PARENT, USER) is None


@pytest.mark.asyncio
async def test_real_host_queued_followup_rebinds_context_through_final_delivery():
    pytest.importorskip("gateway.session")
    adapter = make_adapter()
    adapter.config.typing_indicator = False
    adapter.create_thread = AsyncMock(side_effect=lambda chat, name, **kwargs: {
        "id": "thread-" + kwargs["message_id"], "name": name,
    })
    entered = asyncio.Event()
    release = asyncio.Event()
    adapter._busy_text_debounce_seconds = 0

    async def handler(event):
        if event.message_id == "first":
            entered.set()
            await release.wait()
        assert (await adapter.send(PARENT, "progress-" + event.message_id)).success
        await adapter.stop_typing(PARENT)
        return "final-" + event.message_id

    adapter.set_message_handler(handler)

    def event(anchor, user):
        return MessageEvent(
            text="synthetic review turn", message_id=anchor, message_type=MessageType.TEXT,
            source=adapter.build_source(chat_id=PARENT, chat_type="group", user_id=user,
                                        message_id=anchor),
        )

    try:
        await adapter.handle_message(event("first", USER))
        await asyncio.wait_for(entered.wait(), timeout=2)
        await adapter.handle_message(event("queued", USER))
        await adapter.handle_message(event("parallel", "other-user"))
        release.set()
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(adapter._background_tasks)), timeout=5)
        delivered = {call.kwargs["json"].get("content"): call.args[1]
                     for call in adapter._request.await_args_list
                     if call.args[1].endswith("/messages")}
        for anchor in ("first", "queued", "parallel"):
            for phase in ("progress-", "final-"):
                assert delivered[phase + anchor] == f"/channels/thread-{anchor}/messages"
        assert not adapter._turn_anchor
        assert not adapter._turn_thread_disabled
        assert not adapter._typing_tasks
    finally:
        release.set()
        await adapter.disconnect()
