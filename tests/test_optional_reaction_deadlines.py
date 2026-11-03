"""Optional prompt UI has a batch deadline, without weakening explicit actions."""

import asyncio
from contextlib import asynccontextmanager
import sys
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

import adapter as fluxer
from gateway.config import PlatformConfig


@pytest.fixture
def instance(monkeypatch):
    # Shorten only the optional UI budget, not the primary REST retry contract.
    monkeypatch.setattr(fluxer, "_OPTIONAL_UI_TIMEOUT_SECONDS", 0.05, raising=False)
    return fluxer.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "delivery_verification": False,
        "gateway_state_updates": False,
    }))


@pytest.fixture
def resolvers(monkeypatch):
    import tools

    clarify = ModuleType("tools.clarify_gateway")
    clarify.mark_awaiting_text = Mock()
    clarify.resolve_gateway_clarify = Mock(return_value=True)
    approval = ModuleType("tools.approval")
    approval.resolve_gateway_approval = Mock(return_value=1)
    slash = ModuleType("tools.slash_confirm")
    slash.resolve = AsyncMock()
    for name, module in (("clarify_gateway", clarify), ("approval", approval),
                         ("slash_confirm", slash)):
        monkeypatch.setitem(sys.modules, f"tools.{name}", module)
        monkeypatch.setattr(tools, name, module, raising=False)
    return clarify, approval, slash


async def within_budget(operation):
    # A watchdog, not the implementation: baseline must fail rather than hang.
    return await asyncio.wait_for(operation, 0.12)


def assert_no_new_tasks(before):
    assert not (asyncio.all_tasks() - before), "optional UI left background work alive"


@pytest.mark.asyncio
async def test_split_send_continues_after_stalled_auto_reactions(instance):
    requests = []
    finished = asyncio.Event()

    async def request(method, path, **kwargs):
        requests.append((method, path))
        if method == "PUT":
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
        return {"id": f"message-{len(requests)}"}

    instance._request = request
    before = asyncio.all_tasks()
    result = await within_budget(instance.send("channel", "1️⃣ first\n" + "x" * 5000))
    assert result.success
    assert len(result.raw_response["message_ids"]) >= 2
    assert [method for method, _ in requests].count("POST") >= 2
    assert finished.is_set(), "timed-out affordance must finish cancellation before returning"
    assert_no_new_tasks(before)


@pytest.mark.asyncio
async def test_auto_reaction_deadline_is_shared_by_all_four_options(instance):
    attempted = []

    async def add(*args):
        attempted.append(args)
        await asyncio.sleep(0.035)

    instance._add_reaction = add
    await within_budget(instance._attach_keycap_reactions(
        "channel", "message", "1️⃣ one 2️⃣ two 3️⃣ three 4️⃣ four"))
    assert 1 <= len(attempted) <= 2, "a per-reaction timeout multiplies reply latency"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["stall", "error"])
async def test_clarify_keeps_text_and_all_actions_when_affordances_fail(instance, resolvers, failure):
    instance._request = AsyncMock(return_value={"id": "message"})
    attempted = []

    async def add(chat_id, message_id, emoji):
        attempted.append((chat_id, len(instance._pending_reaction_actions),
                          instance._pending_clarify_cards["clarify"]["chat_id"]))
        if failure == "stall":
            await asyncio.Event().wait()
        raise OSError("synthetic failure")

    instance._add_reaction = add
    before = asyncio.all_tasks()
    result = await within_budget(instance.send_clarify(
        "channel", "Which?", ["one", "two", "three", "four"], "clarify", "session",
        metadata={"thread_id": "thread"},
    ))
    assert result.success
    assert attempted and all(call == ("thread", 4, "thread") for call in attempted)
    assert len(instance._pending_reaction_actions) == 4
    resolvers[0].mark_awaiting_text.assert_called_once_with("clarify")
    content = instance._request.await_args.kwargs["json"]["content"]
    assert "reply with the number" in content
    assert_no_new_tasks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["exec", "slash"])
@pytest.mark.parametrize("failure", ["stall", "error"])
async def test_approval_affordances_are_optional_and_registered_as_a_batch(instance, kind, failure):
    instance._post_message_with_optional_components = AsyncMock(return_value=({"id": "message"}, True))
    expected = 4 if kind == "exec" else 3
    attempted = []

    async def add(chat_id, message_id, emoji):
        attempted.append((chat_id, len(instance._pending_reaction_actions),
                          len(instance._pending_component_actions)))
        if failure == "stall":
            await asyncio.Event().wait()
        raise OSError("synthetic failure")

    instance._add_reaction = add
    before = asyncio.all_tasks()
    if kind == "exec":
        operation = instance.send_exec_approval("channel", "echo test", "session", metadata={"thread_id": "thread"})
    else:
        operation = instance.send_slash_confirm("channel", "Confirm", "Proceed?", "session", "confirm",
                                                metadata={"thread_id": "thread"})
    result = await within_budget(operation)
    assert result.success, "an already delivered prompt must not be retried for a cosmetic failure"
    assert attempted and all(call == ("thread", expected, expected) for call in attempted)
    assert len(instance._pending_reaction_actions) == expected
    assert_no_new_tasks(before)


def register_card(instance):
    instance._register_clarify_card(clarify_id="clarify", message_id="message",
                                    chat_id="thread", content="Which?",
                                    emojis=list(fluxer._CLARIFY_REACTION_EMOJIS))
    instance._pending_reaction_actions["message:1⃣"] = {
        "kind": "clarify", "clarify_id": "clarify", "message_id": "message",
        "channel_id": "thread", "choice": "one",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("notice", ["", "Ended"])
async def test_clarify_retirement_shares_one_deadline_for_edit_and_all_removals(instance, notice):
    register_card(instance)
    attempted = []
    retired = []

    async def request(method, path, **kwargs):
        retired.append(not instance._pending_clarify_cards and not instance._pending_reaction_actions)
        attempted.append((method, path))
        await asyncio.sleep(0.035)
        return {"id": "message"}

    instance._request = request
    before = asyncio.all_tasks()
    await within_budget(instance.retire_clarify_card("clarify", notice))
    assert 1 <= len(attempted) <= 2
    assert retired and all(retired)
    assert all("/channels/thread/messages/message" in path for _, path in attempted)
    assert_no_new_tasks(before)


@pytest.mark.asyncio
async def test_clarify_resolution_precedes_stalled_retirement_ui(instance, resolvers):
    register_card(instance)
    resolved_before_ui = []

    async def request(*args, **kwargs):
        resolved_before_ui.append(resolvers[0].resolve_gateway_clarify.call_count)
        await asyncio.Event().wait()

    instance._request = request
    await within_budget(instance._resolve_clarify_action({"clarify_id": "clarify", "choice": "one"}, user_id="user"))
    assert resolved_before_ui and all(count == 1 for count in resolved_before_ui)
    resolvers[0].resolve_gateway_clarify.assert_called_once_with("clarify", "one")
    assert not instance._pending_reaction_actions
    assert not instance._pending_clarify_cards


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["exec", "slash"])
@pytest.mark.parametrize("interaction", [False, True])
async def test_resolved_approval_ui_cannot_stall_event_dispatch(instance, resolvers, kind, interaction):
    instance._pending_exec_approvals["message"] = {"session_key": "session", "channel_id": "thread"}
    instance._pending_reaction_actions["message:✅"] = {"message_id": "message"}
    action = {"message_id": "message", "session_key": "session", "channel_id": "thread",
              "confirm_id": "confirm", "choice": "once"}
    entered = []

    async def request(method, path, **kwargs):
        count = (resolvers[1].resolve_gateway_approval.call_count if kind == "exec"
                 else resolvers[2].resolve.await_count)
        entered.append((method, len(instance._pending_reaction_actions), count))
        await asyncio.Event().wait()

    instance._request = request
    data = {"id": "interaction", "token": "test-only"} if interaction else None
    method = instance._resolve_exec_approval_action if kind == "exec" else instance._resolve_slash_confirm_action
    before = asyncio.all_tasks()
    await within_budget(method(action, user_id="user", interaction=data))
    assert entered == [("POST" if interaction else "PATCH", 0, 1)]
    if kind == "exec":
        resolvers[1].resolve_gateway_approval.assert_called_once_with("session", "once")
    else:
        resolvers[2].resolve.assert_awaited_once_with("session", "confirm", "once")
    assert_no_new_tasks(before)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["send", "clarify", "exec", "slash", "retire", "resolve"])
async def test_caller_cancellation_is_preserved_and_awaits_optional_cleanup(instance, resolvers, path):
    entered = asyncio.Event()
    finished = asyncio.Event()
    instance._request = AsyncMock(return_value={"id": "message"})
    instance._post_message_with_optional_components = AsyncMock(return_value=({"id": "message"}, False))

    async def stall(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0)
            finished.set()

    instance._add_reaction = stall
    if path == "send":
        operation = instance.send("channel", "1️⃣ one")
    elif path == "clarify":
        operation = instance.send_clarify("channel", "Which?", ["one"], "clarify", "session")
    elif path == "exec":
        operation = instance.send_exec_approval("channel", "echo test", "session")
    elif path == "slash":
        operation = instance.send_slash_confirm("channel", "Confirm", "Proceed?", "session", "confirm")
    elif path == "retire":
        register_card(instance)
        instance._request = stall
        operation = instance.retire_clarify_card("clarify", "Ended")
    else:
        instance._request = stall
        operation = instance._resolve_slash_confirm_action(
            {"message_id": "message", "session_key": "session", "confirm_id": "confirm"}, user_id="user")
    before = asyncio.all_tasks()
    task = asyncio.create_task(operation)
    try:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()
        assert_no_new_tasks(before)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_clarify_affordances_use_final_destination_after_all_reductions(instance, resolvers):
    instance._resolve_outbound_target = AsyncMock(return_value=(
        "gone-thread", {"message_reference": {"message_id": "trigger"}}, "gone-thread"))
    errors = iter([
        (404, {"code": 10008, "message": "Unknown Message"}),
        (400, {"message": "invalid embeds"}),
        (404, {"code": 10003, "message": "Unknown Channel"}),
    ])
    posts = []
    reactions = []

    async def request(method, path, **kwargs):
        if method == "PUT":
            reactions.append((path, len(instance._pending_reaction_actions)))
            return {}
        posts.append((path, kwargs["json"]))
        error = next(errors, None)
        if error:
            status, body = error
            raise httpx.HTTPStatusError("synthetic rejection", request=httpx.Request(method, "https://test.invalid"),
                                        response=httpx.Response(status, json=body))
        return {"id": "message"}

    instance._request = request
    result = await instance.send_clarify("parent", "Which?", ["one", "two"], "clarify", "session",
                                        metadata={"embeds": [{"title": "Options"}]})
    assert result.success
    assert posts[-1][0] == "/channels/parent/messages"
    assert "embeds" not in posts[-1][1]
    assert "message_reference" not in posts[-1][1]
    assert len(reactions) == 2
    assert all(path.startswith("/channels/parent/messages/message/reactions/") and count == 2
               for path, count in reactions)
    assert instance._pending_clarify_cards["clarify"]["chat_id"] == "parent"
    assert {state["channel_id"] for state in instance._pending_reaction_actions.values()} == {"parent"}


@asynccontextmanager
async def stalled_http(mode):
    """Real HTTP client sockets must close on cancellation, including 429 sleep."""
    entered = asyncio.Event()
    disconnected = asyncio.Event()
    handlers = set()
    requests = []

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            if mode == "rate_limit":
                body = b'{"retry_after":60}'
                writer.write(b"HTTP/1.1 429 Too Many Requests\r\nContent-Length: " +
                             str(len(body)).encode() + b"\r\n\r\n" + body)
                await writer.drain()
            entered.set()
            while await reader.read(4096):
                pass
            disconnected.set()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", entered, disconnected, requests
    finally:
        server.close()
        await server.wait_closed()
        active = list(handlers)
        for task in active:
            task.cancel()
        await asyncio.gather(*active, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stall", "rate_limit"])
@pytest.mark.parametrize("cancel", [False, True])
async def test_real_request_timeout_or_cancellation_closes_client_and_socket(instance, resolvers, monkeypatch,
                                                                          mode, cancel):
    monkeypatch.setattr(fluxer, "_OPTIONAL_UI_TIMEOUT_SECONDS", 0.4)
    clients = []
    base = httpx.AsyncClient

    class TrackedClient(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            clients.append(self)

    monkeypatch.setattr(httpx, "AsyncClient", TrackedClient)
    instance.send = AsyncMock(return_value=fluxer.SendResult(success=True, message_id="message"))
    async with stalled_http(mode) as (url, entered, disconnected, requests):
        instance.api_base_url = url
        before = asyncio.all_tasks()
        task = asyncio.create_task(instance.send_clarify("channel", "Which?", ["one", "two"], "clarify", "session"))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await asyncio.wait_for(task, 0.7)
                assert result.success
            await asyncio.wait_for(disconnected.wait(), 0.3)
            assert len(requests) == 1, "optional timeout must stop both retries and later reactions"
            assert clients and all(client.is_closed for client in clients)
            # The loopback handler may need one tick to finish closing its socket.
            await asyncio.sleep(0)
            assert_no_new_tasks(before)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["add", "remove"])
async def test_explicit_reaction_actions_retain_real_request_retry_contract(instance, monkeypatch, action):
    attempts = []
    clients = []
    base = httpx.AsyncClient

    async def respond(request):
        attempts.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"reactions": [
                {"me": True, "emoji": {"name": "✅"}},
                {"me": True, "emoji": {"name": "custom", "id": "123"}},
                {"me": False, "emoji": {"name": "❌"}},
            ]})
        count = attempts.count(request.method)
        if count <= 2:
            return httpx.Response(429, json={"retry_after": 0.05})
        return httpx.Response(204)

    def client(*args, **kwargs):
        result = base(*args, transport=httpx.MockTransport(respond), **kwargs)
        clients.append(result)
        return result

    monkeypatch.setattr(httpx, "AsyncClient", client)
    if action == "add":
        result = await instance.add_reaction("channel", "✅", message_id="message")
        assert attempts == ["PUT", "PUT", "PUT"]
    else:
        result = await instance.remove_reaction("channel", message_id="message")
        assert attempts == ["GET", "DELETE", "DELETE", "DELETE", "DELETE"]
        assert result["removed"] == 2
    assert result["success"] is True
    assert result["message_id"] == "message"
    assert all(client.is_closed for client in clients)
