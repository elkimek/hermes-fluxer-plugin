"""Optional membership must be bounded and keep private identifiers out of logs."""
import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


def install_rest_client(monkeypatch, handler):
    """Keep the real REST retry/error helper, replacing only the HTTP client."""
    import httpx

    clients = []
    calls = []

    class Client:
        closed = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            self.closed = True

        async def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            return await handler(httpx.Request(method, url))

    def factory(**_kwargs):
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return calls, clients


def make_adapter():
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "gateway_state_updates": False,
    }))
    adapter._session_title_for_chat = lambda *_args: "Synthetic title"
    adapter.create_thread = AsyncMock(return_value={"id": "private-thread", "name": "Synthetic title"})
    adapter._inbound_author_by_message["trigger"] = "private-user"
    return adapter


@pytest.mark.asyncio
async def test_slow_member_route_cannot_hold_back_the_ready_reply(monkeypatch):
    adapter = make_adapter()
    monkeypatch.setattr(fluxer_adapter, "_THREAD_MEMBER_ADD_TIMEOUT_SECONDS", 0.01, raising=False)
    cancelled = asyncio.Event()

    async def request(method, _path, **_kwargs):
        if method == "PUT":
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return {"id": "reply"}

    adapter._request = request
    result = await asyncio.wait_for(adapter.send("parent", "ready", reply_to="trigger"), 0.5)
    assert result.success
    assert result.message_id == "reply"
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_member_failure_diagnostics_exclude_private_details(caplog):
    adapter = make_adapter()
    adapter._request = AsyncMock(side_effect=RuntimeError("private-provider-detail"))
    with caplog.at_level(logging.DEBUG, logger=fluxer_adapter.__name__):
        await adapter._add_thread_member("private-thread", "private-user")
    assert "Fluxer thread member add failed" in caplog.text
    assert "private-thread" not in caplog.text
    assert "private-user" not in caplog.text
    assert "private-provider-detail" not in caplog.text


@pytest.mark.asyncio
async def test_member_add_preserves_caller_cancellation():
    adapter = make_adapter()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    adapter._request = request
    task = asyncio.create_task(adapter._add_thread_member("private-thread", "private-user"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [204, 403, 404, 429, "transport", "invalid-json"])
async def test_member_rest_failures_deliver_without_private_diagnostics(monkeypatch, caplog, outcome):
    import httpx

    adapter = make_adapter()

    async def request(incoming):
        if incoming.method == "POST":
            return httpx.Response(200, json={"id": "reply"}, request=incoming)
        if outcome == "transport":
            raise httpx.ConnectError("private-provider-detail", request=incoming)
        if outcome == "invalid-json":
            return httpx.Response(200, text="private-provider-detail", request=incoming)
        return httpx.Response(
            outcome, text="private-provider-detail" if outcome != 204 else "",
            headers={"Retry-After": "0"}, request=incoming,
        )

    calls, clients = install_rest_client(monkeypatch, request)
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", AsyncMock())
    with caplog.at_level(logging.DEBUG, logger=fluxer_adapter.__name__):
        result = await adapter.send("parent", "ready", reply_to="trigger")

    assert result.success
    assert result.message_id == "reply"
    assert [method for method, _url, _kwargs in calls] == (
        ["PUT", "PUT", "PUT", "POST"] if outcome == 429 else ["PUT", "POST"]
    )
    assert all(client.closed for client in clients)
    if outcome != 204:
        assert "Fluxer thread member add failed" in caplog.text
    else:
        assert not caplog.text
    for private in ("private-thread", "private-user", "private-provider-detail", adapter.bot_token):
        assert private not in caplog.text


@pytest.mark.asyncio
async def test_rate_limit_sleep_shares_member_budget_and_closes_client(monkeypatch, caplog):
    import httpx

    adapter = make_adapter()
    monkeypatch.setattr(fluxer_adapter, "_THREAD_MEMBER_ADD_TIMEOUT_SECONDS", 0.02)
    real_sleep = asyncio.sleep
    cancelled = asyncio.Event()
    delays = []

    async def request(incoming):
        if incoming.method == "PUT":
            return httpx.Response(429, headers={"Retry-After": "60"}, request=incoming)
        return httpx.Response(200, json={"id": "reply"}, request=incoming)

    async def retry_sleep(delay):
        delays.append(delay)
        try:
            await real_sleep(delay)
        finally:
            cancelled.set()

    calls, clients = install_rest_client(monkeypatch, request)
    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", retry_sleep)
    with caplog.at_level(logging.DEBUG, logger=fluxer_adapter.__name__):
        result = await asyncio.wait_for(adapter.send("parent", "ready", reply_to="trigger"), 0.5)
    assert result.success
    assert delays == [60.0]
    assert cancelled.is_set()
    assert all(client.closed for client in clients)
    assert [method for method, _url, _kwargs in calls] == ["PUT", "POST"]
    for private in ("private-thread", "private-user", adapter.bot_token):
        assert private not in caplog.text
    # Reusing the turn's thread does not repeat a failed optional membership step.
    assert (await adapter.send("parent", "final", reply_to="trigger")).success
    assert [method for method, _url, _kwargs in calls] == ["PUT", "POST", "POST"]
    adapter.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_send_during_membership_cleans_up_without_posting(monkeypatch):
    adapter = make_adapter()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def request(_incoming):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    calls, clients = install_rest_client(monkeypatch, request)
    task = asyncio.create_task(adapter.send("parent", "ready", reply_to="trigger"))
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 0.5)
        assert cancelled.is_set()
        assert all(client.closed for client in clients)
        assert [method for method, _url, _kwargs in calls] == ["PUT"]
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["denied-user", "missing-mention", "denied-channel", "bot"])
async def test_unadmitted_messages_cannot_supply_a_thread_member(monkeypatch, reason):
    adapter = make_adapter()
    adapter._inbound_author_by_message.clear()
    adapter._allowed_user_ids = {"admitted-user"}
    adapter._allowed_channel_ids = {"parent"}
    adapter.bot_user_id = "test-bot"
    adapter._extract_attachments = AsyncMock(return_value=([], []))
    adapter._resolve_reply_context = AsyncMock(return_value=(None, None))
    adapter.handle_message = AsyncMock()
    adapter._request = AsyncMock(return_value={"id": "reply"})
    data = {
        "id": "trigger", "channel_id": "parent", "channel_type": 0,
        "content": "<@test-bot> help", "author": {"id": "admitted-user"},
    }
    if reason == "denied-user":
        data["author"]["id"] = "denied-user"
    elif reason == "missing-mention":
        data["content"] = "not addressed to the bot"
    elif reason == "denied-channel":
        adapter._allowed_channel_ids = {"another-channel"}
    else:
        data["author"]["bot"] = True
    await adapter._handle_message_create(data, {})
    adapter.handle_message.assert_not_awaited()
    assert "trigger" not in adapter._inbound_author_by_message
    # Even an explicitly anchored standalone send cannot guess a rejected author.
    assert (await adapter.send("parent", "ready", reply_to="trigger")).success
    assert [call.args[0] for call in adapter._request.await_args_list] == ["POST"]


@pytest.mark.asyncio
async def test_admitted_turn_keeps_its_author_when_newer_messages_arrive(monkeypatch):
    adapter = make_adapter()
    adapter._inbound_author_by_message.clear()
    adapter._allowed_user_ids = {"first-user", "second-user"}
    adapter.bot_user_id = "test-bot"
    adapter._extract_attachments = AsyncMock(return_value=([], []))
    adapter._resolve_reply_context = AsyncMock(return_value=(None, None))
    adapter._request = AsyncMock(return_value={"id": "reply"})
    adapter.create_thread = AsyncMock(side_effect=lambda _chat, _name, **kw: {
        "id": "thread-" + kw["message_id"], "name": "Synthetic title",
    })
    both_started = asyncio.Event()
    started = []

    async def handle(_self, event):
        started.append(event.message_id)
        if len(started) == 2:
            both_started.set()
        await both_started.wait()
        assert (await adapter.send("parent", "ready")).success
        assert (await adapter.send("parent", "final")).success

    monkeypatch.setattr(BasePlatformAdapter, "handle_message", handle, raising=False)

    async def incoming(anchor, user):
        await adapter._handle_message_create({
            "id": anchor, "channel_id": "parent", "channel_type": 0,
            "content": "<@test-bot> help", "author": {"id": user},
        }, {})

    await asyncio.wait_for(asyncio.gather(
        incoming("first-trigger", "first-user"), incoming("second-trigger", "second-user"),
    ), 0.5)
    puts = [call.args[1] for call in adapter._request.await_args_list if call.args[0] == "PUT"]
    assert sorted(puts) == [
        "/channels/thread-first-trigger/thread-members/first-user",
        "/channels/thread-second-trigger/thread-members/second-user",
    ]
    assert adapter.create_thread.await_count == 2
