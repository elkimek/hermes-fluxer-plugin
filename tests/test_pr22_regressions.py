"""Offline boundary regressions from the PR 22 review."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer
from gateway.config import PlatformConfig
from kokoro_tts import KokoroSpeechClient


def reaction_adapter():
    adapter = fluxer.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "app.secret"})
    )
    adapter.bot_user_id = "bot-1"
    adapter._allow_all_users = False
    adapter._allowed_user_ids = {"user-1"}
    adapter._known_channel_ids.add("dm-1")
    adapter.handle_message = AsyncMock()
    return adapter


def reaction_data(**extra):
    return {
        "user_id": "user-1",
        "message_id": "message-1",
        "channel_id": "dm-1",
        "emoji": {"name": "1️⃣"},
        **extra,
    }


def requests_for(adapter, *, author="bot-1", channel_type=1):
    async def request(method, path, **kwargs):
        if path == "/channels/dm-1":
            return {"id": "dm-1", "type": channel_type}
        return {
            "id": "message-1",
            "author": {"id": author, "bot": True},
            "content": "1️⃣ Continue",
        }

    adapter._request = AsyncMock(side_effect=request)


@pytest.mark.asyncio
async def test_reaction_forward_rejects_other_bots():
    adapter = reaction_adapter()
    requests_for(adapter, author="other-bot")
    await adapter._handle_reaction_add(reaction_data())
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_type", [0, 2, 3, 11])
async def test_reaction_forward_includes_guild_channels(channel_type):
    adapter = reaction_adapter()
    requests_for(adapter, channel_type=channel_type)
    await adapter._handle_reaction_add(reaction_data())
    # Decision reactions are now forwarded in guild channels too, not just DMs:
    # the answer is routed through the same handle_message path as in a DM.
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_reaction_forward_retries_after_failed_message_lookup():
    adapter = reaction_adapter()
    requests_for(adapter)
    healthy = adapter._request.side_effect
    failed = False

    async def flaky(method, path, **kwargs):
        nonlocal failed
        if "/messages/" in path and not failed:
            failed = True
            raise OSError("temporary outage")
        return await healthy(method, path, **kwargs)

    adapter._request.side_effect = flaky
    await adapter._handle_reaction_add(reaction_data())
    await adapter._handle_reaction_add(reaction_data())
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_retired_clarify_is_not_forwarded_as_a_new_answer():
    adapter = reaction_adapter()
    requests_for(adapter)
    adapter._register_clarify_card(
        clarify_id="c-1",
        message_id="message-1",
        chat_id="dm-1",
        content="Which?",
        emojis=["1️⃣"],
    )
    await adapter.retire_clarify_card("c-1")
    await adapter._handle_reaction_add(reaction_data())
    adapter.handle_message.assert_not_awaited()


def test_clarify_card_eviction_also_bounds_actions():
    adapter = reaction_adapter()
    for index in range(205):
        mid = f"message-{index}"
        cid = f"c-{index}"
        adapter._register_clarify_card(
            clarify_id=cid,
            message_id=mid,
            chat_id="dm-1",
            content="Which?",
            emojis=["1️⃣"],
        )
        adapter._pending_reaction_actions[f"{mid}:1⃣"] = {
            "kind": "clarify",
            "clarify_id": cid,
        }
    assert len(adapter._pending_clarify_cards) <= 200
    assert len(adapter._pending_reaction_actions) <= 200


@pytest.mark.asyncio
async def test_kokoro_timeout_covers_a_blocked_sink(monkeypatch):
    from test_kokoro_tts import install_stream

    install_stream(monkeypatch, chunks=[b"ab" * 64])
    client = KokoroSpeechClient(base_url="http://kokoro.test/v1", voice="test")

    async def sink(chunk):
        await asyncio.sleep(0.4)

    started = asyncio.get_running_loop().time()
    with pytest.raises(RuntimeError, match="did not finish within"):
        await client.text_response_to_sink("hello", sink, timeout=0.05)
    assert asyncio.get_running_loop().time() - started < 0.25


@pytest.mark.asyncio
async def test_kokoro_cancel_closes_a_stalled_http_connection():
    """Real loopback HTTP, not a cooperative fake read() cancellation."""
    disconnected = asyncio.Event()
    connected = asyncio.Event()
    release = asyncio.Event()
    handlers = set()

    async def serve(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            await writer.drain()
            connected.set()
            # Drain the request body, then observe client EOF.
            while await reader.read(4096):
                pass
            disconnected.set()
            await release.wait()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.discard(asyncio.current_task())

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    async def sink(chunk):
        pass

    client = KokoroSpeechClient(base_url=f"http://127.0.0.1:{port}/v1", voice="test")
    task = asyncio.create_task(client.text_response_to_sink("hello", sink, timeout=0.7))
    try:
        await asyncio.wait_for(connected.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # A cancelled to_thread task is NOT a closed HTTP connection.
        await asyncio.wait_for(disconnected.wait(), 0.2)
    finally:
        release.set()
        server.close()
        await server.wait_closed()
        for handler in list(handlers):
            handler.cancel()
        await asyncio.gather(*list(handlers), return_exceptions=True)


@pytest.mark.asyncio
async def test_reaction_concurrent_duplicates_are_single_flight():
    adapter = reaction_adapter()
    requests_for(adapter)
    await asyncio.gather(
        *(adapter._handle_reaction_add(reaction_data()) for _ in range(5))
    )
    adapter.handle_message.assert_awaited_once()


def test_kokoro_yaml_reaches_gateway_and_child(monkeypatch):
    settings = {
        "kokoro_url": "http://tts.test/v1",
        "kokoro_voice": "voice-name",
        "kokoro_model": "kokoro",
        "kokoro_api_key": "synthetic-key",
    }
    names = {key: "FLUXER_VOICE_" + key.upper() for key in settings}
    for name in names.values():
        monkeypatch.delenv(name, raising=False)
    supervisor = fluxer.FluxerVoiceSupervisorProcess(extra={"voice": settings})
    child = supervisor._child_env()
    for key, name in names.items():
        assert child.get(name) == settings[key]
    fluxer._apply_yaml_config({}, {"voice": settings})
    import os

    for key, name in names.items():
        assert os.getenv(name) == settings[key]
    # Explicit environment overrides YAML in both paths.
    monkeypatch.setenv(names["kokoro_voice"], "override")
    assert supervisor._child_env()[names["kokoro_voice"]] == "override"


def fake_clarify(monkeypatch):
    import sys
    import types
    import threading

    stub = types.ModuleType("tools.clarify_gateway")
    stub.mark_awaiting_text = lambda cid: None
    stub.resolve_gateway_clarify = lambda cid, choice: True
    stub._entries = {}
    stub._lock = threading.RLock()
    import tools

    monkeypatch.setattr(tools, "clarify_gateway", stub, raising=False)
    monkeypatch.setitem(sys.modules, "tools.clarify_gateway", stub)
    return stub


@pytest.mark.asyncio
async def test_clarify_does_not_hide_choices_after_four(monkeypatch):
    fake_clarify(monkeypatch)
    adapter = reaction_adapter()
    adapter.send = AsyncMock(
        return_value=fluxer.SendResult(success=True, message_id="message-1")
    )
    adapter._add_reaction = AsyncMock()
    choices = [f"option-{n}" for n in range(6)]
    await adapter.send_clarify("dm-1", "Which?", choices, "c-1", "session-1")
    content = adapter.send.await_args.kwargs["content"]
    assert all(choice in content for choice in choices)


@pytest.mark.asyncio
async def test_clarify_rejects_a_missing_reaction_channel():
    adapter = reaction_adapter()
    action = {
        "kind": "clarify",
        "clarify_id": "c-1",
        "channel_id": "dm-1",
        "choice": "one",
    }
    adapter._pending_reaction_actions["message-1:1⃣"] = action
    adapter._resolve_clarify_action = AsyncMock()
    await adapter._handle_reaction_add(reaction_data(channel_id=""))
    adapter._resolve_clarify_action.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarify_registers_before_its_first_affordance(monkeypatch):
    fake_clarify(monkeypatch)
    adapter = reaction_adapter()
    adapter._delivery_verification_enabled = False
    adapter._request = AsyncMock(return_value={"id": "message-1"})
    registered = []

    async def attach(chat_id, message_id, emoji):
        registered.append(f"{message_id}:1⃣" in adapter._pending_reaction_actions)

    adapter._add_reaction = attach
    await adapter.send_clarify("dm-1", "Which?", ["one", "two"], "c-1", "session-1")
    assert registered and all(registered)


@pytest.mark.asyncio
async def test_local_stt_never_dispatches_via_cloud_selection(monkeypatch):
    import scripts.fluxer_stt_voice_loop as loop

    def cloud(*args, **kwargs):
        return {"provider": "cloud"}

    def local(*args, **kwargs):
        return {"provider": "local"}

    monkeypatch.setattr(loop, "transcribe_audio", cloud)
    monkeypatch.setattr(loop, "transcribe_audio_local_fallback", local, raising=False)
    result = loop.transcribe_with_provider("unused.wav", provider="local", model=None)
    assert result["provider"] == "local"


@pytest.mark.asyncio
async def test_reaction_choice_resolves_real_host_clarify_in_its_thread():
    cg = pytest.importorskip("tools.clarify_gateway")
    adapter = reaction_adapter()
    adapter.send = AsyncMock(
        return_value=fluxer.SendResult(success=True, message_id="message-1")
    )
    adapter._add_reaction = AsyncMock()
    adapter.edit_message = AsyncMock()
    adapter._request = AsyncMock(return_value={})
    adapter._remember_message_destination("message-1", "thread-1")
    session = "agent:main:fluxer:thread:channel-1:thread-1"
    entry = cg.register("review-clarify", session, "Which?", ["one", "two"])
    try:
        await adapter.send_clarify(
            "channel-1", "Which?", ["one", "two"], "review-clarify", session
        )
        await adapter._handle_reaction_add(reaction_data(channel_id="wrong-thread"))
        assert not entry.event.is_set()
        await adapter._handle_reaction_add(
            reaction_data(channel_id="thread-1", user_id="stranger")
        )
        assert not entry.event.is_set()
        await adapter._handle_reaction_add(
            reaction_data(channel_id="thread-1", emoji={"name": "2️⃣"})
        )
        assert entry.event.is_set()
        assert entry.response == "two"
        assert "review-clarify" not in adapter._pending_clarify_cards
        assert not adapter._pending_reaction_actions
        # The same retired click does not become a synthetic private message.
        await adapter._handle_reaction_add(
            reaction_data(channel_id="thread-1", emoji={"name": "2️⃣"})
        )
        adapter.handle_message.assert_not_awaited()
    finally:
        cg.clear_session(session)
