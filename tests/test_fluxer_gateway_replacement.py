"""Reconnect cleanup must not reject the replacement it just installed."""
import asyncio
from unittest.mock import AsyncMock

import pytest
import websockets

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


class Websocket:
    def __init__(self):
        self.closed = asyncio.Event()
        self.listening = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        self.listening.set()
        await self.closed.wait()
        raise StopAsyncIteration

    async def close(self):
        self.closed.set()
        # Real websocket close yields and lets the old listener run its finally.
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_reconnect_accepts_replacement_after_old_listener_cleanup(monkeypatch):
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "gateway_state_updates": False,
        "backlog_recovery": False,
    }))
    adapter.gateway_url = "wss://gateway.example.test"
    adapter._closing = False
    adapter._running = True
    old = Websocket()
    new = Websocket()
    adapter._ws = old
    old_listener = asyncio.create_task(adapter._listen_loop())
    adapter._listener_task = old_listener
    await old.listening.wait()
    connected = asyncio.Event()
    adapter._mark_connected = connected.set
    adapter._voice_supervisor.start = lambda: None
    adapter._recover_backlog = AsyncMock()
    connect = AsyncMock(return_value=new)
    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(fluxer_adapter, "_RECONNECT_BASE_DELAY", 0.01)
    recovery = asyncio.create_task(adapter._reconnect_loop("heartbeat"))
    adapter._reconnect_task = recovery
    try:
        await asyncio.wait_for(connected.wait(), 0.25)
        await asyncio.wait_for(recovery, 0.25)
        assert connect.await_count == 1
        assert adapter._ws is new
        assert adapter._listener_is_alive()
    finally:
        adapter._closing = True
        recovery.cancel()
        listener = adapter._listener_task
        if listener is not None:
            listener.cancel()
        old_listener.cancel()
        await asyncio.gather(recovery, old_listener, *([listener] if listener else []), return_exceptions=True)


@pytest.mark.asyncio
async def test_replacement_that_dies_during_backlog_still_reconnects(monkeypatch):
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "gateway_state_updates": False,
        "backlog_recovery": False,
    }))
    adapter.gateway_url = "wss://gateway.example.test"
    adapter._closing = False
    adapter._running = True
    first, second = Websocket(), Websocket()
    connect = AsyncMock(side_effect=[first, second])
    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.setattr(fluxer_adapter, "_RECONNECT_BASE_DELAY", 0.01)
    connected = asyncio.Event()
    adapter._mark_connected = connected.set
    adapter._voice_supervisor.start = lambda: None

    async def recover_backlog(**kwargs):
        if adapter._ws is first:
            await first.close()
            # Exercise the real listener's finally while recovery owns the slot.
            await adapter._listener_task
        else:
            await second.listening.wait()

    adapter._recover_backlog = recover_backlog
    recovery = asyncio.create_task(adapter._reconnect_loop("test"))
    try:
        await asyncio.wait_for(connected.wait(), 2)
        await asyncio.wait_for(recovery, 2)
        assert connect.await_count == 2
        assert adapter._ws is second
        assert adapter._listener_is_alive()
    finally:
        adapter._closing = True
        recovery.cancel()
        listener = adapter._listener_task
        if listener:
            listener.cancel()
        await asyncio.gather(recovery, *([listener] if listener else []), return_exceptions=True)
