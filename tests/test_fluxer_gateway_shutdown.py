"""Gateway resources belong to the adapter, including failed/canceled setup."""
import asyncio
from unittest.mock import AsyncMock

import pytest
import websockets

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


def make_adapter():
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "gateway_state_updates": False,
        "backlog_recovery": False,
    }))
    adapter.gateway_url = "wss://gateway.example.test"
    # The standalone gateway shim omits the host's status-writing helper.
    adapter._set_fatal_error = lambda *args, **kwargs: None
    adapter._voice_supervisor.start = lambda: True
    adapter._voice_supervisor.stop = AsyncMock()
    return adapter


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
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("abort", ["cancel", "error"])
async def test_aborted_connect_closes_listener_and_watchdog(monkeypatch, abort):
    adapter = make_adapter()
    websocket = Websocket()
    monkeypatch.setattr(websockets, "connect", AsyncMock(return_value=websocket))
    register_started, release = asyncio.Event(), asyncio.Event()
    owned = []

    async def register():
        owned.extend([adapter._listener_task, adapter._watchdog_task])
        register_started.set()
        await release.wait()
        raise RuntimeError("registration failed")

    adapter._maybe_register_native_commands = register
    connect = asyncio.create_task(adapter.connect())
    try:
        await asyncio.wait_for(register_started.wait(), 2)
        if abort == "cancel":
            connect.cancel()
            with pytest.raises(asyncio.CancelledError):
                await connect
        else:
            release.set()
            assert await connect is False
        assert all(task.done() for task in owned), "aborted setup leaked owned gateway tasks"
        assert websocket.closed.is_set()
        assert adapter._ws is None
        assert adapter not in fluxer_adapter._LIVE_ADAPTERS
    finally:
        adapter._closing = True
        release.set()
        connect.cancel()
        for task in owned:
            task.cancel()
        await websocket.close()
        await asyncio.gather(connect, *owned, return_exceptions=True)
        fluxer_adapter._LIVE_ADAPTERS.discard(adapter)


@pytest.mark.asyncio
async def test_shutdown_blocks_recovery_before_waiting_for_voice():
    adapter = make_adapter()
    adapter._closing = False
    stopping, release = asyncio.Event(), asyncio.Event()

    async def stop():
        stopping.set()
        await release.wait()

    adapter._voice_supervisor.stop = stop
    shutdown = asyncio.create_task(adapter.disconnect())
    try:
        await asyncio.wait_for(stopping.wait(), 2)
        assert adapter._closing, "voice shutdown must not leave reconnect admission open"
        adapter._schedule_reconnect("listener stopped while voice was leaving")
        assert adapter._reconnect_task is None
    finally:
        release.set()
        await shutdown


@pytest.mark.asyncio
async def test_disconnect_awaits_watchdog_finalization():
    adapter = make_adapter()
    started, retiring, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def watchdog():
        started.set()
        try:
            await asyncio.Future()
        finally:
            retiring.set()
            await release.wait()

    watchdog_task = asyncio.create_task(watchdog())
    adapter._watchdog_task = watchdog_task
    await started.wait()
    shutdown = asyncio.create_task(adapter.disconnect())
    try:
        await asyncio.wait_for(retiring.wait(), 2)
        assert not shutdown.done(), "disconnect returned with a detached live watchdog"
        release.set()
        await asyncio.wait_for(shutdown, 2)
        assert watchdog_task.done()
        assert adapter._watchdog_task is None
    finally:
        release.set()
        await asyncio.gather(shutdown, watchdog_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancel_during_retirement_finishes_closing_detached_socket(monkeypatch):
    adapter = make_adapter()
    old = Websocket()
    closing, release = asyncio.Event(), asyncio.Event()

    async def slow_close():
        closing.set()
        await release.wait()
        old.closed.set()

    old.close = slow_close
    adapter._ws = old
    listener = asyncio.create_task(adapter._listen_loop())
    adapter._listener_task = listener
    await old.listening.wait()
    dial = AsyncMock()
    monkeypatch.setattr(websockets, "connect", dial)
    replacement = asyncio.create_task(adapter._connect_gateway_once())
    try:
        await asyncio.wait_for(closing.wait(), 2)
        replacement.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await replacement
        assert old.closed.is_set(), "cancellation abandoned a detached socket"
        assert listener.done()
        assert adapter._ws is None
        dial.assert_not_awaited()
    finally:
        adapter._closing = True
        release.set()
        listener.cancel()
        await asyncio.gather(replacement, listener, return_exceptions=True)


@pytest.mark.asyncio
async def test_current_heartbeat_failure_requests_recovery_even_while_disconnected():
    adapter = make_adapter()
    adapter._closing = False
    adapter._running = False  # Reconnect has not yet published connected.
    adapter._ws = Websocket()
    adapter._send_heartbeat = AsyncMock(side_effect=OSError("socket send failed"))
    scheduled = []
    adapter._schedule_reconnect = scheduled.append
    await asyncio.wait_for(adapter._heartbeat_loop(1000), 2)
    assert scheduled == ["heartbeat loop stopped"]
