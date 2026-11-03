"""Gateway supervision: a dead receive path must reconnect, never go quiet.

Recorded from a live RCA (2026-10-30). A Fluxer gateway session died with no
inbound traffic from 17:32:26 until the next gateway restart 16 hours later and
not one log line after "Fluxer gateway reconnected after no close frame received
or sent". The adapter still reported itself connected; there was no websocket and
no listener task. These tests pin the three defects that allowed it:

1. ``_schedule_reconnect`` dropped a disconnect request whenever a reconnect loop
   happened to be running, and that loop then reported success over a listener
   that had already exited.
2. The heartbeat loop, the only component that closes a zombie socket, could die
   at ``debug`` level with no reconnect.
3. Nothing watched the receive path, so a listener that ended outside the
   reconnect path left the platform silently deaf.
"""

import asyncio

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


def make_adapter() -> fluxer_adapter.FluxerAdapter:
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "allow_all_users": True,
        "require_mention": False, "gateway_state_updates": False,
        "backlog_recovery": False,
    }))


class FakeWs:
    """Minimal stand-in holding a listener open until closed."""

    def __init__(self) -> None:
        self._closed = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self._closed.wait()
        raise StopAsyncIteration

    async def send(self, _payload):
        return None

    async def close(self):
        self._closed.set()


def install_listener_task(adapter: fluxer_adapter.FluxerAdapter):
    """Install a listener task and hand it back so a caller can await it."""

    async def noop() -> None:
        return None

    task = asyncio.get_running_loop().create_task(noop())
    adapter._listener_task = task
    return task


def dead_listener(adapter: fluxer_adapter.FluxerAdapter) -> None:
    """Install a listener task that any later await lets run to completion."""

    install_listener_task(adapter)


@pytest.mark.asyncio
async def test_reconnect_loop_refuses_to_report_success_over_a_dead_listener():
    # The reconnect connects the socket, but the listener it spawns is already
    # gone by the time the attempt finishes. That must never publish "connected".
    adapter = make_adapter()
    adapter._closing = False
    adapter._voice_supervisor.start = lambda: True
    adapter._marked_connected = False
    adapter._mark_connected = lambda: setattr(adapter, "_marked_connected", True)

    attempts = []

    async def connect_and_die() -> None:
        attempts.append(1)
        # Finish the listener before returning, so liveness is checked against a
        # task that has genuinely exited rather than one still queued.
        await asyncio.gather(install_listener_task(adapter), return_exceptions=True)

    adapter._connect_gateway_once = connect_and_die

    original = fluxer_adapter._RECONNECT_BASE_DELAY
    fluxer_adapter._RECONNECT_BASE_DELAY = 0.0
    try:
        task = asyncio.get_running_loop().create_task(adapter._reconnect_loop("test"))
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    finally:
        fluxer_adapter._RECONNECT_BASE_DELAY = original

    assert adapter._marked_connected is False, "must not publish connected over a dead listener"
    assert len(attempts) > 1, "must keep retrying, not stop"


@pytest.mark.asyncio
async def test_schedule_reconnect_marks_request_when_a_loop_is_already_running():
    # Dropping the request here is what left the platform deaf: the running loop
    # had created the listener that just died.
    adapter = make_adapter()
    adapter._closing = False

    async def still_running():
        await asyncio.sleep(30)

    task = asyncio.get_running_loop().create_task(still_running())
    adapter._reconnect_task = task
    adapter._reconnect_requested = False

    adapter._schedule_reconnect("listener died mid-reconnect")

    assert adapter._reconnect_requested is True
    task.cancel()


@pytest.mark.asyncio
async def test_watchdog_reconnects_when_the_listener_task_is_gone():
    # No log line, no websocket, no listener: the exact live state. The watchdog
    # must notice and arm a reconnect.
    adapter = make_adapter()
    adapter._closing = False
    adapter._ws = None
    adapter._reconnect_task = None
    dead_listener(adapter)

    scheduled = []
    adapter._schedule_reconnect = lambda reason: scheduled.append(reason)
    adapter._mark_disconnected = lambda: setattr(adapter, "_disconnected", True)
    adapter._disconnected = False

    original = fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS
    fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = 0.01
    try:
        watchdog = asyncio.get_running_loop().create_task(adapter._gateway_watchdog_loop())
        await asyncio.sleep(0.1)
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
    finally:
        fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = original

    assert scheduled, "watchdog must arm a reconnect for a missing listener"
    assert adapter._disconnected is True


@pytest.mark.asyncio
async def test_watchdog_is_not_blinded_by_a_cleared_running_flag():
    # The live failure mode was a teardown path that left the adapter believing
    # it was stopped while the process kept running. Liveness is decided by the
    # socket and the listener, never by the flag.
    adapter = make_adapter()
    adapter._closing = False
    adapter._running = False
    adapter._ws = None
    adapter._reconnect_task = None
    dead_listener(adapter)

    scheduled = []
    adapter._schedule_reconnect = lambda reason: scheduled.append(reason)
    adapter._mark_disconnected = lambda: None

    original = fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS
    fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = 0.01
    try:
        watchdog = asyncio.get_running_loop().create_task(adapter._gateway_watchdog_loop())
        await asyncio.sleep(0.1)
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
    finally:
        fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = original

    assert scheduled, "a cleared _running flag must not disable supervision"


@pytest.mark.asyncio
async def test_watchdog_defers_to_an_in_flight_reconnect():
    # While a reconnect is already running it owns recovery; the watchdog must not
    # stack a second one on top.
    adapter = make_adapter()
    adapter._closing = False
    adapter._ws = None
    dead_listener(adapter)

    async def in_flight():
        await asyncio.sleep(30)

    adapter._reconnect_task = asyncio.get_running_loop().create_task(in_flight())

    scheduled = []
    adapter._schedule_reconnect = lambda reason: scheduled.append(reason)
    adapter._mark_disconnected = lambda: None

    original = fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS
    fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = 0.01
    try:
        watchdog = asyncio.get_running_loop().create_task(adapter._gateway_watchdog_loop())
        await asyncio.sleep(0.1)
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
    finally:
        fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = original
        if not adapter._reconnect_task.done():
            adapter._reconnect_task.cancel()

    assert scheduled == [], "an in-flight reconnect must not be duplicated"


@pytest.mark.asyncio
async def test_watchdog_leaves_a_healthy_receive_path_alone():
    adapter = make_adapter()
    adapter._closing = False
    adapter._ws = FakeWs()
    adapter._reconnect_task = None

    async def live_listener():
        await asyncio.sleep(30)

    listener = asyncio.get_running_loop().create_task(live_listener())
    adapter._listener_task = listener

    scheduled = []
    adapter._schedule_reconnect = lambda reason: scheduled.append(reason)

    original = fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS
    fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = 0.01
    try:
        watchdog = asyncio.get_running_loop().create_task(adapter._gateway_watchdog_loop())
        await asyncio.sleep(0.1)
        watchdog.cancel()
        try:
            await watchdog
        except asyncio.CancelledError:
            pass
    finally:
        fluxer_adapter._GATEWAY_WATCHDOG_INTERVAL_SECONDS = original
        listener.cancel()

    assert scheduled == [], "a live listener must not be disturbed"
