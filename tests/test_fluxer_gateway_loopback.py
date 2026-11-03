"""Exercise gateway ownership against an actual loopback websocket server."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest
import websockets

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


@pytest.mark.asyncio
async def test_loopback_gateway_replaces_recovers_and_shuts_down(monkeypatch):
    established = asyncio.Queue()
    heartbeats = asyncio.Queue()
    sockets = []

    async def gateway(socket, _path=None):
        sockets.append(socket)
        await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 1000}}))
        try:
            async for raw in socket:
                message = json.loads(raw)
                if message["op"] == 2:
                    assert message["d"]["token"] == "test-only"
                    await socket.send(json.dumps({
                        "op": 0, "t": "READY", "d": {"user": {"id": "test-bot"}},
                    }))
                    established.put_nowait(socket)
                elif message["op"] == 1:
                    await socket.send(json.dumps({"op": 11}))
                    heartbeats.put_nowait(socket)
        except websockets.exceptions.ConnectionClosed:
            pass

    monkeypatch.setattr(fluxer_adapter, "_RECONNECT_BASE_DELAY", 0.01)
    async with websockets.serve(gateway, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
            "bot_token": "test-only", "gateway_state_updates": False,
            "backlog_recovery": False,
        }))
        adapter.gateway_url = f"ws://127.0.0.1:{port}"
        adapter._voice_supervisor.start = lambda: True
        adapter._voice_supervisor.stop = AsyncMock()
        try:
            assert await adapter.connect()
            first = await asyncio.wait_for(established.get(), 2)
            assert await adapter.wait_until_gateway_ready(2)
            assert await asyncio.wait_for(heartbeats.get(), 2) is first
            watchdog = adapter._watchdog_task
            adapter._schedule_reconnect("loopback replacement")
            replacement = adapter._reconnect_task
            assert replacement is not None
            second = await asyncio.wait_for(established.get(), 2)
            await asyncio.wait_for(replacement, 2)
            assert first is not second
            assert len(sockets) == 2, "retiring a real listener must not cause reconnect churn"
            assert adapter._watchdog_task is watchdog
            # A real current-connection death still schedules another attempt.
            await second.close()
            third = await asyncio.wait_for(established.get(), 2)
            assert await adapter.wait_until_gateway_ready(2)
            assert third is not second
            assert len(sockets) == 3
            owned = [adapter._listener_task, adapter._heartbeat_task, watchdog]
            await adapter.disconnect()
            assert all(task.done() for task in owned if task is not None)
            assert adapter._ws is None
            assert adapter not in fluxer_adapter._LIVE_ADAPTERS
        finally:
            await adapter.disconnect()
