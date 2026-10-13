"""Presence diagnostics never include configuration or websocket error text."""

import logging

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_send", [False, True])
async def test_presence_logs_only_fixed_outcomes(monkeypatch, caplog, fail_send):
    monkeypatch.delenv("FLUXER_PRESENCE_STATUS", raising=False)
    monkeypatch.delenv("FLUXER_PRESENCE_AFK", raising=False)
    adapter = fluxer_adapter.FluxerAdapter(
        PlatformConfig(enabled=True, extra={"bot_token": "test-only", "presence_status": "dnd"})
    )

    class Websocket:
        async def send(self, data):
            if fail_send:
                raise RuntimeError("private-websocket-diagnostic")

    adapter._ws = Websocket()
    with caplog.at_level(logging.INFO, logger=fluxer_adapter.__name__):
        await adapter._send_presence_update()

    assert "Fluxer presence update" in caplog.text
    assert "dnd" not in caplog.text
    assert "private-websocket-diagnostic" not in caplog.text
    if fail_send:
        assert "RuntimeError" in caplog.text
    else:
        assert "sent" in caplog.text
