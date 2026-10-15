"""Long-response transport contract, permanent-error classification, heartbeat watchdog.

Three contract seams the host's stream consumer and retry ladder read directly:

- ``SendResult`` for a SPLIT payload: ``message_id`` is the LAST chunk (so later
  edits target the newest bubble) and ``continuation_message_ids`` carries the
  earlier ones in send order; a failure after some chunks landed sets
  ``raw_response["partial_overflow"]`` so nothing re-sends the visible head.
- An oversized FINAL edit must split across the existing message plus
  continuations, and a continuation failure must report
  ``partial_overflow`` / ``delivered_prefix`` / ``last_message_id`` rather than a
  bare success that hides the tail.
- Failure classification: a permanent 4xx is not retryable, and a read/write
  timeout on a POST that may already have landed is not retried either.
- The heartbeat watchdog must actually fire when the server stops acking.
"""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import AsyncMock

import httpx
import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

FLUXER_API = "https://api.fluxer.app/v1"


def _adapter(**extra_overrides) -> fluxer_adapter.FluxerAdapter:
    extra = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "delivery_verification": False,
    }
    extra.update(extra_overrides)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=extra))


def _status_error(status: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", f"{FLUXER_API}/channels/chan-1/messages")
    response = httpx.Response(status, request=request, text=body)
    return httpx.HTTPStatusError(f"{status} error", request=request, response=response)


def _chunk_indicator_stripped(chunk: str) -> str:
    return re.sub(r" \(\d+/\d+\)$", "", chunk)


# ── a split send reports the LAST id plus the earlier ones ─────────────────────


@pytest.mark.asyncio
async def test_a_split_send_reports_the_last_id_and_the_earlier_continuations():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=[{"id": "chunk-1"}, {"id": "chunk-2"}])

    result = await adapter.send("chan-1", "x" * 4500)

    assert result.success is True
    # The host retargets edits at ``message_id``; that must be the newest bubble.
    assert result.message_id == "chunk-2"
    assert result.continuation_message_ids == ("chunk-1",)
    assert result.raw_response["message_ids"] == ["chunk-1", "chunk-2"]


@pytest.mark.asyncio
async def test_a_single_chunk_send_carries_no_continuations():
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={"id": "only-1"})

    result = await adapter.send("chan-1", "short")

    assert result.success is True
    assert result.message_id == "only-1"
    assert result.continuation_message_ids == ()


@pytest.mark.asyncio
async def test_a_mid_split_failure_reports_partial_overflow_and_the_delivered_head():
    """A failure after chunk 1 landed must not let the host re-send chunk 1."""
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=[{"id": "chunk-1"}, _status_error(500, "boom")])

    result = await adapter.send("chan-1", "y" * 9000)

    assert result.success is False
    raw = result.raw_response
    assert raw["partial_overflow"] is True
    assert raw["last_message_id"] == "chunk-1"
    assert raw["delivered_chunks"] == 1
    assert raw["continuation_message_ids"] == ()
    assert result.message_id == "chunk-1"
    # Exactly two attempts: the visible head is never sent again.
    assert adapter._request.await_count == 2


@pytest.mark.asyncio
async def test_a_failure_before_any_chunk_landed_is_a_plain_failure():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=_status_error(500, "boom"))

    result = await adapter.send("chan-1", "z" * 9000)

    assert result.success is False
    assert not (isinstance(result.raw_response, dict)
                and result.raw_response.get("partial_overflow"))


# ── an oversized final edit splits instead of silently truncating ──────────────


def _record_requests(adapter, responses):
    """Record every (method, path, body-copy) and serve ``responses`` in order."""
    calls: list[tuple[str, str, dict]] = []
    queue = list(responses)

    async def fake_request(method, path, **kwargs):
        # Copy: the adapter mutates its payload dicts in place on fallback paths.
        calls.append((method, path, dict(kwargs.get("json") or {})))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    adapter._request = fake_request
    return calls


@pytest.mark.asyncio
async def test_an_oversized_final_edit_splits_and_reports_the_last_id():
    adapter = _adapter()
    calls = _record_requests(adapter, [{"id": "preview-1"}, {"id": "cont-1"}])

    result = await adapter.edit_message("chan-1", "preview-1", "w" * 4500, finalize=True)

    assert result.success is True
    assert result.message_id == "cont-1"
    assert result.continuation_message_ids == ("preview-1",) or result.continuation_message_ids == ("cont-1",)
    methods = [c[0] for c in calls]
    # First chunk edits the preview in place; the tail is a fresh continuation.
    assert methods[0] == "PATCH"
    assert methods[1] == "POST"
    assert calls[0][1] == "/channels/chan-1/messages/preview-1"
    assert calls[1][1] == "/channels/chan-1/messages"
    # Every chunk stays inside the UTF-16 cap.
    assert all(fluxer_adapter.utf16_len(c[2]["content"]) <= fluxer_adapter.MAX_MESSAGE_LENGTH
               for c in calls)


@pytest.mark.asyncio
async def test_a_continuation_failure_reports_partial_overflow_not_success():
    """The tail must never be hidden behind a bare success."""
    adapter = _adapter()
    calls = _record_requests(adapter, [{"id": "preview-1"}, _status_error(500, "boom")])

    result = await adapter.edit_message("chan-1", "preview-1", "v" * 9000, finalize=True)

    assert result.success is False
    raw = result.raw_response
    assert raw["partial_overflow"] is True
    assert raw["last_message_id"] == "preview-1"
    assert raw["delivered_chunks"] == 1
    assert raw["total_chunks"] > 1
    # The delivered prefix is the text actually on screen, indicators stripped, so
    # the host can send only the missing tail.
    assert "v" * 100 in raw["delivered_prefix"]
    assert not raw["delivered_prefix"].endswith("(1/2)")
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_a_fitting_final_edit_still_takes_the_single_patch_path():
    adapter = _adapter()
    calls = _record_requests(adapter, [{"id": "preview-1"}])

    result = await adapter.edit_message("chan-1", "preview-1", "short answer", finalize=True)

    assert result.success is True
    assert [c[0] for c in calls] == ["PATCH"]
    assert calls[0][2]["content"] == "short answer"


@pytest.mark.asyncio
async def test_an_oversized_interim_edit_truncates_without_splitting():
    """Mid-stream splitting moves the edit target and loops; only finals split."""
    adapter = _adapter()
    calls = _record_requests(adapter, [{"id": "preview-1"}])

    result = await adapter.edit_message("chan-1", "preview-1", "u" * 9000, finalize=False)

    assert result.success is True
    assert [c[0] for c in calls] == ["PATCH"]
    assert fluxer_adapter.utf16_len(calls[0][2]["content"]) <= fluxer_adapter.MAX_MESSAGE_LENGTH


# ── permanent refusals are not retryable; a landed POST is never re-sent ───────


@pytest.mark.asyncio
async def test_a_permanent_4xx_refusal_is_not_retryable():
    adapter = _adapter()
    adapter._request = AsyncMock(
        side_effect=_status_error(403, '{"code":"MISSING_PERMISSIONS"}'))

    result = await adapter.send("chan-1", "hello")

    assert result.success is False
    assert result.retryable is False
    assert result.error_kind == "forbidden"


@pytest.mark.asyncio
async def test_a_bad_request_is_not_retryable():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=_status_error(400, '{"code":"SERVER_ERROR"}'))

    result = await adapter.send("chan-1", "hello")

    assert result.success is False
    assert result.retryable is False


@pytest.mark.asyncio
async def test_a_server_error_is_still_retryable():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=_status_error(500, "boom"))

    result = await adapter.send("chan-1", "hello")

    assert result.success is False
    assert result.retryable is True


@pytest.mark.asyncio
async def test_a_connect_failure_is_retryable_but_a_read_timeout_is_not():
    """A read timeout may already have been accepted: retrying duplicates it."""
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    refused = await adapter.send("chan-1", "hello")
    assert refused.success is False
    assert refused.retryable is True

    adapter2 = _adapter()
    adapter2._request = AsyncMock(side_effect=httpx.ReadTimeout(""))
    timed_out = await adapter2.send("chan-1", "hello")
    assert timed_out.success is False
    assert timed_out.retryable is False
    # httpx timeouts stringify to ""; the error must still name the class, or the
    # host's own timeout check reads it as a formatting failure and re-sends.
    assert "timeout" in (timed_out.error or "").lower()


@pytest.mark.asyncio
async def test_a_permanent_edit_refusal_is_not_retryable():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=_status_error(404, '{"code":"UNKNOWN_MESSAGE"}'))

    result = await adapter.edit_message("chan-1", "preview-1", "hello", finalize=True)

    assert result.success is False
    assert result.retryable is False


# ── the length contract the host reads off the adapter ────────────────────────


def test_the_adapter_declares_the_fluxer_length_cap_and_unit():
    adapter = _adapter()

    # Host default is 4096 codepoints; Fluxer counts UTF-16 units and caps at 4000.
    assert fluxer_adapter.FluxerAdapter.MAX_MESSAGE_LENGTH == 4000
    assert adapter.message_len_fn("abc") == 3
    assert adapter.message_len_fn("😀") == 2
    resolved = getattr(adapter, "max_message_length_for_chat", None)
    if callable(resolved):
        assert resolved("chan-1") == 4000


# ── the heartbeat watchdog must actually fire ─────────────────────────────────


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def monotonic(self) -> float:
        return self.t


class _FakeWS:
    def __init__(self, *, ack: bool, adapter, clock) -> None:
        self.sent: list[dict] = []
        self.closed = False
        self._ack = ack
        self._adapter = adapter
        self._clock = clock

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))
        if self._ack:
            self._adapter._awaiting_heartbeat_ack = False
            self._adapter._last_heartbeat_ack_at = self._clock.t

    async def close(self) -> None:
        self.closed = True


class _FakeTime:
    """Stand-in for the adapter's ``time`` module so the event loop keeps real time."""

    def __init__(self, clock: "_FakeClock") -> None:
        self._clock = clock

    def monotonic(self) -> float:
        return self._clock.t

    def time(self) -> float:
        return self._clock.t


async def _drive_heartbeat(adapter, monkeypatch, *, ack: bool, tick_limit: int = 12):
    """Run the heartbeat loop on a fake clock; stop it after ``tick_limit`` ticks."""
    clock = _FakeClock()
    ws = _FakeWS(ack=ack, adapter=adapter, clock=clock)
    adapter._ws = ws
    adapter._closing = False
    # Patch ONLY the adapter's view of time: replacing the real ``time.monotonic``
    # freezes the event loop's own clock and the run never returns.
    monkeypatch.setattr(fluxer_adapter, "time", _FakeTime(clock))
    real_sleep = asyncio.sleep
    ticks = {"n": 0}

    async def fake_sleep(seconds: float) -> None:
        clock.t += seconds
        ticks["n"] += 1
        if ticks["n"] >= tick_limit:
            # A clean stop for a healthy run: the loop's own exit condition.
            adapter._closing = True
        await real_sleep(0)

    monkeypatch.setattr(fluxer_adapter.asyncio, "sleep", fake_sleep)
    await asyncio.wait_for(adapter._heartbeat_loop(1000), timeout=5)
    return ws, ticks["n"]


@pytest.mark.asyncio
async def test_the_heartbeat_watchdog_fires_when_no_ack_arrives(monkeypatch):
    adapter = _adapter()
    ws, ticks = await _drive_heartbeat(adapter, monkeypatch, ack=False)

    assert ws.closed is True, "a missing heartbeat ACK must close the websocket"
    # It fires once the window (2.5 intervals) has elapsed, not after the tick limit.
    assert ticks <= 5


@pytest.mark.asyncio
async def test_the_heartbeat_watchdog_stays_quiet_while_acks_arrive(monkeypatch):
    adapter = _adapter()
    ws, ticks = await _drive_heartbeat(adapter, monkeypatch, ack=True)

    assert ws.closed is False
    assert ticks > 5  # ran to the tick limit without tripping
    assert adapter._last_heartbeat_ack_at is not None
