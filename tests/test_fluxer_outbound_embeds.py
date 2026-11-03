"""Outbound embeds: metadata surface, first-chunk placement, knob, degrade gate."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


def make_adapter(extra=None):
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "delivery_verification": False,
        "allow_all_users": True, "require_mention": False,
        "gateway_state_updates": False, **(extra or {}),
    }))
    adapter._request = AsyncMock(return_value={"id": "synthetic-output"})
    return adapter


def sent_payloads(adapter):
    return [call.kwargs["json"] for call in adapter._request.await_args_list]


@pytest.mark.asyncio
async def test_embeds_ride_metadata_and_first_chunk_only():
    adapter = make_adapter()
    embed = {"title": "verdict", "description": "gate card", "color": 0x3366CC}
    result = await adapter.send(
        "channel", "x" * 5000,  # forces a split: continuation must stay text-only
        metadata={"embeds": [embed]},
    )
    assert result.success
    payloads = sent_payloads(adapter)
    assert len(payloads) >= 2
    assert payloads[0]["embeds"][0]["type"] == "rich"
    assert payloads[0]["embeds"][0]["title"] == "verdict"
    assert all("embeds" not in p for p in payloads[1:])


@pytest.mark.asyncio
async def test_embed_knob_off_strips_embeds():
    adapter = make_adapter({"outbound_embeds": False})
    await adapter.send("channel", "hi", metadata={"embeds": [{"title": "t"}]})
    assert all("embeds" not in p for p in sent_payloads(adapter))


@pytest.mark.asyncio
async def test_embed_rejection_degrades_to_text_and_sticks():
    adapter = make_adapter()
    rejected = RuntimeError("400 INVALID_FORM_BODY")
    rejected.response = SimpleNamespace(status_code=400)  # type: ignore[attr-defined]
    adapter._request = AsyncMock(side_effect=[
        rejected, {"id": "fallback"},
        {"id": "next"},
    ])
    result = await adapter.send("channel", "hi", metadata={"embeds": [{"title": "t"}]})
    assert result.success
    first, second = sent_payloads(adapter)
    assert "embeds" not in second  # retry without embeds
    assert adapter._outbound_embeds_supported is False
    # Next send does not even attempt embeds.
    await adapter.send("channel", "again", metadata={"embeds": [{"title": "t"}]})
    assert "embeds" not in sent_payloads(adapter)[2]


@pytest.mark.asyncio
async def test_transient_send_failure_keeps_embeds_enabled():
    adapter = make_adapter()
    boom = RuntimeError("upstream timeout")
    boom.response = SimpleNamespace(status_code=503)  # type: ignore[attr-defined]
    adapter._request = AsyncMock(side_effect=boom)
    result = await adapter.send("channel", "hi", metadata={"embeds": [{"title": "t"}]})
    assert not result.success  # the error propagates; no blind retry
    assert adapter._outbound_embeds_supported is True  # not degraded


@pytest.mark.asyncio
async def test_edit_with_junk_embeds_leaves_existing_embeds():
    adapter = make_adapter()
    await adapter.edit_message("channel", "m1", "edited", metadata={"embeds": "junk"})
    payload = adapter._request.await_args.kwargs["json"]
    assert "embeds" not in payload  # junk must not PATCH an empty list over existing embeds


@pytest.mark.asyncio
async def test_thread_gone_fallback_keeps_embeds_enabled():
    adapter = make_adapter({"thread_replies": True})
    missing = RuntimeError("10003 UNKNOWN_CHANNEL")
    missing.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
    adapter._request = AsyncMock(side_effect=[missing, {"id": "parent"}])
    adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    result = await adapter.send("channel", "hi", reply_to="trigger",
                                metadata={"embeds": [{"title": "t"}]})
    assert result.success
    assert adapter._outbound_embeds_supported is True  # embeds were not the fault
    assert sent_payloads(adapter)[-1].get("embeds")  # embeds landed on the fallback


def test_normalize_shapes_and_caps():
    norm = fluxer_adapter.FluxerAdapter._normalize_outbound_embeds
    assert norm(None) == []
    assert norm("nope") == []
    one = norm({"title": "t" * 300, "type": "link"})
    assert one[0]["type"] == "rich" and len(one[0]["title"]) == 256
    many = norm([{"title": str(i)} for i in range(15)] + ["junk", None])
    assert len(many) == 10


@pytest.mark.asyncio
async def test_edit_message_embeds_only_when_provided():
    adapter = make_adapter()
    await adapter.edit_message("channel", "mid-1", "updated",
                               metadata={"embeds": {"title": "card"}})
    await adapter.edit_message("channel", "mid-2", "updated")
    first, second = sent_payloads(adapter)
    assert first["embeds"] == [{"type": "rich", "title": "card"}]
    assert "embeds" not in second  # absent metadata must not clear existing embeds


@pytest.mark.asyncio
async def test_edit_degrades_on_body_rejection_and_keeps_session_enabled():
    adapter = make_adapter()
    rejected = RuntimeError("400 bad message_reference")
    rejected.response = SimpleNamespace(status_code=400, text='{"message": "invalid message reference"}')  # type: ignore[attr-defined]
    adapter._request = AsyncMock(side_effect=[rejected, {"id": "m1"}])
    result = await adapter.edit_message("channel", "m1", "edited", metadata={"embeds": [{"title": "t"}]})
    assert result.success
    second = adapter._request.await_args_list[1].kwargs["json"]
    assert "embeds" not in second
    assert adapter._outbound_embeds_supported is True  # body blamed a different field


@pytest.mark.asyncio
async def test_edit_degrade_disabled_when_body_blames_embeds():
    adapter = make_adapter()
    rejected = RuntimeError("400 INVALID_FORM_BODY")
    rejected.response = SimpleNamespace(status_code=400, text='{"embeds": "invalid"}')  # type: ignore[attr-defined]
    adapter._request = AsyncMock(side_effect=[rejected, {"id": "m1"}])
    await adapter.edit_message("channel", "m1", "edited", metadata={"embeds": [{"title": "t"}]})
    assert adapter._outbound_embeds_supported is False

@pytest.mark.asyncio
async def test_embed_rejection_uses_payload_not_session_flag():
    """A concurrent sibling may flip the session flag mid-flight; this payload
    still carries embeds and must still degrade to text."""
    adapter = make_adapter()
    rejected = RuntimeError("400 INVALID_FORM_BODY")
    rejected.response = SimpleNamespace(  # type: ignore[attr-defined]
        status_code=400, text='{"embeds":["invalid"]}'
    )
    snapshots = []

    async def flip_then_maybe_ok(*_a, **kwargs):
        # Copy before the degrade path mutates the same dict in place.
        snapshots.append(dict(kwargs.get("json") or {}))
        if len(snapshots) == 1:
            adapter._outbound_embeds_supported = False
            raise rejected
        return {"id": "fallback"}

    adapter._request = AsyncMock(side_effect=flip_then_maybe_ok)
    result = await adapter.send("channel", "hi", metadata={"embeds": [{"title": "t"}]})
    assert result.success
    assert len(snapshots) == 2
    assert "embeds" in snapshots[0]
    assert "embeds" not in snapshots[1]


@pytest.mark.asyncio
async def test_thread_gone_parent_embed_rejection_degrades():
    """Thread 404 then parent 400 on embeds must still deliver content-only."""
    adapter = make_adapter({"thread_replies": True})
    missing = RuntimeError("10003 UNKNOWN_CHANNEL")
    missing.response = SimpleNamespace(status_code=404)  # type: ignore[attr-defined]
    rejected = RuntimeError("400 INVALID_FORM_BODY")
    rejected.response = SimpleNamespace(  # type: ignore[attr-defined]
        status_code=400, text='{"embeds":["invalid"]}'
    )
    adapter._request = AsyncMock(side_effect=[missing, rejected, {"id": "parent-text"}])
    adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    result = await adapter.send(
        "channel", "hi", reply_to="trigger", metadata={"embeds": [{"title": "t"}]}
    )
    assert result.success
    payloads = sent_payloads(adapter)
    assert "embeds" not in payloads[-1]
    assert adapter._outbound_embeds_supported is False


@pytest.mark.asyncio
@pytest.mark.parametrize("parent_fallback", [False, True])
async def test_embed_rejection_logs_omit_request_details(caplog, parent_fallback):
    adapter = make_adapter({"thread_replies": parent_fallback})
    private_marker = "synthetic-private-request-marker"
    request = httpx.Request(
        "POST", f"https://example.invalid/channels/{private_marker}/messages?token={private_marker}"
    )
    response = httpx.Response(400, request=request, json={"embeds": ["invalid"]})
    rejected = httpx.HTTPStatusError(
        f"request rejected: {private_marker}", request=request, response=response
    )
    errors = []
    if parent_fallback:
        missing = RuntimeError("10003 UNKNOWN_CHANNEL")
        missing.response = SimpleNamespace(status_code=404)
        errors.append(missing)
        adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    adapter._request = AsyncMock(side_effect=[*errors, rejected, {"id": "text"}])
    with caplog.at_level("WARNING", logger=fluxer_adapter.__name__):
        result = await adapter.send(
            "channel", "hi", reply_to="trigger", metadata={"embeds": [{"title": "t"}]}
        )
    assert result.success
    assert private_marker not in caplog.text
    assert "400" in caplog.text
    assert "HTTPStatusError" in caplog.text


@pytest.mark.asyncio
async def test_unrelated_body_rejection_log_does_not_claim_embeds_disabled(caplog):
    adapter = make_adapter()
    rejected = RuntimeError("400 unrelated body error")
    rejected.response = SimpleNamespace(status_code=400, text='{"content": "invalid"}')
    adapter._request = AsyncMock(side_effect=[rejected, {"id": "text"}])
    with caplog.at_level("WARNING", logger=fluxer_adapter.__name__):
        result = await adapter.send("channel", "hi", metadata={"embeds": [{"title": "t"}]})
    assert result.success
    assert adapter._outbound_embeds_supported is True
    assert "disabling" not in caplog.text.lower()
    assert "enabled" in caplog.text.lower()

