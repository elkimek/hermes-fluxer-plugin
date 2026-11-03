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
@pytest.mark.parametrize("operation", ["send", "parent", "edit"])
async def test_embed_rejection_logs_omit_request_details(caplog, operation):
    adapter = make_adapter({"thread_replies": operation == "parent"})
    private_marker = "synthetic-private-request-marker"
    request = httpx.Request(
        "POST", f"https://example.invalid/channels/{private_marker}/messages?token={private_marker}"
    )
    response = httpx.Response(400, request=request, json={"embeds": ["invalid"]})
    rejected = httpx.HTTPStatusError(
        f"request rejected: {private_marker}", request=request, response=response
    )
    errors = []
    if operation == "parent":
        missing = RuntimeError("10003 UNKNOWN_CHANNEL")
        missing.response = SimpleNamespace(status_code=404)
        errors.append(missing)
        adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    adapter._request = AsyncMock(side_effect=[*errors, rejected, {"id": "text"}])
    with caplog.at_level("WARNING", logger=fluxer_adapter.__name__):
        if operation == "edit":
            result = await adapter.edit_message(
                "channel", "message", "hi", metadata={"embeds": [{"title": "t"}]}
            )
        else:
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


@pytest.mark.parametrize("path,limit", [
    (("title",), 256),
    (("description",), 4096),
    (("footer", "text"), 2048),
    (("author", "name"), 256),
    (("fields", 0, "name"), 256),
    (("fields", 0, "value"), 1024),
    (("image", "description"), 4096),
    (("thumbnail", "description"), 4096),
])
def test_embed_text_caps_use_utf16_without_mutating_input(path, limit):
    from copy import deepcopy

    # Leave one UTF-16 unit at the boundary: the next emoji must not be split.
    text = "x" * (limit - 1) + "😀" * 5
    source = {
        "title": "card", "description": "body", "footer": {"text": "footer"},
        "author": {"name": "author"}, "fields": [{"name": "field", "value": "value"}],
        "image": {"url": "https://example.invalid/image.png", "description": "image"},
        "thumbnail": {"url": "https://example.invalid/thumb.png", "description": "thumb"},
    }
    target = source
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = text
    original = deepcopy(source)
    normalized = fluxer_adapter.FluxerAdapter._normalize_outbound_embeds(source)[0]
    capped = normalized
    for key in path:
        capped = capped[key]
    assert fluxer_adapter.utf16_len(capped) <= limit
    assert capped == "x" * (limit - 1)
    assert source == original


def test_embed_field_count_cap_preserves_input():
    fields = [{"name": "field", "value": "value"} for _ in range(26)]
    source = {"title": "t", "fields": fields}
    normalized = fluxer_adapter.FluxerAdapter._normalize_outbound_embeds(source)
    assert len(normalized[0]["fields"]) == 25
    assert len(source["fields"]) == 26


@pytest.mark.asyncio
@pytest.mark.parametrize("first_error", ["embed", "thread"])
@pytest.mark.parametrize("reply_mode", ["off", "first", "all"])
async def test_embed_and_deleted_thread_fallbacks_compose_in_either_order(first_error, reply_mode):
    from copy import deepcopy

    adapter = make_adapter({"thread_replies": True, "reply_to_mode": reply_mode})
    missing = httpx.HTTPStatusError(
        "unknown channel", request=httpx.Request("POST", "https://example.invalid/messages"),
        response=httpx.Response(404, json={"code": 10003}),
    )
    rejected = httpx.HTTPStatusError(
        "invalid embed", request=httpx.Request("POST", "https://example.invalid/messages"),
        response=httpx.Response(422, json={"embeds": ["invalid"]}),
    )
    errors = [rejected, missing] if first_error == "embed" else [missing, rejected]
    snapshots = []

    async def request(method, path, **kwargs):
        snapshots.append((path, deepcopy(kwargs["json"])))
        if len(snapshots) <= len(errors):
            raise errors[len(snapshots) - 1]
        return {"id": f"output-{len(snapshots)}"}

    adapter._request = AsyncMock(side_effect=request)
    adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    result = await adapter.send(
        "channel", "x" * 5000, reply_to="trigger", metadata={"embeds": [{"title": "t"}]}
    )
    assert result.success
    assert len(snapshots) == 4  # two explicit rejections, then two accepted chunks
    assert snapshots[0][0] == "/channels/doomed-thread/messages"
    assert all(path == "/channels/channel/messages" for path, _ in snapshots[2:])
    assert snapshots[0][1]["embeds"]
    assert "embeds" not in snapshots[2][1]
    assert "embeds" not in snapshots[3][1]
    assert bool(snapshots[2][1].get("message_reference")) is (reply_mode != "off")
    assert bool(snapshots[3][1].get("message_reference")) is (reply_mode == "all")
    for message_id in result.raw_response["message_ids"]:
        assert adapter._destination_for_message(message_id, "other") == "channel"
    assert adapter._outbound_embeds_supported is False
    # Thread abandonment sticks for the turn, not only this chunk.
    later = await adapter.send("channel", "later", reply_to="trigger")
    assert later.success
    assert snapshots[-1][0] == "/channels/channel/messages"
    adapter.create_thread.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["send", "edit", "parent"])
async def test_unrelated_rejection_cannot_reenable_embeds_disabled_by_sibling(operation):
    adapter = make_adapter({"thread_replies": operation == "parent"})
    rejected = RuntimeError("400 content error")
    rejected.response = SimpleNamespace(status_code=400, text='{"content": "invalid"}')
    missing = RuntimeError("10003 UNKNOWN_CHANNEL")
    missing.response = SimpleNamespace(status_code=404)
    snapshots = []

    async def request(method, path, **kwargs):
        snapshots.append(dict(kwargs["json"]))
        if operation == "parent" and len(snapshots) == 1:
            raise missing
        if "embeds" in kwargs["json"]:
            # Another in-flight request has established that embeds are unsupported.
            adapter._outbound_embeds_supported = False
            raise rejected
        return {"id": "text"}

    adapter._request = AsyncMock(side_effect=request)
    adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    metadata = {"embeds": [{"title": "t"}]}
    if operation == "edit":
        result = await adapter.edit_message("channel", "message", "hi", metadata=metadata)
    else:
        result = await adapter.send("channel", "hi", reply_to="trigger", metadata=metadata)
    assert result.success
    assert "embeds" in snapshots[0]
    assert "embeds" not in snapshots[-1]
    assert adapter._outbound_embeds_supported is False


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503, None])
@pytest.mark.parametrize("operation", ["send", "edit"])
async def test_embed_fallback_never_retries_ambiguous_or_non_body_errors(operation, status):
    adapter = make_adapter()
    failure = RuntimeError("request failed")
    failure.response = SimpleNamespace(status_code=status)
    adapter._request = AsyncMock(side_effect=failure)
    metadata = {"embeds": [{"title": "t"}]}
    if operation == "edit":
        result = await adapter.edit_message("channel", "message", "hi", metadata=metadata)
    else:
        result = await adapter.send("channel", "hi", metadata=metadata)
    assert not result.success
    assert adapter._request.await_count == 1
    assert adapter._outbound_embeds_supported is True


@pytest.mark.asyncio
@pytest.mark.parametrize("first_error", ["embed", "thread", "both"])
@pytest.mark.parametrize("status", [400, 422, 429, 503, None])
async def test_reduced_payload_stops_retrying_after_fallbacks_are_consumed(first_error, status):
    adapter = make_adapter({"thread_replies": first_error != "embed"})
    missing = RuntimeError("10003 UNKNOWN_CHANNEL")
    missing.response = SimpleNamespace(status_code=404)
    rejected = RuntimeError("invalid embed")
    rejected.response = SimpleNamespace(status_code=400, text='{"embeds": "invalid"}')
    terminal = RuntimeError("terminal failure")
    terminal.response = SimpleNamespace(status_code=status, text='{"embeds": "invalid"}')
    errors = {"embed": [rejected], "thread": [missing], "both": [missing, rejected]}[first_error]
    # In the thread-only case no embeds are present, so no embed fallback is available.
    metadata = {"embeds": [{"title": "t"}]} if first_error != "thread" else None
    adapter._request = AsyncMock(side_effect=[*errors, terminal])
    adapter.create_thread = AsyncMock(return_value={"id": "doomed-thread"})
    result = await adapter.send("channel", "hi", reply_to="trigger", metadata=metadata)
    assert not result.success
    assert adapter._request.await_count == len(errors) + 1
