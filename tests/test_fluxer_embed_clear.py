"""An explicit empty embed list means clear, never silent preservation."""
from unittest.mock import AsyncMock

import httpx
import pytest

import adapter
from gateway.config import PlatformConfig


def make_adapter():
    instance = adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "delivery_verification": False,
        "gateway_state_updates": False,
    }))
    instance._request = AsyncMock(return_value={"id": "synthetic-message", "embeds": []})
    return instance


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,supported", [(True, True), (False, True), (True, False)])
async def test_explicit_empty_embeds_clear_even_when_new_cards_are_disabled(enabled, supported):
    instance = make_adapter()
    instance._outbound_embeds_enabled = enabled
    instance._outbound_embeds_supported = supported
    metadata = {"embeds": []}
    result = await instance.edit_message("synthetic-channel", "synthetic-message", "plain text", metadata=metadata)
    assert result.success
    assert instance._request.await_args.kwargs["json"]["embeds"] == []
    assert metadata == {"embeds": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [None, {}, {"embeds": None}, {"embeds": "junk"}, {"embeds": [None, 7]}])
async def test_absent_or_malformed_embeds_do_not_clear_existing_cards(metadata):
    instance = make_adapter()
    result = await instance.edit_message("synthetic-channel", "synthetic-message", "plain text", metadata=metadata)
    assert result.success
    assert "embeds" not in instance._request.await_args.kwargs["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 422])
async def test_rejected_explicit_clear_does_not_report_content_only_success(status, caplog):
    instance = make_adapter()
    marker = "private-edit-diagnostic"
    request = httpx.Request("PATCH", "https://example.invalid/" + marker)
    response = httpx.Response(status, request=request, json={"embeds": ["unsupported"]})
    rejected = httpx.HTTPStatusError(marker, request=request, response=response)
    instance._request = AsyncMock(side_effect=[rejected, {"id": "synthetic-message"}])
    with caplog.at_level("WARNING", logger=adapter.__name__):
        result = await instance.edit_message("synthetic-channel", "synthetic-message", "plain text", metadata={"embeds": []})
    assert result.success is False
    assert instance._request.await_count == 1
    assert marker not in caplog.text
    assert "synthetic-message" not in caplog.text
