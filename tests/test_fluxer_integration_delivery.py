"""The embed, reference and container fallbacks are finite and composable."""
import itertools
from unittest.mock import AsyncMock

import pytest

from test_fluxer_integration import HttpError, make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("order", list(itertools.permutations(("reference", "embed", "thread"))))
async def test_all_rejection_orders_reduce_once_and_keep_the_final_destination(order):
    adapter = make_adapter(reply_to_mode="all")
    adapter._remember_thread_id({"id": "thread"}, "parent")
    adapter.truncate_message = lambda *args, **kwargs: ["first", "second"]
    remaining = list(order)
    calls = []
    successes = []

    async def request(method, path, **kwargs):
        body = kwargs.get("json", {})
        calls.append((path, dict(body)))
        # The provider keeps rejecting a bad reference, not merely the first POST.
        if "reference" not in remaining and "message_reference" in body:
            raise HttpError(404, "UNKNOWN_MESSAGE")
        if remaining:
            kind = remaining.pop(0)
            if kind == "reference":
                assert "message_reference" in body
                raise HttpError(404, "UNKNOWN_MESSAGE")
            if kind == "embed":
                assert "embeds" in body
                raise HttpError(400, "embeds unsupported")
            assert path == "/channels/thread/messages"
            raise HttpError(404, "UNKNOWN_CHANNEL")
        successes.append(path)
        return {"id": f"message-{len(successes)}"}

    adapter._request = request
    result = await adapter.send(
        "thread", "text", reply_to="old-reference", metadata={"embeds": [{"title": "Card"}]},
    )

    assert result.success, (order, result.error, calls)
    assert successes == ["/channels/parent/messages", "/channels/parent/messages"]
    assert len(calls) == 5  # three explicit rejections and two delivered chunks
    assert "message_reference" not in calls[-1][1]
    assert "embeds" not in calls[-1][1]
    assert adapter._dead_containers["thread"] == "parent"
    assert adapter._destination_for_message("message-1", "thread") == "parent"
    assert (await adapter.send("thread", "later")).success
    assert calls[-1][0] == "/channels/parent/messages"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [TimeoutError("ambiguous"), HttpError(500, "server")])
async def test_reductions_do_not_retry_an_ambiguous_delivery(failure):
    adapter = make_adapter()
    adapter._remember_thread_id({"id": "thread"}, "parent")
    adapter._request = AsyncMock(side_effect=failure)
    result = await adapter.send("thread", "text", reply_to="old", metadata={"embeds": [{"title": "Card"}]})
    assert not result.success
    adapter._request.assert_awaited_once()
