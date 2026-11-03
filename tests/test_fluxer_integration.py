"""Cross-contribution contracts exercised against the integrated adapter."""
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


class HttpError(Exception):
    def __init__(self, code, body):
        self.response = type("Response", (), {"status_code": code, "text": body})()
        super().__init__(f"HTTP {code}")


def make_adapter(**options):
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "app.secret", "allow_all_users": True, "require_mention": False,
        "thread_replies": True, "delivery_verification": False, **options,
    }))


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "human", "bot"])
async def test_adoption_uses_server_ownership_not_the_create_refusal(owner):
    adapter = make_adapter()
    adapter.bot_user_id = "bot"
    adapter.create_thread = AsyncMock(side_effect=HttpError(400, "THREAD_ALREADY_CREATED_FOR_MESSAGE"))
    adapter._fetch_channel_name = AsyncMock(return_value="Human label")
    info = {"type": 11, "parent_id": "parent"}
    if owner is not None:
        info["owner_id"] = owner
    adapter._fetch_thread_info = AsyncMock(return_value=info)
    adapter._schedule_title_rename = lambda *args: None
    adapter._threads_pending["anchor"] = "parent"

    target, _, automatic = await adapter._resolve_outbound_target("anchor", None, None)

    assert target == "anchor"
    assert adapter._thread_is_bot_owned("anchor") is (None if owner is None else owner == "bot")
    assert automatic == ("anchor" if owner == "bot" else None)
    assert ("anchor" in adapter._mentioned_threads) is (owner == "bot")
    adapter._fetch_thread_info.assert_awaited_once_with("anchor")


@pytest.mark.asyncio
async def test_channel_title_resolution_uses_written_guard_not_display_label():
    adapter = make_adapter()
    adapter._session_origin_for_key = lambda key: ("parent", "user")
    adapter._record_auto_thread("parent", "anchor", "Human label", "user")
    adapter._remember_thread_guard("anchor", "Adapter placeholder")
    adapter.rename_thread = AsyncMock(return_value=False)

    await adapter._rename_thread_for_session("agent:main:fluxer:channel:parent:user", "New title")

    adapter.rename_thread.assert_awaited_once_with(
        "anchor", "New title", only_if_current_name="Adapter placeholder"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "human", "bot"])
async def test_adoption_adds_only_the_admitted_author_to_a_bot_owned_thread(owner):
    adapter = make_adapter()
    adapter.bot_user_id = "bot"
    adapter.create_thread = AsyncMock(side_effect=HttpError(400, "THREAD_ALREADY_CREATED_FOR_MESSAGE"))
    adapter._fetch_channel_name = AsyncMock(return_value="Placeholder")
    adapter._fetch_thread_info = AsyncMock(return_value={
        "type": 11, "parent_id": "parent", **({"owner_id": owner} if owner else {}),
    })
    adapter._inbound_author_by_message["anchor"] = "admitted-author"
    adapter._schedule_title_rename = lambda *args: None
    adapter._add_thread_member = AsyncMock()

    assert await adapter._start_thread_from_message("parent", "anchor") == "anchor"
    assert await adapter._start_thread_from_message("parent", "anchor") == "anchor"

    if owner == "bot":
        adapter._add_thread_member.assert_awaited_once_with("anchor", "admitted-author")
    else:
        adapter._add_thread_member.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["thread", "creation-failed", "deleted", "embed-deleted"])
async def test_clarify_session_anchor_routes_and_registers_in_the_final_container(monkeypatch, outcome):
    import sys
    import types

    # Keep the adapter routing real without requiring the host clarify service.
    stub = types.ModuleType("tools.clarify_gateway")
    stub.mark_awaiting_text = lambda cid: None
    monkeypatch.setitem(sys.modules, "tools.clarify_gateway", stub)
    adapter = make_adapter()
    adapter._is_multi_select_clarify = lambda cid: False
    adapter._threads_pending["anchor"] = "parent"
    adapter.create_thread = AsyncMock(return_value={"id": "anchor"})
    adapter._schedule_title_rename = lambda *args: None
    if outcome == "creation-failed":
        adapter.create_thread.side_effect = HttpError(403, "forbidden")
    posts = []
    failure_steps = ["embed", "deleted"] if outcome == "embed-deleted" else (["deleted"] if outcome == "deleted" else [])

    async def request(method, path, **kwargs):
        assert method == "POST"
        posts.append(path)
        if failure_steps:
            step = failure_steps.pop(0)
            if step == "embed":
                assert "embeds" in kwargs["json"]
                raise HttpError(400, "embeds unsupported")
            assert path == "/channels/anchor/messages"
            raise HttpError(404, "UNKNOWN_CHANNEL")
        return {"id": "card"}

    adapter._request = request
    exposed = []

    async def react(chat, message, emoji):
        assert len(adapter._pending_reaction_actions) == 2
        exposed.append(chat)

    adapter._add_reaction = react
    result = await adapter.send_clarify(
        "parent", "Choose", ["one", "two"], "clarification",
        "agent:main:fluxer:thread:anchor:user", metadata={"embeds": [{"title": "Card"}]},
    )
    assert result.success, result.error
    adapter.create_thread.assert_awaited_once()
    expected = "anchor" if outcome == "thread" else "parent"
    assert posts[-1] == f"/channels/{expected}/messages"
    assert exposed == [expected, expected]
    assert adapter._pending_clarify_cards["clarification"]["chat_id"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("info", [{}, {"type": 11, "parent_id": "parent"}, {"type": 11, "parent_id": "parent", "owner_id": "human"}])
async def test_adoption_cannot_reuse_stale_bot_privileges_when_fresh_owner_is_unknown(info):
    adapter = make_adapter()
    adapter.bot_user_id = "bot"
    adapter._remember_thread_id({"id": "anchor"}, "parent")
    adapter.create_thread = AsyncMock(side_effect=HttpError(400, "THREAD_ALREADY_CREATED_FOR_MESSAGE"))
    adapter._fetch_channel_name = AsyncMock(return_value="Placeholder")
    adapter._fetch_thread_info = AsyncMock(return_value=info)
    adapter._inbound_author_by_message["anchor"] = "admitted-author"
    adapter._add_thread_member = AsyncMock()
    adapter._schedule_title_rename = lambda *args: None

    assert await adapter._start_thread_from_message("parent", "anchor") == "anchor"
    target, _, automatic = await adapter._resolve_outbound_target("anchor", None, None)

    assert target == "anchor"
    assert adapter._thread_is_bot_owned("anchor") is not True
    assert "anchor" not in adapter._mentioned_threads
    assert automatic is None
    adapter._add_thread_member.assert_not_awaited()
