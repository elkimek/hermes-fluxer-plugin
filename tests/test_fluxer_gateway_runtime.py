"""Real Hermes gateway and read-only database integration, with no network."""

import asyncio
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType


def make_adapter():
    adapter = fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra={
        "bot_token": "test-only", "thread_replies": True,
        "delivery_verification": False, "allow_all_users": True,
        "require_mention": False, "gateway_state_updates": False,
    }))
    adapter._request = AsyncMock(return_value={"id": "synthetic-output"})
    adapter._multipart_request = AsyncMock(return_value={"id": "synthetic-upload"})
    adapter.create_thread = AsyncMock(side_effect=lambda chat, name, **kw: {
        "id": "thread-" + kw["message_id"], "name": name,
    })
    return adapter


@pytest.mark.asyncio
async def test_real_gateway_delivers_concurrent_progress_and_final_in_own_threads():
    pytest.importorskip("gateway.session")
    adapter = make_adapter()
    adapter.config.typing_indicator = False
    started = []
    both = asyncio.Event()

    async def handler(incoming):
        started.append(incoming.message_id)
        if len(started) == 2:
            both.set()
        await both.wait()
        await adapter.send_typing("channel")
        assert (await adapter.send("channel", "progress-" + incoming.message_id)).success
        await adapter.stop_typing("channel")
        return "final-" + incoming.message_id

    adapter.set_message_handler(handler)
    try:
        for anchor, user in [("message-a", "alice"), ("message-b", "bob")]:
            source = adapter.build_source(chat_id="channel", chat_type="group", user_id=user,
                                          message_id=anchor)
            incoming = MessageEvent(text="synthetic request", source=source,
                                    message_id=anchor, message_type=MessageType.TEXT)
            await adapter.handle_message(incoming)
        while adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*tuple(adapter._background_tasks)), timeout=5)
        delivered = {call.kwargs["json"].get("content"): call.args[1]
                     for call in adapter._request.await_args_list
                     if call.args[1].endswith("/messages")}
        for anchor in ["message-a", "message-b"]:
            for kind in ["progress-", "final-"]:
                assert delivered[kind + anchor] == "/channels/thread-" + anchor + "/messages"
        assert not adapter._typing_tasks
        assert not adapter._turn_anchor
        assert not adapter._turn_thread_disabled
    finally:
        await adapter.disconnect()


def test_named_profile_uses_real_owned_read_only_database(monkeypatch, tmp_path):
    state = pytest.importorskip("hermes_state")
    constants = pytest.importorskip("hermes_constants")
    profiles = pytest.importorskip("hermes_cli.profiles")
    homes = {name: tmp_path / name for name in ["default", "secondary"]}
    for name, home in homes.items():
        home.mkdir()
        writer = state.SessionDB(db_path=home / "state.db")
        writer.create_session(session_id=name, source="fluxer", chat_id="same-channel",
                              user_id="same-user", session_key="agent:main:fluxer:group:same-channel:same-user")
        writer.set_session_title(name, name + " synthetic title")
        writer.close()
    monkeypatch.setattr(profiles, "get_profile_dir", lambda name: homes[name])
    marker = constants.set_hermes_home_override(str(homes["default"]))
    adapter = make_adapter()
    injected = object()
    BasePlatformAdapter.set_session_store(adapter, injected)
    adapter.set_owner_profile("secondary")
    try:
        assert adapter._session_title_for_chat("same-channel", "same-user") == "secondary synthetic title"
        assert adapter._session_db.read_only
        assert adapter._session_db.db_path == homes["secondary"] / "state.db"
        assert adapter._session_store is injected
    finally:
        adapter._close_session_store()
        constants.reset_hermes_home_override(marker)
