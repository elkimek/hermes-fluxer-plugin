"""New routing diagnostics explain outcomes without exposing private context."""
import asyncio
import logging
import sys
import types
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer
from test_fluxer_integration import make_adapter


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["lookup", "poll", "title", "typing", "clarify", "guard"])
async def test_routing_diagnostics_do_not_include_private_ids_or_text(monkeypatch, caplog, operation):
    adapter = make_adapter()
    private = ("private-chat", "private-user", "private-session", "private-title", "private-label")
    caplog.set_level(logging.DEBUG, logger=fluxer.logger.name)
    if operation == "lookup":
        adapter._get_session_store = lambda: types.SimpleNamespace(find_session_by_origin=lambda **kw: None)
        assert adapter._session_title_for_chat(private[0], private[1]) is None
    elif operation == "poll":
        adapter._session_title_for_chat = lambda *a: private[3]
        adapter.rename_thread = AsyncMock(return_value=False)
        await adapter._rename_thread_when_titled(private[0], private[0], private[4], private[1])
    elif operation == "title":
        adapter._resolve_title_rename_target = lambda key: (private[0], private[4])
        adapter.rename_thread = AsyncMock(return_value=True)
        await adapter._rename_thread_for_session(private[2], private[3])
    elif operation == "typing":
        adapter._request = AsyncMock(return_value={})
        await adapter.send_typing(private[0])
        await asyncio.sleep(0)
        await adapter.stop_typing(private[0])
    elif operation == "clarify":
        import tools
        stub = types.ModuleType("tools.clarify_gateway")
        stub.resolve_gateway_clarify = lambda *a: True
        monkeypatch.setitem(sys.modules, "tools.clarify_gateway", stub)
        monkeypatch.setattr(tools, "clarify_gateway", stub, raising=False)
        adapter.retire_clarify_card = AsyncMock()
        await adapter._resolve_clarify_action({"clarify_id": private[2], "choice": private[3]}, user_id=private[1])
    else:
        adapter._fetch_channel_name = AsyncMock(return_value=private[3])
        assert not await adapter.rename_thread(private[0], "new", only_if_current_name=private[4])
    assert caplog.records, "Keep an outcome diagnostic, not silent failure"
    assert not any(value in caplog.text for value in private)
