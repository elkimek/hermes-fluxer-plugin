"""Auto-thread naming: placeholder from the trigger, semantic rename from the title.

Mirrors the Discord adapter's two-phase thread naming so a Fluxer thread is
identifiable the moment it appears and carries the session title afterwards.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {"bot_token": "app.secret", "allow_all_users": True, "require_mention": False}
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


# ── placeholder derivation ──────────────────────────────────────────────────────


def test_derive_auto_thread_name_strips_mention_markup():
    name = fluxer_adapter._derive_auto_thread_name(
        "<@1553261288962924544>  can   we  fix\n the thread naming"
    )

    assert name == "can we fix the thread naming"


def test_derive_auto_thread_name_falls_back_when_there_is_no_text():
    assert fluxer_adapter._derive_auto_thread_name("<@1553261288962924544>") == fluxer_adapter._DEFAULT_THREAD_NAME
    assert fluxer_adapter._derive_auto_thread_name("") == fluxer_adapter._DEFAULT_THREAD_NAME


def test_derive_auto_thread_name_caps_to_the_name_budget_in_utf16_units():
    name = fluxer_adapter._derive_auto_thread_name("x" * 200)

    assert fluxer_adapter.utf16_len(name) <= fluxer_adapter._THREAD_NAME_LIMIT
    assert name.endswith("...")

    emoji_name = fluxer_adapter._derive_auto_thread_name("\U0001f9ab" * 60)

    assert fluxer_adapter.utf16_len(emoji_name) <= fluxer_adapter._THREAD_NAME_LIMIT


# ── thread creation / reporting ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_start_thread_from_message_names_the_thread_from_the_trigger():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})

    thread_id = await adapter._start_thread_from_message("chan-1", "msg-1")

    assert thread_id == "th-1"
    assert adapter.create_thread.await_args.args[:2] == ("chan-1", "deploy the thing")
    assert adapter.auto_thread_info_for_chat("chan-1") == ("th-1", "deploy the thing")


@pytest.mark.asyncio
async def test_start_thread_from_message_reuses_the_thread_for_the_same_anchor():
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})

    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "th-1"
    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "th-1"
    adapter.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_wait_for_auto_thread_info_returns_a_recorded_thread_without_waiting():
    adapter = _adapter()
    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")

    info = await adapter.wait_for_auto_thread_info("chan-1", timeout=5)

    assert info == ("th-1", "deploy the thing")


@pytest.mark.asyncio
async def test_wait_for_auto_thread_info_wakes_when_a_thread_is_opened():
    adapter = _adapter()
    waiting = asyncio.create_task(adapter.wait_for_auto_thread_info("chan-1", timeout=5))
    await asyncio.sleep(0)

    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")

    assert await asyncio.wait_for(waiting, timeout=5) == ("th-1", "deploy the thing")


@pytest.mark.asyncio
async def test_wait_for_auto_thread_info_returns_none_when_no_thread_is_opened():
    adapter = _adapter()

    assert await adapter.wait_for_auto_thread_info("chan-1", timeout=0.01) is None


@pytest.mark.asyncio
async def test_forget_thread_drops_the_reported_auto_thread():
    adapter = _adapter()
    adapter._threads_by_anchor[("chan-1", "msg-1")] = "th-1"
    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")

    adapter._forget_thread("th-1")

    assert adapter.auto_thread_info_for_chat("chan-1") is None


# ── rename ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rename_thread_patches_the_channel_name():
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={"id": "th-1", "name": "Deploy the thing"})

    assert await adapter.rename_thread("th-1", "  Deploy   the thing  ") is True
    adapter._request.assert_awaited_once_with("PATCH", "/channels/th-1", json={"name": "Deploy the thing"})


@pytest.mark.asyncio
async def test_rename_thread_declines_when_the_name_was_changed_by_a_human():
    adapter = _adapter()
    adapter._fetch_channel_name = AsyncMock(return_value="my own thread name")
    adapter._request = AsyncMock()

    renamed = await adapter.rename_thread("th-1", "Deploy the thing", only_if_current_name="deploy the thing")

    assert renamed is False
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_thread_applies_when_the_name_still_matches_the_placeholder():
    adapter = _adapter()
    adapter._fetch_channel_name = AsyncMock(return_value="deploy the thing")
    adapter._request = AsyncMock(return_value={})

    assert await adapter.rename_thread("th-1", "Deploy the thing", only_if_current_name="deploy the thing") is True


@pytest.mark.asyncio
async def test_rename_thread_declines_when_the_name_cannot_be_read():
    adapter = _adapter()
    adapter._fetch_channel_name = AsyncMock(return_value=None)
    adapter._request = AsyncMock()

    assert await adapter.rename_thread("th-1", "Deploy the thing", only_if_current_name="deploy the thing") is False
    adapter._request.assert_not_awaited()


@pytest.mark.asyncio
async def test_rename_thread_reports_failure_instead_of_raising():
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=RuntimeError("MISSING_PERMISSIONS"))

    assert await adapter.rename_thread("th-1", "Deploy the thing") is False


@pytest.mark.asyncio
async def test_rename_thread_ignores_unknown_keyword_arguments_and_empty_names():
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={})

    assert await adapter.rename_thread("th-1", "") is False

    assert await adapter.rename_thread(
        "th-1", "Deploy the thing", prefer_connector_created=True, parent_chat_id="chan-1"
    ) is True


@pytest.mark.asyncio
async def test_rename_thread_truncates_to_the_name_budget():
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={})

    assert await adapter.rename_thread("th-1", "y" * 200) is True

    applied = adapter._request.await_args.kwargs["json"]["name"]
    assert fluxer_adapter.utf16_len(applied) <= fluxer_adapter._THREAD_NAME_LIMIT


@pytest.mark.asyncio
async def test_rename_thread_keeps_the_cached_placeholder_in_step():
    adapter = _adapter()
    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")
    adapter._request = AsyncMock(return_value={})

    await adapter.rename_thread("th-1", "Deploy the thing")

    assert adapter.auto_thread_info_for_chat("chan-1") == ("th-1", "Deploy the thing")
