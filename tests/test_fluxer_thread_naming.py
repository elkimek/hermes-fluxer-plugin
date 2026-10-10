"""Auto-thread naming: placeholder from the trigger, session title when there is one.

Mirrors the Discord adapter's two-phase thread naming, but self-contained in the
plugin: the gateway's session title is read back from the Hermes session store
rather than pushed by the host, so no Hermes core change is involved.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import types
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

SESSIONS_DDL = """
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    session_key TEXT,
    source TEXT,
    chat_id TEXT,
    chat_type TEXT,
    title TEXT,
    title_source TEXT,
    last_activity_at REAL
)
"""


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {"bot_token": "app.secret", "allow_all_users": True, "require_mention": False}
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


@pytest.fixture
def state_db(monkeypatch):
    """A stand-in Hermes session store, reached through HERMES_HOME."""
    home = tempfile.mkdtemp(prefix="fluxer-title-home-")
    db = Path(home) / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(SESSIONS_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", home)
    return db


def _add_session(db: Path, **row) -> None:
    values = {
        "id": "sess-1", "session_key": "agent:main:fluxer:channel:chan-1:user-1",
        "source": "fluxer", "chat_id": "chan-1", "chat_type": "channel",
        "title": None, "title_source": None, "last_activity_at": 1.0,
    }
    values.update(row)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO sessions (id, session_key, source, chat_id, chat_type, title, title_source, "
        "last_activity_at) VALUES (:id, :session_key, :source, :chat_id, :chat_type, :title, "
        ":title_source, :last_activity_at)",
        values,
    )
    conn.commit()
    conn.close()


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


# ── session title lookup ────────────────────────────────────────────────────────


def test_session_title_is_read_from_the_session_store(state_db):
    adapter = _adapter()
    _add_session(state_db, title="Deploy the thing", title_source="llm")

    assert adapter._session_title_for_chat("chan-1") == "Deploy the thing"


def test_session_title_ignores_derived_titles_and_other_chats(state_db):
    adapter = _adapter()
    _add_session(state_db, id="derived", title="deploy the thing", title_source="derived")
    _add_session(state_db, id="other", chat_id="chan-2", title="Other chat", title_source="llm")

    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_prefers_the_most_recent_row(state_db):
    adapter = _adapter()
    _add_session(state_db, id="old", title="Older", title_source="llm", last_activity_at=1.0)
    _add_session(state_db, id="new", title="Newer", title_source="llm", last_activity_at=9.0)

    assert adapter._session_title_for_chat("chan-1") == "Newer"


def test_session_title_is_none_without_a_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()

    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_survives_an_unreadable_store(state_db):
    state_db.write_bytes(b"not a database")
    adapter = _adapter()

    assert adapter._session_title_for_chat("chan-1") is None


def test_chat_id_is_resolved_from_a_session_key(state_db):
    adapter = _adapter()
    _add_session(state_db, session_key="agent:main:fluxer:channel:chan-9:user-1", chat_id="chan-9")

    assert adapter._chat_id_for_session("agent:main:fluxer:channel:chan-9:user-1") == "chan-9"
    assert adapter._chat_id_for_session("agent:main:fluxer:channel:nope:user-1") is None


# ── thread creation ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_thread_is_named_from_the_trigger_when_no_title_exists(state_db):
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})
    adapter._schedule_title_rename = lambda *args: None

    thread_id = await adapter._start_thread_from_message("chan-1", "msg-1")

    assert thread_id == "th-1"
    assert adapter.create_thread.await_args.args[:2] == ("chan-1", "deploy the thing")
    assert adapter.auto_thread_info_for_chat("chan-1") == ("th-1", "deploy the thing")


@pytest.mark.asyncio
async def test_thread_is_born_with_the_session_title_when_one_exists(state_db):
    adapter = _adapter()
    _add_session(state_db, title="Deploy the thing", title_source="llm")
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "Deploy the thing"})
    adapter._schedule_title_rename = lambda *args: None

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert adapter.create_thread.await_args.args[:2] == ("chan-1", "Deploy the thing")
    assert adapter.auto_thread_info_for_chat("chan-1") == ("th-1", "Deploy the thing")


@pytest.mark.asyncio
async def test_thread_without_a_title_yet_is_watched(state_db):
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert watched == [("chan-1", "th-1", "deploy the thing")]


@pytest.mark.asyncio
async def test_thread_creation_reuses_the_thread_for_the_same_anchor(state_db):
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})
    adapter._schedule_title_rename = lambda *args: None

    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "th-1"
    assert await adapter._start_thread_from_message("chan-1", "msg-1") == "th-1"
    adapter.create_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_forget_thread_drops_the_reported_auto_thread():
    adapter = _adapter()
    adapter._threads_by_anchor[("chan-1", "msg-1")] = "th-1"
    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")

    adapter._forget_thread("th-1")

    assert adapter.auto_thread_info_for_chat("chan-1") is None


# ── waiting for the title ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_thread_waiting_for_its_title_is_renamed(state_db):
    adapter = _adapter()
    _add_session(state_db, title="Deploy the thing", title_source="llm")
    adapter.rename_thread = AsyncMock(return_value=True)

    with patch.object(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_awaited_once_with(
        "th-1", "Deploy the thing", only_if_current_name="deploy the thing"
    )


@pytest.mark.asyncio
async def test_a_rename_declined_by_the_guard_is_not_retried(state_db):
    adapter = _adapter()
    _add_session(state_db, title="Deploy the thing", title_source="llm")
    adapter.rename_thread = AsyncMock(return_value=False)  # a human renamed the thread

    with patch.object(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_title_is_waited_for_until_it_appears(state_db):
    adapter = _adapter()
    adapter.rename_thread = AsyncMock(return_value=True)
    results = iter([None, None, "Deploy the thing"])

    def fake_lookup(chat_id):
        return next(results, "Deploy the thing")

    adapter._session_title_for_chat = fake_lookup
    with patch.object(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_wait_for_a_title_is_bounded(state_db):
    adapter = _adapter()
    adapter.rename_thread = AsyncMock()
    with patch.object(fluxer_adapter, "_TITLE_RENAME_WINDOW_SECONDS", 0.05), patch.object(
        fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01
    ):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_title_matching_the_placeholder_is_not_renamed(state_db):
    adapter = _adapter()
    _add_session(state_db, title="deploy the thing", title_source="llm")
    adapter.rename_thread = AsyncMock()
    with patch.object(fluxer_adapter, "_TITLE_RENAME_WINDOW_SECONDS", 0.05), patch.object(
        fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01
    ):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_one_lookup_runs_per_thread(state_db):
    adapter = _adapter()
    adapter._rename_thread_when_titled = AsyncMock()
    adapter._schedule_title_rename("chan-1", "th-1", "placeholder")
    adapter._schedule_title_rename("chan-1", "th-1", "placeholder")

    assert len(adapter._title_tasks) == 1
    await asyncio.gather(*list(adapter._title_tasks.values()))
    adapter._rename_thread_when_titled.assert_awaited_once()


# ── /title ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_title_command_renames_the_thread(state_db):
    adapter = _adapter()
    _add_session(state_db, session_key="agent:main:fluxer:channel:chan-1:user-1")
    adapter._record_auto_thread("chan-1", "th-1", "deploy the thing")
    adapter.rename_thread = AsyncMock(return_value=True)

    await adapter._rename_thread_for_session("agent:main:fluxer:channel:chan-1:user-1", "Renamed by hand")

    adapter.rename_thread.assert_awaited_once_with(
        "th-1", "Renamed by hand", only_if_current_name="deploy the thing"
    )


@pytest.mark.asyncio
async def test_the_title_command_ignores_a_chat_with_no_thread(state_db):
    adapter = _adapter()
    _add_session(state_db, session_key="agent:main:fluxer:channel:chan-1:user-1")
    adapter.rename_thread = AsyncMock()

    await adapter._rename_thread_for_session("agent:main:fluxer:channel:chan-1:user-1", "Renamed by hand")

    adapter.rename_thread.assert_not_awaited()


def test_the_pre_command_hook_only_reacts_to_a_fluxer_title():
    adapter = _adapter()
    calls: list = []
    adapter.rename_thread_from_session_title = lambda key, title: calls.append((key, title))

    fluxer_adapter._LIVE_ADAPTERS.add(adapter)
    try:
        fluxer_adapter.on_pre_command(
            surface="gateway", command="title", args_raw="Renamed by hand",
            session_key="agent:main:fluxer:channel:chan-1:user-1", platform="fluxer",
        )
        fluxer_adapter.on_pre_command(
            surface="gateway", command="title", args_raw="Renamed by hand",
            session_key="agent:main:fluxer:channel:chan-1:user-1", platform="discord",
        )
        fluxer_adapter.on_pre_command(
            surface="gateway", command="new", args_raw="",
            session_key="agent:main:fluxer:channel:chan-1:user-1", platform="fluxer",
        )
    finally:
        fluxer_adapter._LIVE_ADAPTERS.discard(adapter)

    assert calls == [("agent:main:fluxer:channel:chan-1:user-1", "Renamed by hand")]


def test_the_pre_command_hook_hands_the_rename_to_the_gateway_loop(monkeypatch, state_db):
    adapter = _adapter()
    scheduled: list = []

    class Loop:
        def is_closed(self):
            return False

    def fake_run_coroutine_threadsafe(coro, loop):
        scheduled.append(coro)
        coro.close()

    monkeypatch.setattr(fluxer_adapter.asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    adapter._loop = Loop()

    adapter.rename_thread_from_session_title("agent:main:fluxer:channel:chan-1:user-1", "Renamed by hand")

    assert len(scheduled) == 1


def test_the_pre_command_hook_is_a_no_op_without_a_loop(state_db):
    adapter = _adapter()

    adapter.rename_thread_from_session_title("agent:main:fluxer:channel:chan-1:user-1", "Renamed by hand")


def test_register_wires_the_pre_command_hook():
    registered: list = []

    class Context:
        def register_hook(self, name, callback):
            registered.append((name, callback))

        def register_platform(self, **kwargs):
            self.platform_kwargs = kwargs

    ctx = Context()
    fluxer_adapter.register(ctx)

    assert registered == [("pre_command", fluxer_adapter.on_pre_command)]


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
