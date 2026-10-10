"""Auto-thread naming: placeholder from the trigger, session title when there is one.

Mirrors the Discord adapter's two-phase thread naming, but self-contained in the
plugin: the gateway's session title is read back through Hermes' own read-only
``SessionDB`` rather than pushed by the host, so no Hermes core change is involved.

``FakeSessionStore`` below deliberately stands in for that store. It reimplements
the semantics the plugin depends on — including core's refusal to guess between
several live participants — against a temp database, so these tests run in the
standalone plugin repository where the Hermes tree is not importable.
``test_the_real_store_exposes_what_the_plugin_calls`` covers the same seam in the
other direction and skips when the runtime is absent.
"""

from __future__ import annotations

import asyncio
import inspect
import sqlite3
import tempfile
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
    thread_id TEXT,
    user_id TEXT,
    title TEXT,
    title_source TEXT,
    started_at REAL,
    ended_at REAL,
    last_activity_at REAL
)
"""


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {"bot_token": "app.secret", "allow_all_users": True, "require_mention": False}
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


class FakeSessionStore:
    """Stand-in for core's ``SessionDB(read_only=True)``, over a temp store.

    Only the calls the plugin makes, with their documented semantics: live sessions
    only, and a lookup with several distinct participants returns ``None`` rather
    than handing back another participant's session.
    """

    def __init__(self, db: Path, *, explode: bool = False) -> None:
        self.db = db
        self.explode = explode
        self.closed = False

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        if self.explode:
            raise sqlite3.DatabaseError("store unreadable")
        conn = sqlite3.connect(f"file:{self.db}?mode=ro", uri=True)
        try:
            cursor = conn.execute(sql, params)
            names = [col[0] for col in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def find_session_by_origin(self, *, platform, chat_id, thread_id=None, user_id=None):
        rows = self._rows(
            "SELECT id, user_id, started_at FROM sessions WHERE LOWER(source) = LOWER(?) "
            "AND session_key IS NOT NULL AND chat_id = ? AND ended_at IS NULL "
            "ORDER BY started_at DESC",
            (platform, str(chat_id)),
        )
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len({u for u in (str(r.get("user_id") or "").strip() for r in rows) if u}) > 1:
            return None
        return str(rows[0]["id"])

    def get_session_title(self, session_id):
        rows = self._rows("SELECT title FROM sessions WHERE id = ?", (session_id,))
        return rows[0]["title"] if rows else None

    def get_session_title_source(self, session_id):
        rows = self._rows("SELECT title, title_source FROM sessions WHERE id = ?", (session_id,))
        return rows[0]["title_source"] if rows and rows[0]["title"] is not None else None

    def list_sessions_rich(self, *, source=None, session_key=None, limit=20, **_kw):
        rows = self._rows(
            "SELECT * FROM sessions WHERE source = ? AND session_key = ?", (source, session_key)
        )
        return rows[:limit]

    def close(self):
        self.closed = True


@pytest.fixture
def state_db(monkeypatch):
    """A stand-in Hermes session store, reached through HERMES_HOME.

    Also installs the fake store as the opener, so an adapter built inside the test
    reads through it — and, being ``monkeypatch``-scoped, uninstalls it afterwards
    instead of leaking a global into the next test.
    """
    home = tempfile.mkdtemp(prefix="fluxer-title-home-")
    db = Path(home) / "state.db"
    conn = sqlite3.connect(db)
    conn.executescript(SESSIONS_DDL)
    conn.commit()
    conn.close()
    monkeypatch.setenv("HERMES_HOME", home)
    store = FakeSessionStore(db)
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda: store)
    return db


@pytest.fixture
def store(state_db):
    """The fake store the adapter under test will open (the same instance)."""
    return fluxer_adapter._open_session_store()


def _add_session(db: Path, **row) -> None:
    values = {
        "id": "sess-1", "session_key": "agent:main:fluxer:channel:chan-1:user-1",
        "source": "fluxer", "chat_id": "chan-1", "chat_type": "channel",
        "thread_id": None, "user_id": "user-1", "title": None, "title_source": None,
        "started_at": 1.0, "ended_at": None, "last_activity_at": 1.0,
    }
    values.update(row)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO sessions (id, session_key, source, chat_id, chat_type, thread_id, user_id, "
        "title, title_source, started_at, ended_at, last_activity_at) VALUES (:id, :session_key, "
        ":source, :chat_id, :chat_type, :thread_id, :user_id, :title, :title_source, :started_at, "
        ":ended_at, :last_activity_at)",
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


# ── the store seam ──────────────────────────────────────────────────────────────


def test_no_raw_sqlite_read_is_shipped():
    """The plugin must not hand-roll a read-only URI: a raw f-string truncates at a
    '?' or '#' in the home path and silently opens the wrong database. Core's store
    owns that construction, so adapter.py must not reach for sqlite3 at all."""
    source = Path(fluxer_adapter.__file__).read_text()

    assert "sqlite3.connect" not in source
    assert "?mode=ro" not in source


def test_the_store_is_opened_lazily_once(monkeypatch):
    opened: list = []

    def fake_open():
        opened.append(1)
        return FakeSessionStore(Path("/nonexistent"))

    monkeypatch.setattr(fluxer_adapter, "_open_session_store", fake_open)
    adapter = _adapter()

    assert adapter._session_store() is not None
    assert adapter._session_store() is not None
    assert len(opened) == 1


def test_an_absent_store_is_not_retried(monkeypatch):
    opened: list = []

    def fake_open():
        opened.append(1)
        return None

    monkeypatch.setattr(fluxer_adapter, "_open_session_store", fake_open)
    adapter = _adapter()

    assert adapter._session_store() is None
    assert adapter._session_store() is None
    assert len(opened) == 1


def test_closing_the_store_releases_it_and_allows_a_reopen(monkeypatch):
    opened: list = []

    def fake_open():
        store = FakeSessionStore(Path("/nonexistent"))
        opened.append(store)
        return store

    monkeypatch.setattr(fluxer_adapter, "_open_session_store", fake_open)
    adapter = _adapter()
    first = adapter._session_store()

    adapter._close_session_store()

    assert first.closed is True
    assert adapter._session_db is None
    assert adapter._session_store() is not first
    assert len(opened) == 2


@pytest.fixture
def real_read_only_store(tmp_path):
    """A genuine ``SessionDB`` (core's, not the fake) over a freshly built store.

    Built writable first so core creates its own schema, then reopened read-only —
    the same handle shape the plugin opens at runtime. Skips where the Hermes
    runtime is not importable (the standalone plugin CI), which is why the fake
    above mirrors the semantics instead of us having no coverage at all there.
    """
    hermes_state = pytest.importorskip("hermes_state", reason="Hermes runtime not importable")

    home = tmp_path / "home"
    home.mkdir()
    writer = hermes_state.SessionDB(db_path=home / "state.db")
    writer.create_session(
        session_id="s1", source="fluxer", chat_id="chan-1",
        session_key="agent:main:fluxer:channel:chan-1:user-1", user_id="user-1",
    )
    writer.set_auto_title("s1", "Deploy the thing", source="llm")
    writer.close()

    store = hermes_state.SessionDB(db_path=home / "state.db", read_only=True)
    try:
        yield store
    finally:
        store.close()


def test_the_real_store_exposes_what_the_plugin_calls(real_read_only_store):
    """The seam in the other direction: core's real ``SessionDB`` must carry the
    calls the adapter makes, with the parameters it passes."""
    store = real_read_only_store

    assert store.read_only is True
    for name, expected in (
        ("find_session_by_origin", {"platform", "chat_id", "thread_id", "user_id"}),
        ("get_session_title", {"session_id"}),
        ("get_session_title_source", {"session_id"}),
        ("list_sessions_rich", {"source", "session_key", "limit"}),
    ):
        assert hasattr(store, name), f"SessionDB lost {name}"
        params = set(inspect.signature(getattr(store, name)).parameters)
        assert expected <= params, f"{name} no longer accepts {expected - params}"
    assert callable(store.close)


def test_the_plugin_reads_a_real_store(monkeypatch, real_read_only_store):
    """End to end against core's own store: no fake anywhere in the read path."""
    monkeypatch.setattr(fluxer_adapter, "_open_session_store", lambda: real_read_only_store)
    adapter = _adapter()

    assert adapter._session_title_for_chat("chan-1") == "Deploy the thing"
    assert adapter._session_title_for_chat("chan-1", "user-1") == "Deploy the thing"
    assert adapter._session_title_for_chat("no-such-chat") is None
    assert (
        adapter._chat_id_for_session("agent:main:fluxer:channel:chan-1:user-1") == "chan-1"
    )


def test_a_write_through_the_store_is_refused(tmp_path):
    """The handle must be genuinely read-only, not merely treated as such."""
    hermes_state = pytest.importorskip("hermes_state", reason="Hermes runtime not importable")

    home = tmp_path / "home"
    home.mkdir()
    writable = hermes_state.SessionDB(db_path=home / "state.db")
    writable.create_session(session_id="s1", source="fluxer", chat_id="c1", session_key="k1")
    writable.close()

    store = hermes_state.SessionDB(db_path=home / "state.db", read_only=True)
    try:
        with pytest.raises(Exception):
            store.set_session_title("s1", "nope")
    finally:
        store.close()


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


def test_session_title_prefers_the_most_recently_started_session(state_db):
    adapter = _adapter()
    _add_session(state_db, id="old", title="Older", title_source="llm", started_at=1.0)
    _add_session(state_db, id="new", title="Newer", title_source="llm", started_at=9.0)

    assert adapter._session_title_for_chat("chan-1") == "Newer"


def test_session_title_uses_the_triggering_participants_session(state_db):
    """A channel holds one session per participant; the caller names which one."""
    adapter = _adapter()
    _add_session(state_db, id="a", user_id="user-a", title="A's work", title_source="llm")
    _add_session(state_db, id="b", user_id="user-b", title="B's work", title_source="llm")

    assert adapter._session_title_for_chat("chan-1", "user-b") == "B's work"
    assert adapter._session_title_for_chat("chan-1", "user-a") == "A's work"


def test_session_title_is_none_when_several_participants_are_live_and_none_is_named(state_db):
    """Core refuses to guess between participants; the plugin must not either."""
    adapter = _adapter()
    _add_session(state_db, id="a", user_id="user-a", title="A's work", title_source="llm")
    _add_session(state_db, id="b", user_id="user-b", title="B's work", title_source="llm")

    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_ignores_an_ended_session(state_db):
    adapter = _adapter()
    _add_session(state_db, title="Finished", title_source="llm", ended_at=5.0)

    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_is_none_without_a_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter = _adapter()

    assert adapter._session_store() is None
    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_survives_an_unreadable_store(store):
    store.explode = True
    adapter = _adapter()

    assert adapter._session_title_for_chat("chan-1") is None


def test_session_title_survives_a_store_that_raises_unexpectedly(state_db):
    adapter = _adapter()

    def boom(**_kw):
        raise RuntimeError("boom")

    adapter._session_store().find_session_by_origin = boom

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
async def test_thread_creation_looks_up_the_triggering_author(store, state_db):
    adapter = _adapter()
    _add_session(state_db, user_id="user-b", title="B's work", title_source="llm")
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter._inbound_author_by_message["msg-1"] = "user-b"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "B's work"})
    adapter._schedule_title_rename = lambda *args: None
    seen: list = []
    original = store.find_session_by_origin

    def spy(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)

    store.find_session_by_origin = spy

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert seen and seen[0]["user_id"] == "user-b"
    assert adapter.create_thread.await_args.args[:2] == ("chan-1", "B's work")


@pytest.mark.asyncio
async def test_thread_without_a_title_yet_is_watched(state_db):
    adapter = _adapter()
    adapter._inbound_text_by_message["msg-1"] = "deploy the thing"
    adapter.create_thread = AsyncMock(return_value={"id": "th-1", "name": "deploy the thing"})
    watched: list = []
    adapter._schedule_title_rename = lambda *args: watched.append(args)

    await adapter._start_thread_from_message("chan-1", "msg-1")

    assert watched == [("chan-1", "th-1", "deploy the thing", None)]


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

    adapter._session_title_for_chat = lambda chat_id, user_id=None: next(results, "Deploy the thing")
    with patch.object(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing")

    adapter.rename_thread.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_poller_carries_the_triggering_user(state_db):
    adapter = _adapter()
    _add_session(state_db, user_id="user-b", title="B's work", title_source="llm")
    adapter.rename_thread = AsyncMock(return_value=True)

    with patch.object(fluxer_adapter, "_TITLE_RENAME_POLL_INTERVAL", 0.01):
        await adapter._rename_thread_when_titled("chan-1", "th-1", "deploy the thing", "user-b")

    adapter.rename_thread.assert_awaited_once_with(
        "th-1", "B's work", only_if_current_name="deploy the thing"
    )


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
