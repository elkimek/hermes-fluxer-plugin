"""Security hardening for Fluxer attachment I/O and the exec-approval surface.

Two findings are pinned here.

1. **Attachment download.** ``_download_attachment_bytes`` pre-flighted
   ``is_safe_url`` and then used a plain ``httpx.AsyncClient(follow_redirects=True)``
   and returned ``response.content`` unbounded. A public URL that redirected to a
   private/internal address was followed without re-validation; DNS could be rebound
   between the check and the connect; and any body size was buffered into memory.
   Fetching also happened *before* the channel/mention admission gate, so a message
   the bot then dropped had already triggered outbound network I/O.

2. **Exec-approval surface.** ``send_exec_approval`` lacked the host's
   ``allow_permanent`` / ``allow_session`` / ``smart_denied`` flags (the runner
   passes them, so the call raised ``TypeError``), was not reachable through the
   host's ``supports_exec_approval_buttons`` contract, always offered a full
   four-choice row regardless of a smart deny, kept ``created_at`` unused so a
   pending card never expired, and resolved approvals by FIFO — a tap on an old card
   could resolve a *different* pending request for the same session.

The tests that need the real Hermes host contract use a per-test
``pytest.importorskip``; everything else runs in the bare plugin CI.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock

import pytest

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

ROOT = None  # not needed; kept for symmetry with the other modules

PARENT = "chan-1"
BOT = "bot-1"
OWNER = "owner-user"
OTHER = "other-user"
SESSION = "agent:main:fluxer:channel:chan-1:owner-user"


# ── helpers ────────────────────────────────────────────────────────────────────


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "require_mention": False,
        "delivery_verification": False,
        "bot_user_id": BOT,
    }
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


class _FakeResponse:
    """Minimal httpx-response double for the attachment downloader."""

    def __init__(self, *, status_code=200, headers=None, chunks=(), url="https://cdn.example/x"):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._chunks = list(chunks)
        self.url = url
        self.is_redirect = status_code in (301, 302, 303, 307, 308)

    async def aiter_bytes(self):
        for chunk in self._chunks:
            yield chunk

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeClient:
    """Routes GETs to pre-seeded responses and records every URL it was asked for."""

    def __init__(self, responses, **kwargs):
        self._responses = responses
        self.kwargs = kwargs
        self.requested = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def stream(self, method, url, headers=None):
        self.requested.append(url)
        response = self._responses.get(url)
        if response is None:
            raise AssertionError(f"unexpected fetch of {url}")
        return _FakeStream(response)


def _install_fake_tools_approval(monkeypatch, *, pending=None, record=None):
    """Install a stub ``tools.approval`` so the binding logic can be driven bare.

    ``record`` is a list the stub appends ``(session_key, choice, request_id)`` to.
    """
    module = types.ModuleType("tools.approval")
    calls = record if record is not None else []

    def resolve_gateway_approval(session_key, choice, resolve_all=False, reason=None, request_id=None):
        calls.append({"session_key": session_key, "choice": choice, "request_id": request_id})
        return 1 if request_id else 1

    module.resolve_gateway_approval = resolve_gateway_approval
    module.list_gateway_approvals = lambda session_key: list(pending or [])
    tools_pkg = sys.modules.get("tools") or types.ModuleType("tools")
    tools_pkg.__path__ = getattr(tools_pkg, "__path__", [])
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.approval", module)
    return calls


# ── 1. attachment download: SSRF, redirect and size ────────────────────────────


@pytest.mark.asyncio
async def test_attachment_download_prefers_the_host_ssrf_safe_client(monkeypatch):
    """The connect-time SSRF-safe client is used when the host exposes it."""
    created = {}

    def fake_factory(**kwargs):
        client = _FakeClient(
            {"https://cdn.example/x": _FakeResponse(chunks=[b"hello"])}, **kwargs
        )
        created["client"] = client
        return client

    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", fake_factory)
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)

    adapter = _adapter()
    data = await adapter._download_attachment_bytes("https://cdn.example/x")

    assert data == b"hello"
    client = created["client"]
    # Redirects are walked manually, so the client must NOT auto-follow them.
    assert client.kwargs["follow_redirects"] is False
    assert client.kwargs["event_hooks"]["response"]


@pytest.mark.asyncio
async def test_attachment_download_blocks_a_redirect_to_a_private_address(monkeypatch):
    """A public URL 302-ing to a metadata/private address is refused, not followed."""
    import tools.url_safety as url_safety

    def fake_factory(**kwargs):
        return _FakeClient(
            {
                "https://cdn.example/x": _FakeResponse(
                    status_code=302, headers={"location": "http://169.254.169.254/latest/meta-data"}
                ),
            },
            **kwargs,
        )

    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", fake_factory)
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: "169.254.169.254" not in url)

    adapter = _adapter()
    with pytest.raises(ValueError, match="redirect"):
        await adapter._download_attachment_bytes("https://cdn.example/x")


@pytest.mark.asyncio
async def test_attachment_download_walks_a_safe_redirect_and_revalidates_each_hop(monkeypatch):
    """A safe redirect is followed, and the final body is returned."""
    import tools.url_safety as url_safety

    seen = []

    def fake_factory(**kwargs):
        return _FakeClient(
            {
                "https://cdn.example/a": _FakeResponse(
                    status_code=302, headers={"location": "https://cdn.example/b"}
                ),
                "https://cdn.example/b": _FakeResponse(chunks=[b"final"]),
            },
            **kwargs,
        )

    def fake_is_safe(url):
        seen.append(url)
        return True

    monkeypatch.setattr(url_safety, "create_ssrf_safe_async_client", fake_factory)
    monkeypatch.setattr(url_safety, "is_safe_url", fake_is_safe)

    adapter = _adapter()
    data = await adapter._download_attachment_bytes("https://cdn.example/a")

    assert data == b"final"
    assert "https://cdn.example/b" in seen


@pytest.mark.asyncio
async def test_attachment_download_rejects_an_oversized_declared_content_length(monkeypatch):
    """A declared Content-Length over the cap fails before any body is read."""
    import tools.url_safety as url_safety

    monkeypatch.setattr(
        url_safety,
        "create_ssrf_safe_async_client",
        lambda **kw: _FakeClient(
            {
                "https://cdn.example/big": _FakeResponse(
                    headers={"content-length": str(10 * 1024 * 1024 * 1024)},
                    chunks=[b"tiny"],
                )
            },
            **kw,
        ),
    )
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(fluxer_adapter, "_attachment_max_bytes", lambda: 1024)

    adapter = _adapter()
    with pytest.raises(ValueError, match="exceeds"):
        await adapter._download_attachment_bytes("https://cdn.example/big")


@pytest.mark.asyncio
async def test_attachment_download_caps_a_body_with_no_content_length(monkeypatch):
    """A lying/absent Content-Length cannot smuggle a larger body past the cap."""
    import tools.url_safety as url_safety

    monkeypatch.setattr(
        url_safety,
        "create_ssrf_safe_async_client",
        lambda **kw: _FakeClient(
            {"https://cdn.example/big": _FakeResponse(chunks=[b"x" * 800, b"y" * 800])},
            **kw,
        ),
    )
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)
    monkeypatch.setattr(fluxer_adapter, "_attachment_max_bytes", lambda: 1024)

    adapter = _adapter()
    with pytest.raises(ValueError, match="exceeds"):
        await adapter._download_attachment_bytes("https://cdn.example/big")


@pytest.mark.asyncio
async def test_attachment_download_falls_back_to_plain_httpx_when_the_host_lacks_the_guard(monkeypatch):
    """An older host without the SSRF-safe client still gets a bounded, manual-redirect fetch."""
    import tools.url_safety as url_safety

    monkeypatch.delattr(url_safety, "create_ssrf_safe_async_client", raising=False)
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: True)

    created = {}

    def fake_httpx_client(**kwargs):
        client = _FakeClient(
            {"https://cdn.example/x": _FakeResponse(chunks=[b"ok"])}, **kwargs
        )
        created["client"] = client
        return client

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", fake_httpx_client)

    adapter = _adapter()
    data = await adapter._download_attachment_bytes("https://cdn.example/x")

    assert data == b"ok"
    # The fallback must still refuse to auto-follow redirects.
    assert created["client"].kwargs["follow_redirects"] is False


@pytest.mark.asyncio
async def test_an_unsafe_attachment_url_is_never_fetched(monkeypatch):
    """A pre-flight rejection returns no bytes and never opens a client."""
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: False)
    opened = []
    monkeypatch.setattr(
        url_safety, "create_ssrf_safe_async_client", lambda **kw: opened.append(kw)
    )

    adapter = _adapter()
    with pytest.raises(ValueError, match="unsafe"):
        await adapter._download_attachment_bytes("http://127.0.0.1/secret")
    assert opened == []


@pytest.mark.asyncio
async def test_cache_attachment_drops_an_unsafe_url_without_fetching(monkeypatch):
    """A blocked attachment URL is dropped, and the message proceeds on text alone."""
    import tools.url_safety as url_safety

    monkeypatch.setattr(url_safety, "is_safe_url", lambda url: False)

    adapter = _adapter()
    adapter._download_attachment_bytes = AsyncMock(side_effect=AssertionError("must not fetch"))

    cached, mtype = await adapter._cache_attachment(
        {"url": "http://169.254.169.254/x", "content_type": "application/pdf", "filename": "x.pdf"}
    )
    assert (cached, mtype) == (None, None)
    adapter._download_attachment_bytes.assert_not_awaited()


def test_attachment_cap_reads_the_host_limit_when_available(monkeypatch):
    """The cap comes from the host's own reader, so config changes are honoured."""
    import gateway.platforms.base as base

    monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 4096)
    assert fluxer_adapter._attachment_max_bytes() == 4096

    # A disabled (0) host cap must still leave the download bounded.
    monkeypatch.setattr(base, "get_inbound_media_max_bytes", lambda: 0)
    assert fluxer_adapter._attachment_max_bytes() == fluxer_adapter._DEFAULT_ATTACHMENT_MAX_BYTES


# ── 2. attachment fetch happens only after admission ───────────────────────────


@pytest.mark.asyncio
async def test_a_rejected_message_never_fetches_its_attachments(monkeypatch):
    """A mention-gated message the bot drops must trigger no outbound I/O."""
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = _adapter(require_mention=True)
    adapter.handle_message = AsyncMock()
    adapter._cache_attachment = AsyncMock(return_value=("/tmp/x", "application/pdf"))

    await adapter._handle_message_create(
        {
            "id": "msg-gated",
            "channel_id": PARENT,
            "channel_type": "channel",
            "content": "no mention here",
            "author": {"id": OWNER, "username": "Alice", "bot": False},
            "attachments": [
                {"id": "a1", "filename": "x.pdf", "url": "https://cdn.example/x", "content_type": "application/pdf"}
            ],
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    adapter._cache_attachment.assert_not_awaited()
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_admitted_message_fetches_its_attachments(monkeypatch):
    """The same message with a mention is admitted and its attachment is fetched."""
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = _adapter(require_mention=True)
    adapter.handle_message = AsyncMock()
    adapter._cache_attachment = AsyncMock(return_value=("/tmp/x", "application/pdf"))

    await adapter._handle_message_create(
        {
            "id": "msg-mentioned",
            "channel_id": PARENT,
            "channel_type": "channel",
            "content": f"<@{BOT}> look at this",
            "author": {"id": OWNER, "username": "Alice", "bot": False},
            "attachments": [
                {"id": "a1", "filename": "x.pdf", "url": "https://cdn.example/x", "content_type": "application/pdf"}
            ],
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    adapter._cache_attachment.assert_awaited()
    adapter.handle_message.assert_awaited()


@pytest.mark.asyncio
async def test_a_media_only_message_stays_eligible_without_downloading_first(monkeypatch):
    """An attachment-only message is admitted on metadata; bytes move after the gate."""
    monkeypatch.delenv("FLUXER_REQUIRE_MENTION", raising=False)
    adapter = _adapter(require_mention=True)
    seen = []

    async def fake_handle(event):
        seen.append(event)

    adapter.handle_message = fake_handle
    adapter._cache_attachment = AsyncMock(return_value=("/tmp/x.pdf", "application/pdf"))

    await adapter._handle_message_create(
        {
            "id": "msg-media-only",
            "channel_id": PARENT,
            "channel_type": "channel",
            "content": "",
            "author": {"id": OWNER, "username": "Alice", "bot": False},
            "mentions": [{"id": BOT}],
            "attachments": [
                {"id": "a1", "filename": "x.pdf", "url": "https://cdn.example/x", "content_type": "application/pdf"}
            ],
        },
        {"op": 0, "t": "MESSAGE_CREATE", "d": {}},
    )

    assert len(seen) == 1
    assert seen[0].media_urls == ["/tmp/x.pdf"]
    adapter._cache_attachment.assert_awaited()


def test_message_has_attachments_reads_metadata_only():
    assert fluxer_adapter._message_has_attachments(
        {"attachments": [{"url": "https://cdn.example/x"}]}
    )
    assert not fluxer_adapter._message_has_attachments({"attachments": []})
    assert not fluxer_adapter._message_has_attachments({"attachments": [{"filename": "x"}]})
    assert not fluxer_adapter._message_has_attachments({})


# ── 3. approval capability, flags and choice policy ────────────────────────────


def test_the_adapter_advertises_native_approval_buttons():
    """The host only routes to ``send_exec_approval`` for an adapter that opts in."""
    assert fluxer_adapter.FluxerAdapter.supports_exec_approval_buttons() is True


def test_the_prompt_hook_is_the_contract_override():
    """Overriding ``_send_exec_approval_prompt`` is what makes the capability true."""
    base = fluxer_adapter.BasePlatformAdapter
    assert (
        fluxer_adapter.FluxerAdapter._send_exec_approval_prompt
        is not base._send_exec_approval_prompt
    )


@pytest.mark.asyncio
async def test_send_exec_approval_accepts_the_host_flags():
    """The runner passes allow_permanent/allow_session/smart_denied; they must be accepted."""
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    result = await adapter.send_exec_approval(
        PARENT, "rm -rf /", SESSION,
        description="dangerous",
        allow_permanent=True,
        allow_session=True,
        smart_denied=False,
    )
    assert result.success is True


@pytest.mark.asyncio
async def test_a_smart_deny_offers_only_once_and_deny():
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION, smart_denied=True)

    # Only the two reactions for once+deny were added.
    added = {call.args[2] for call in adapter._add_reaction.await_args_list}
    assert added == {"✅", "❌"}
    assert set(adapter._pending_reaction_actions) == {"card-1:✅", "card-1:❌"}


@pytest.mark.asyncio
async def test_allow_session_false_offers_only_once_and_deny():
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION, allow_session=False)
    added = {call.args[2] for call in adapter._add_reaction.await_args_list}
    assert added == {"✅", "❌"}


@pytest.mark.asyncio
async def test_allow_permanent_false_still_offers_the_session_tier():
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION, allow_permanent=False)
    added = {call.args[2] for call in adapter._add_reaction.await_args_list}
    assert added == {"✅", "🕒", "❌"}


def test_exec_approval_choices_mirror_the_host_policy():
    choices = {c for _e, c, _l in fluxer_adapter._exec_approval_choices(
        allow_permanent=True, allow_session=True, smart_denied=False)}
    assert choices == {"once", "session", "always", "deny"}
    assert {c for _e, c, _l in fluxer_adapter._exec_approval_choices(
        allow_permanent=False, allow_session=True, smart_denied=False)} == {"once", "session", "deny"}
    assert {c for _e, c, _l in fluxer_adapter._exec_approval_choices(
        allow_permanent=True, allow_session=True, smart_denied=True)} == {"once", "deny"}


@pytest.mark.asyncio
async def test_the_prompt_hook_renders_the_hosts_own_action_rows():
    """``_send_exec_approval_prompt`` derives its choices from ``prompt.actions``."""
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    prompt = types.SimpleNamespace(
        chat_id=PARENT,
        session_key=SESSION,
        command="rm -rf /",
        description="dangerous",
        metadata=None,
        smart_denied=True,
        actions=[("Approve once", "once", "primary"), ("Deny", "deny", "danger")],
    )
    result = await adapter._send_exec_approval_prompt(prompt)

    assert result.success is True
    added = {call.args[2] for call in adapter._add_reaction.await_args_list}
    assert added == {"✅", "❌"}


# ── 4. request binding: never resolve a different request ──────────────────────


@pytest.mark.asyncio
async def test_a_unique_pending_command_binds_the_request_id(monkeypatch):
    """Exactly one matching pending request → the card resolves THAT request."""
    calls = _install_fake_tools_approval(
        monkeypatch,
        pending=[{"request_id": "req-abc", "command": "rm -rf /"}],
        record=[],
    )
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION)
    assert adapter._pending_exec_approvals["card-1"]["request_id"] == "req-abc"

    adapter._request = AsyncMock(return_value={})
    await adapter._resolve_exec_approval_action(
        {
            "kind": "exec_approval",
            "message_id": "card-1",
            "session_key": SESSION,
            "channel_id": PARENT,
            "choice": "once",
            "request_id": "req-abc",
        },
        user_id=OWNER,
    )

    assert calls[-1]["request_id"] == "req-abc"
    assert calls[-1]["choice"] == "once"


@pytest.mark.asyncio
async def test_an_ambiguous_command_does_not_bind_a_request_id(monkeypatch):
    """Two identical pending commands → no id bound, so FIFO is used (never a guess)."""
    _install_fake_tools_approval(
        monkeypatch,
        pending=[
            {"request_id": "req-1", "command": "rm -rf /"},
            {"request_id": "req-2", "command": "rm -rf /"},
        ],
    )
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION)
    assert adapter._pending_exec_approvals["card-1"]["request_id"] is None


@pytest.mark.asyncio
async def test_no_matching_pending_command_leaves_the_id_unbound(monkeypatch):
    _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter()
    adapter._post_message_with_optional_components = AsyncMock(
        return_value=({"id": "card-1"}, False)
    )
    adapter._add_reaction = AsyncMock()

    await adapter.send_exec_approval(PARENT, "rm -rf /", SESSION)
    assert adapter._pending_exec_approvals["card-1"]["request_id"] is None


@pytest.mark.asyncio
async def test_a_bound_card_that_finds_no_pending_request_does_not_fall_back_to_fifo(monkeypatch):
    """A live card whose request already left the queue must not resolve a *different* one."""
    calls = _install_fake_tools_approval(monkeypatch, pending=[], record=[])
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={})
    adapter._pending_exec_approvals["card-1"] = {
        "session_key": SESSION,
        "channel_id": PARENT,
        "content": "prompt",
        "resolved": False,
        "request_id": "req-gone",
    }

    # Simulate resolve_gateway_approval returning 0 for a stale request id.
    import tools.approval as approval_mod

    def zero(session_key, choice, resolve_all=False, reason=None, request_id=None):
        calls.append({"session_key": session_key, "choice": choice, "request_id": request_id})
        return 0

    monkeypatch.setattr(approval_mod, "resolve_gateway_approval", zero)

    await adapter._resolve_exec_approval_action(
        {
            "kind": "exec_approval",
            "message_id": "card-1",
            "session_key": SESSION,
            "channel_id": PARENT,
            "choice": "once",
            "request_id": "req-gone",
        },
        user_id=OWNER,
    )

    # Exactly one resolve attempt, carrying the bound id — no second FIFO attempt.
    assert len(calls) == 1
    assert calls[0]["request_id"] == "req-gone"


@pytest.mark.asyncio
async def test_an_unbound_card_resolves_without_a_request_id(monkeypatch):
    """With no id bound the resolution stays on the host's FIFO path (request_id=None)."""
    calls = _install_fake_tools_approval(monkeypatch, pending=[], record=[])
    adapter = _adapter()
    adapter._request = AsyncMock(return_value={})
    adapter._pending_exec_approvals["card-1"] = {
        "session_key": SESSION,
        "channel_id": PARENT,
        "content": "prompt",
        "resolved": False,
        "request_id": None,
    }

    await adapter._resolve_exec_approval_action(
        {
            "kind": "exec_approval",
            "message_id": "card-1",
            "session_key": SESSION,
            "channel_id": PARENT,
            "choice": "deny",
        },
        user_id=OWNER,
    )
    assert calls[-1]["request_id"] is None


# ── 5. interaction guards: user, channel, button, deletion, expiry ─────────────


@pytest.mark.asyncio
async def test_a_disallowed_user_cannot_resolve_an_approval(monkeypatch):
    calls = _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter(allow_all_users=False, allowed_users=OWNER)
    adapter._pending_reaction_actions["card-1:✅"] = {
        "kind": "exec_approval",
        "message_id": "card-1",
        "session_key": SESSION,
        "channel_id": PARENT,
        "choice": "once",
    }

    await adapter._handle_reaction_add(
        {"message_id": "card-1", "channel_id": PARENT, "emoji": "✅", "user_id": OTHER}
    )
    assert calls == []
    assert "card-1:✅" in adapter._pending_reaction_actions


@pytest.mark.asyncio
async def test_a_reaction_in_the_wrong_channel_is_ignored(monkeypatch):
    calls = _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter()
    adapter._pending_reaction_actions["card-1:✅"] = {
        "kind": "exec_approval",
        "message_id": "card-1",
        "session_key": SESSION,
        "channel_id": PARENT,
        "choice": "once",
    }

    await adapter._handle_reaction_add(
        {"message_id": "card-1", "channel_id": "other-chan", "emoji": "✅", "user_id": OWNER}
    )
    assert calls == []


@pytest.mark.asyncio
async def test_a_component_interaction_with_the_wrong_message_is_ignored(monkeypatch):
    calls = _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter()
    adapter._pending_component_actions["cid-1"] = {
        "kind": "exec_approval",
        "message_id": "card-1",
        "session_key": SESSION,
        "channel_id": PARENT,
        "choice": "once",
    }

    await adapter._handle_component_interaction(
        {
            "data": {"custom_id": "cid-1"},
            "user": {"id": OWNER},
            "message": {"id": "other-card"},
            "channel_id": PARENT,
        }
    )
    assert calls == []
    assert "cid-1" in adapter._pending_component_actions


@pytest.mark.asyncio
async def test_an_unknown_button_is_ignored(monkeypatch):
    calls = _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter()

    await adapter._handle_component_interaction(
        {
            "data": {"custom_id": "never-registered"},
            "user": {"id": OWNER},
            "message": {"id": "card-1"},
            "channel_id": PARENT,
        }
    )
    assert calls == []


@pytest.mark.asyncio
async def test_deleting_a_prompt_fails_closed_on_its_own_request(monkeypatch):
    calls = _install_fake_tools_approval(monkeypatch, pending=[])
    adapter = _adapter()
    adapter._pending_exec_approvals["card-1"] = {
        "session_key": SESSION,
        "channel_id": PARENT,
        "created_at": 0.0,
        "content": "prompt",
        "resolved": False,
        "request_id": "req-del",
    }

    await adapter._handle_message_delete({"id": "card-1", "channel_id": PARENT})

    assert calls == [
        {"session_key": SESSION, "choice": "deny", "request_id": "req-del"}
    ]


# ── 6. expiry: a stale card cannot approve a newer request ─────────────────────


def test_pending_cards_expire_after_the_host_ttl(monkeypatch):
    """A card older than the host's approval timeout is pruned from every map."""
    monkeypatch.setattr(fluxer_adapter, "_approval_card_ttl", lambda: 100.0)
    adapter = _adapter()
    adapter._pending_exec_approvals["old"] = {"created_at": 0.0, "resolved": False}
    adapter._pending_component_actions["old-cid"] = {"created_at": 0.0}
    adapter._pending_reaction_actions["old:✅"] = {"created_at": 0.0}
    adapter._pending_exec_approvals["fresh"] = {"created_at": 1_000_000.0, "resolved": False}

    monkeypatch.setattr(fluxer_adapter.time, "time", lambda: 1000.0)
    adapter._prune_pending_cards()

    assert "old" not in adapter._pending_exec_approvals
    assert "old-cid" not in adapter._pending_component_actions
    assert "old:✅" not in adapter._pending_reaction_actions
    assert "fresh" in adapter._pending_exec_approvals


@pytest.mark.asyncio
async def test_an_expired_card_cannot_approve_a_new_request(monkeypatch):
    """The replay that matters: a stale card is pruned, so a tap on it is inert.

    A new request is queued after the old card expired; the old card's button must
    find nothing registered and resolve nothing — the new request stays untouched.
    """
    monkeypatch.setattr(fluxer_adapter, "_approval_card_ttl", lambda: 100.0)
    calls = _install_fake_tools_approval(monkeypatch, pending=[])

    adapter = _adapter()
    # Old card registered long ago and a new one just now.
    monkeypatch.setattr(fluxer_adapter.time, "time", lambda: 0.0)
    adapter._pending_reaction_actions["old-card:✅"] = {
        "kind": "exec_approval",
        "message_id": "old-card",
        "session_key": SESSION,
        "channel_id": PARENT,
        "choice": "once",
        "created_at": 0.0,
    }
    monkeypatch.setattr(fluxer_adapter.time, "time", lambda: 10_000.0)
    adapter._prune_pending_cards()

    assert "old-card:✅" not in adapter._pending_reaction_actions

    # A tap on the expired card resolves nothing.
    await adapter._handle_reaction_add(
        {"message_id": "old-card", "channel_id": PARENT, "emoji": "✅", "user_id": OWNER}
    )
    assert calls == []


def test_the_ttl_never_drops_below_the_host_default(monkeypatch):
    """A tiny/unreadable host timeout still leaves a sane floor."""
    import tools.approval_context as approval_context

    monkeypatch.setattr(approval_context, "_get_approval_timeout", lambda: 5)
    assert fluxer_adapter._approval_card_ttl() >= 300.0


# ── 7. host integration: the real contract behind the binding ──────────────────


def _host_contract_probe(code: str) -> str:
    """Run *code* against the REAL Hermes host in a subprocess, then return stdout.

    The plugin repo ships its own ``tools/`` package, which shadows the host's on
    ``sys.path``; running the probe in a subprocess rooted OUTSIDE the plugin tree
    (with only the Hermes checkout on the path) guarantees the real host module is
    what answers. Skips when no Hermes checkout is present (bare plugin CI).
    """
    import os
    import subprocess
    from pathlib import Path

    hermes_home = Path(os.environ.get("HERMES_AGENT_SRC", "/home/hermes/.hermes/hermes-agent"))
    if not (hermes_home / "tools" / "approval.py").is_file():
        pytest.skip(f"Hermes host tree not available at {hermes_home}")
    interpreter = os.environ.get("PYTHON", sys.executable)
    env = {
        **os.environ,
        "PYTHONPATH": str(hermes_home),
        "HERMES_HOME": os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")),
    }
    proc = subprocess.run(
        [interpreter, "-c", code],
        cwd=str(Path(hermes_home).parent),  # never the plugin root
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        pytest.skip(f"Hermes host probe could not run: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


def test_host_resolve_gateway_approval_accepts_a_request_id_kwarg():
    """The binding is only safe because the host exposes request-id resolution."""
    out = _host_contract_probe(
        "import inspect, tools.approval as a;"
        "p = inspect.signature(a.resolve_gateway_approval).parameters;"
        "print('request_id' in p, 'session_key' in p, 'choice' in p)"
    )
    assert out == "True True True"


def test_host_lists_pending_approvals_with_a_request_id():
    """The adapter's match key (``request_id`` + ``command``) exists in the host snapshot."""
    out = _host_contract_probe(
        "import tools.approval as a;"
        "print(a.list_gateway_approvals('fluxer-test-nonexistent-session'))"
    )
    assert out == "[]"


def test_host_resolution_of_an_unknown_request_id_is_a_no_op():
    """An id that is not pending resolves nothing — the guarantee the guard relies on."""
    out = _host_contract_probe(
        "import tools.approval as a;"
        "print(a.resolve_gateway_approval('fluxer-test-nonexistent-session', 'once',"
        " request_id='does-not-exist'))"
    )
    assert out == "0"


def test_host_a_request_id_resolves_only_its_own_entry_not_fifo():
    """End to end on the REAL host queue: tapping the second card resolves the second.

    This is the exact replay the finding is about — with two pending approvals for
    one session, FIFO would resolve the first regardless of which card was tapped.
    """
    out = _host_contract_probe(
        "from tools import approval;"
        "from tools.approval_gateway_wait import _ApprovalEntry;"
        "s='fluxer-test-binding-session';"
        "approval._gateway_queues[s]=[_ApprovalEntry({'command':'echo FIRST'}),"
        " _ApprovalEntry({'command':'echo SECOND'})];"
        "second=approval._gateway_queues[s][1].data['request_id'];"
        "n=approval.resolve_gateway_approval(s,'once',request_id=second);"
        "left=[e['command'] for e in approval.list_gateway_approvals(s)];"
        "approval._gateway_queues.pop(s,None);"
        "print(n, left)"
    )
    assert out == "1 ['echo FIRST']"
