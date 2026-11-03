"""Agent tool surface: registration and read-tool formatting."""

import pytest
import adapter as fluxer_adapter
import agent_tools


@pytest.fixture(autouse=True)
def _isolate_tool_secrets(monkeypatch):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)


class RecordingCtx:
    def __init__(self):
        self.tools = []

    def register_tool(self, **kwargs):
        self.tools.append(kwargs)

    def register_platform(self, **kwargs):
        pass

    def register_hook(self, *args, **kwargs):
        pass


def test_register_registers_four_tools_in_fluxer_toolset():
    ctx = RecordingCtx()
    count = agent_tools.register(ctx)
    assert count == 4
    assert {t["name"] for t in ctx.tools} == {
        "flux_recall", "flux_search", "flux_members", "flux_audit"}
    assert all(t["toolset"] == "fluxer" for t in ctx.tools)
    assert all(t["requires_env"] == ["FLUXER_BOT_TOKEN"] for t in ctx.tools)


def test_full_register_path_includes_agent_tools(monkeypatch):
    registered = {}
    monkeypatch.setattr(agent_tools, "register", lambda ctx, **kw: registered.update(n=4) or 4)
    ctx = RecordingCtx()
    fluxer_adapter.register(ctx)  # must not raise; platform + tools both land
    assert registered.get("n") == 4


def test_flux_recall_formats_and_defaults_home(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    seen = {}

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        if not path.endswith("/messages"):
            return {"id": "111111111111111111", "guild_id": "222222222222222222"}
        seen.update(method=method, path=path, params=params)
        return [
            {"id": "1", "timestamp": "t", "author": {"username": "op"},
             "content": "hello\nworld"},
            {"id": "2", "timestamp": "t", "author": {"username": "bot"},
             "content": "", "embeds": [{"title": "card"}]},
        ]

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_recall({})
    assert seen["path"] == "/channels/111111111111111111/messages"
    assert seen["params"]["limit"] == 20
    assert "op: hello\nworld" in out
    assert "embeds=1" in out


def test_flux_recall_reports_unconfigured(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: ("", "", "", ""))
    assert "unavailable" in agent_tools._flux_recall({})


def test_flux_search_extracts_hits(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    seen = {}

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        if method == "GET":
            return {"id": "111111111111111111", "guild_id": "222222222222222222"}
        seen.update(body=body)
        return {"hits": [{"id": "9", "timestamp": "t",
                          "author": {"username": "op"}, "content": "needle"}]}

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_search({"query": "needle", "limit": 5})
    assert seen["body"]["context_channel_id"] == "111111111111111111"
    assert seen["body"]["hits_per_page"] == 5
    assert "needle" in out


def test_flux_members_formats(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **k: [
        {"user": {"id": "u1", "username": "terahertz", "global_name": "Tera"}, "nick": None},
    ])
    out = agent_tools._flux_members({"limit": 10})
    assert "u1 Tera (@terahertz)" in out


def test_flux_audit_formats_entries(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **k: {
        "audit_log_entries": [
            {"id": "e1", "action_type": 73, "user_id": "u1",
             "changes": [{"new_value": "msg-9"}], "reason": "probe cleanup"},
        ],
        "users": [{"id": "u1", "username": "hermes-bot"}],
    })
    out = agent_tools._flux_audit({"limit": 3})
    assert "action=73 actor=hermes-bot target=? reason=probe cleanup" in out
    assert '"new_value": "msg-9"' in out


def test_flux_recall_kwargs_override_positional_dict(monkeypatch):
    """Direct kwargs (tests) win over the gateway-style positional dict."""
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    seen = {}

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        seen.update(params=params, path=path)
        if "/messages" not in path:  # scoping's channel lookup
            return {"id": "123456789012345678", "guild_id": "222222222222222222"}
        return []

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_recall({"limit": 3, "channel_id": "from-dict"}, limit=7, channel_id="123456789012345678")
    assert seen["params"]["limit"] == 7
    assert seen["path"] == "/channels/123456789012345678/messages"
    assert out == "No messages in range."


def test_clamp_int_bounds():
    assert agent_tools._clamp_int("500", 20, 1, 100) == 100
    assert agent_tools._clamp_int("junk", 20, 1, 100) == 20


def test_missing_secret_scope_degrades_to_unconfigured(monkeypatch):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    monkeypatch.setattr(agent_tools, "_derive_home_guild", lambda *a: "")
    assert agent_tools._resolve() == ("https://api.fluxer.app/v1", "", "", "")
    out = agent_tools._flux_recall({})
    assert "unavailable" in out


def test_recall_refuses_channel_outside_home_guild(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        assert path == "/channels/999888777666555"
        return {"id": "999888777666555", "guild_id": "guild-999"}

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_recall({"channel_id": "999888777666555"})
    assert "outside the home guild" in out


def test_recall_scope_global_widens(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": "global" if key == "FLUXER_TOOL_SCOPE" else default)
    seen = {}

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        seen["path"] = path
        return [{"id": "9", "timestamp": "t", "author": {"username": "x"}, "content": "hi"}]

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_recall({"channel_id": "999888777666555"})
    assert seen["path"] == "/channels/999888777666555/messages"
    assert "x: hi" in out


def test_scoping_fails_closed_without_home_guild(monkeypatch):
    """No FLUXER_HOME_GUILD_ID and no derivable guild: reads are refused."""
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", ""))
    monkeypatch.setattr(agent_tools, "_derive_home_guild", lambda *a: "")
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": "" if key == "FLUXER_TOOL_SCOPE" else default)
    out = agent_tools._flux_recall({"channel_id": "123456789012345678"})
    assert "outside the home guild" in out


def test_home_guild_derived_from_home_channel(monkeypatch):
    secrets = {"FLUXER_BOT_TOKEN": "tok", "FLUXER_HOME_CHANNEL": "111111111111111111"}
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": secrets.get(key, default))

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        if "/messages" in path:
            return [{"id": "9", "timestamp": "t", "author": {"username": "x"}, "content": "hi"}]
        return {"id": path.rsplit("/", 1)[-1], "guild_id": "222222222222222222"}

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    out = agent_tools._flux_recall({"channel_id": "999888777666555"})
    assert "x: hi" in out


def test_search_channel_scoping_refuses_foreign_guild(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": "" if key == "FLUXER_TOOL_SCOPE" else default)
    monkeypatch.setattr(agent_tools, "_fluxer_request",
                        lambda *a, **k: {"id": "c", "guild_id": "other"})
    out = agent_tools._flux_search({"query": "x", "channel_id": "123456789012345678"})
    assert "outside the home guild" in out


def test_non_snowflake_channel_id_rejected(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (
        "http://fluxer.local", "tok", "111111111111111111", "222222222222222222"))
    called = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **k: called.append(a))
    out = agent_tools._flux_recall({"channel_id": "../../admin"})
    assert "must be a channel id" in out
    assert not called


def test_base_url_defaults_to_hosted(monkeypatch):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    monkeypatch.setattr(agent_tools, "_derive_home_guild", lambda *a: "")
    base, token, home, guild = agent_tools._resolve()
    assert base == "https://api.fluxer.app/v1"


def test_request_url_uses_resolved_api_base(monkeypatch):
    """Default / hosted / self-hosted bases must not double-append /api/v1."""
    seen = []

    class _Resp:
        def read(self, _n):
            return b"{}"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=15):
        seen.append(req.full_url)
        return _Resp()

    monkeypatch.setattr(agent_tools.urllib.request.OpenerDirector, "open", lambda self, req, timeout=15: fake_urlopen(req, timeout))

    agent_tools._fluxer_request("GET", "/channels/1", "https://api.fluxer.app/v1", "tok")
    agent_tools._fluxer_request("GET", "/channels/1", "https://fluxer.example/api", "tok")
    # resolve() output for an empty secret
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    monkeypatch.setattr(agent_tools, "_derive_home_guild", lambda *a: "")
    base, *_ = agent_tools._resolve()
    agent_tools._fluxer_request("GET", "/channels/1", base, "tok")

    assert seen == [
        "https://api.fluxer.app/v1/channels/1",
        "https://fluxer.example/api/channels/1",
        "https://api.fluxer.app/v1/channels/1",
    ]


def test_api_base_preserves_scoped_and_hosted(monkeypatch):
    assert agent_tools._api_base("https://api.fluxer.app") == "https://api.fluxer.app/v1"
    assert agent_tools._api_base("https://api.fluxer.app/v1") == "https://api.fluxer.app/v1"
    assert agent_tools._api_base("https://fluxer.example") == "https://fluxer.example/api"
    assert agent_tools._api_base("https://fluxer.example/api/v1") == "https://fluxer.example/api/v1"


def test_scheme_refusal(monkeypatch):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    with pytest.raises(ValueError):
        agent_tools._fluxer_request("GET", "/x", "file:///etc", "tok")

def test_guild_lookups_are_fresh_across_deployments(monkeypatch):
    """Identical channel ids on two deployments must not share a guild mapping."""
    calls = []

    def fake_request(method, path, base, token, params=None, body=None, timeout=15):
        calls.append(base)
        if base.endswith("/a"):
            return {"id": "123456789012345678", "guild_id": "444444444444444444"}
        return {"id": "123456789012345678", "guild_id": "555555555555555555"}

    monkeypatch.setattr(agent_tools, "_fluxer_request", fake_request)
    assert agent_tools._derive_home_guild("https://fluxer.example/a", "tok", "123456789012345678") == "444444444444444444"
    assert agent_tools._derive_home_guild("https://fluxer.example/b", "tok", "123456789012345678") == "555555555555555555"
    assert len(calls) == 2  # second profile/base must not reuse the first cache entry
    assert agent_tools._channel_allowed(
        "123456789012345678", "https://fluxer.example/b", "tok", "555555555555555555"
    )

