"""Read-tool trust boundaries; all credentials and responses are synthetic."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import urllib.error

import pytest

import agent_tools
import adapter

CHANNEL = "111111111111111111"
GUILD = "222222222222222222"
FOREIGN = "333333333333333333"
BASE = "https://fluxer.example/api"


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (BASE, "test-token", CHANNEL, GUILD))
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)


@pytest.mark.parametrize("handler,args", [
    (agent_tools._flux_recall, {}),
    (agent_tools._flux_search, {"query": "needle"}),
])
def test_default_channel_is_subject_to_guild_boundary(configured, monkeypatch, handler, args):
    calls = []

    def request(method, path, *a, **kw):
        calls.append(path)
        return {"id": CHANNEL, "guild_id": FOREIGN}

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    assert "refused" in handler(args)
    assert calls == [f"/channels/{CHANNEL}"]


@pytest.mark.parametrize("home", ["", "../admin", "１２３４５", 12345])
def test_invalid_default_channel_never_fetches_content(configured, monkeypatch, home):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (BASE, "test-token", home, GUILD))
    calls = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: calls.append(a))
    assert "failed" in agent_tools._flux_recall({})
    assert not calls


@pytest.mark.parametrize("args", [
    {"before": "../../admin"}, {"before": []},
    {"channel_id": "１２３４５"}, {"channel_id": [CHANNEL]},
])
def test_invalid_recall_arguments_fail_before_lookup(configured, monkeypatch, args):
    calls = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: calls.append(a))
    assert "failed" in agent_tools._flux_recall(args)
    assert not calls


@pytest.mark.parametrize("query", [None, {}, [], 123, "x" * 4001])
def test_search_query_must_be_bounded_text(configured, monkeypatch, query):
    calls = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: calls.append(a))
    assert "failed" in agent_tools._flux_search({"query": query})
    assert not calls


@pytest.mark.parametrize("guild", ["../../admin", "１２３４５", [], ""])
@pytest.mark.parametrize("handler", [agent_tools._flux_members, agent_tools._flux_audit])
def test_guild_tools_validate_configured_id(configured, monkeypatch, guild, handler):
    monkeypatch.setattr(agent_tools, "_resolve", lambda: (BASE, "test-token", CHANNEL, guild))
    calls = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: calls.append(a))
    assert "unavailable" in handler({})
    assert not calls


@pytest.mark.parametrize("response", [[], "bad", {"guild_id": GUILD},
                                        {"id": FOREIGN, "guild_id": GUILD},
                                        {"id": CHANNEL, "guild_id": "bad"}])
def test_channel_metadata_is_validated(monkeypatch, response):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: response)
    assert agent_tools._derive_home_guild(BASE, "test-token", CHANNEL) == ""
    assert not agent_tools._channel_allowed(CHANNEL, BASE, "test-token", GUILD)


def test_authorization_lookups_do_not_reuse_stale_guilds(monkeypatch):
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": default)
    guilds = iter([GUILD, FOREIGN, GUILD, FOREIGN])
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: {
        "id": CHANNEL, "guild_id": next(guilds),
    })
    assert agent_tools._derive_home_guild(BASE, "test-token", CHANNEL) == GUILD
    assert agent_tools._derive_home_guild(BASE, "test-token", CHANNEL) == FOREIGN
    assert agent_tools._channel_allowed(CHANNEL, BASE, "test-token", GUILD)
    assert not agent_tools._channel_allowed(CHANNEL, BASE, "test-token", GUILD)


def test_long_message_content_is_not_silently_lost(configured, monkeypatch):
    content = "x" * 250 + "\ncritical answer after 200 characters"

    def request(method, path, *a, **kw):
        if path == f"/channels/{CHANNEL}":
            return {"id": CHANNEL, "guild_id": GUILD}
        message = {"id": "444444444444444444", "author": {"username": "tester"},
                   "content": content, "embeds": [{"description": "card answer"}],
                   "attachments": [{"filename": "report.txt", "url": "https://cdn.example/report.txt"}]}
        return {"hits": [message]} if method == "POST" else [message]

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    for out in [agent_tools._flux_recall({}), agent_tools._flux_search({"query": "answer"})]:
        assert content in out
        assert "card answer" in out
        assert "https://cdn.example/report.txt" in out


@pytest.mark.parametrize("handler,payload,args", [
    (agent_tools._flux_recall, [{"author": [], "content": 7}], {}),
    (agent_tools._flux_search, {"hits": "not a list"}, {"query": "needle"}),
    (agent_tools._flux_search, [], {"query": "needle"}),
    (agent_tools._flux_members, [{"user": []}], {}),
    (agent_tools._flux_audit, {"audit_log_entries": "bad"}, {}),
    (agent_tools._flux_audit, {"audit_log_entries": [{}], "users": 42}, {}),
    (agent_tools._flux_audit, {"audit_log_entries": [{"changes": {"key": "x"}}]}, {}),
])
def test_malformed_response_is_not_empty_success(configured, monkeypatch, handler, payload, args):
    def request(method, path, *a, **kw):
        if path == f"/channels/{CHANNEL}":
            return {"id": CHANNEL, "guild_id": GUILD}
        return payload

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    assert "unexpected response shape" in handler(args)


@pytest.mark.parametrize("handler,args", [
    (agent_tools._flux_recall, {}), (agent_tools._flux_search, {"query": "x"}),
    (agent_tools._flux_members, {}), (agent_tools._flux_audit, {}),
])
def test_errors_do_not_disclose_provider_or_secret_details(configured, monkeypatch, handler, args, caplog):
    marker = "synthetic-sensitive-token-and-query"

    def request(method, path, *a, **kw):
        if path == f"/channels/{CHANNEL}":
            return {"id": CHANNEL, "guild_id": GUILD}
        raise RuntimeError(marker)

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    out = handler(args)
    assert "failed" in out
    assert marker not in out + caplog.text


def test_clamp_handles_nonfinite_values():
    assert agent_tools._clamp_int(float("inf"), 20, 1, 100) == 20


def test_registration_supplies_function_schema():
    tools = []
    agent_tools.register(None, register_fn=lambda **kw: tools.append(kw))
    for tool in tools:
        assert tool["schema"]["name"] == tool["name"]
        assert tool["schema"]["description"] == tool["description"]
        assert tool["schema"]["parameters"]["type"] == "object"


def test_registration_diagnostics_do_not_include_raw_exception(caplog):
    def register(**kw):
        raise RuntimeError("synthetic-sensitive-registration-error")

    assert agent_tools.register(None, register_fn=register) == 0
    assert "synthetic-sensitive" not in caplog.text


def test_request_rejects_oversized_json_even_with_valid_prefix(monkeypatch):
    monkeypatch.setattr(agent_tools, "_MAX_RESPONSE_BYTES", 8)

    class Response:
        def read(self, n):
            return (b"{}" + b" " * 20)[:n]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    # Patch both transports so this remains independent of redirect implementation.
    monkeypatch.setattr(agent_tools.urllib.request, "urlopen", lambda *a, **kw: Response())
    monkeypatch.setattr(agent_tools.urllib.request.OpenerDirector, "open", lambda *a, **kw: Response())
    with pytest.raises(ValueError, match="response.*large"):
        agent_tools._fluxer_request("GET", "/channels/12345", BASE, "test-token")


def test_redirect_never_forwards_bot_credentials():
    received = []

    class Sink(BaseHTTPRequestHandler):
        def do_GET(self):
            received.append(self.headers.get("Authorization"))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    sink = ThreadingHTTPServer(("127.0.0.1", 0), Sink)

    class Redirect(Sink):
        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{sink.server_port}/stolen")
            self.end_headers()

    source = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
    threads = [Thread(target=server.serve_forever, daemon=True) for server in [sink, source]]
    for thread in threads:
        thread.start()
    try:
        try:
            agent_tools._fluxer_request("GET", "/channels/12345",
                                        f"http://127.0.0.1:{source.server_port}", "synthetic-token")
        except (ValueError, urllib.error.HTTPError):
            pass
        assert not received
    finally:
        for server in [source, sink]:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)


def test_host_context_profiles_never_share_guild_authorization(monkeypatch, tmp_path):
    scope = pytest.importorskip("agent.secret_scope")
    constants = pytest.importorskip("hermes_constants")
    monkeypatch.setattr(scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "wrong-process-token")
    monkeypatch.setenv("FLUXER_TOOL_SCOPE", "global")
    requests = []

    def request(method, path, base, token, **kw):
        requests.append(token)
        return {"id": CHANNEL, "guild_id": GUILD if token == "profile-a" else FOREIGN}

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    for name, expected in [("a", GUILD), ("b", FOREIGN)]:
        secret_binding = scope.set_secret_scope({"FLUXER_BOT_TOKEN": f"profile-{name}",
                                              "FLUXER_HOME_CHANNEL": CHANNEL})
        home_binding = constants.set_hermes_home_override(tmp_path / name)
        try:
            assert agent_tools._resolve() == ("https://api.fluxer.app/v1", f"profile-{name}", CHANNEL, expected)
            assert not agent_tools._channel_allowed(CHANNEL, BASE, f"profile-{name}", "")
        finally:
            constants.reset_hermes_home_override(home_binding)
            scope.reset_secret_scope(secret_binding)
    assert requests == ["profile-a", "profile-b"]


def test_unscoped_multiplex_calls_fail_closed_without_raw_error(monkeypatch):
    scope = pytest.importorskip("agent.secret_scope")
    monkeypatch.setattr(scope, "_MULTIPLEX_ACTIVE", True)
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "wrong-process-token")
    scope_binding = scope.set_secret_scope(None)
    try:
        out = agent_tools._flux_recall({})
        assert "unavailable" in out
        assert "get_secret" not in out
    finally:
        scope.reset_secret_scope(scope_binding)


def test_host_registry_exposes_usable_function_parameters(monkeypatch):
    registry_module = pytest.importorskip("tools.registry")
    monkeypatch.setattr(agent_tools, "_get_secret", lambda key, default="": "test-token" if key == "FLUXER_BOT_TOKEN" else default)
    registry = registry_module.ToolRegistry()
    assert agent_tools.register(None, register_fn=registry.register) == 4
    definitions = registry.get_definitions({"flux_recall", "flux_search", "flux_members", "flux_audit"})
    assert len(definitions) == 4
    for definition in definitions:
        function = definition["function"]
        assert function["description"]
        assert function["parameters"]["type"] == "object"
        assert "limit" in function["parameters"]["properties"]
    assert "required" in next(d["function"] for d in definitions if d["function"]["name"] == "flux_search")["parameters"]


def test_output_budget_marks_omissions(configured, monkeypatch):
    monkeypatch.setattr(agent_tools, "_MAX_OUTPUT_CHARS", 1200, raising=False)

    def request(method, path, *a, **kw):
        if path == f"/channels/{CHANNEL}":
            return {"id": CHANNEL, "guild_id": GUILD}
        return [{"id": "444444444444444444", "content": "x" * 2000}]

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    out = agent_tools._flux_recall({})
    assert len(out) <= 1200
    assert "truncated" in out.lower()


def test_audit_preserves_all_changes_and_target(configured, monkeypatch):
    payload = {"audit_log_entries": [{"id": "444444444444444444", "target_id": FOREIGN,
               "action_type": 72, "changes": [{"key": "name", "old_value": "before", "new_value": "after"},
                                              {"key": "topic", "new_value": "important second change"}]}]}
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: payload)
    out = agent_tools._flux_audit({})
    assert FOREIGN in out
    assert "important second change" in out
    assert json.dumps(payload["audit_log_entries"][0]["changes"], ensure_ascii=False) in out


@pytest.mark.parametrize("handler,args", [
    (agent_tools._flux_recall, ([],)), (agent_tools._flux_search, ("needle",)),
    (agent_tools._flux_members, (None,)), (agent_tools._flux_audit, ({}, {})),
])
def test_invalid_callback_shape_cannot_turn_into_default_read(configured, monkeypatch, handler, args):
    calls = []
    monkeypatch.setattr(agent_tools, "_fluxer_request", lambda *a, **kw: calls.append(a))
    assert "failed" in handler(*args)
    assert not calls


def test_api_search_bounds_match_documented_contract(configured, monkeypatch):
    calls = []

    def request(method, path, *a, **kw):
        calls.append((method, kw))
        if method == "GET":
            return {"id": CHANNEL, "guild_id": GUILD}
        return {"messages": []}

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    assert "failed" in agent_tools._flux_search({"query": "x" * 1025})
    assert not calls
    assert agent_tools._flux_search({"query": "x", "limit": 50}) == "No hits."
    assert calls[-1][1]["body"]["hits_per_page"] == 25


def test_search_indexing_is_not_an_empty_result(configured, monkeypatch):
    def request(method, path, *a, **kw):
        if method == "GET":
            return {"id": CHANNEL, "guild_id": GUILD}
        return {"indexing": True}

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    out = agent_tools._flux_search({"query": "needle"})
    assert "indexing" in out.lower() and "retry" in out.lower()


def test_adapter_registration_exception_is_sanitized(monkeypatch, caplog):
    class Context:
        def register_platform(self, **kw):
            pass

        def register_hook(self, *a, **kw):
            pass

    def broken_register(ctx):
        raise RuntimeError("synthetic-sensitive-registration-error")

    monkeypatch.setattr(agent_tools, "register", broken_register)
    adapter.register(Context())
    assert "synthetic-sensitive" not in caplog.text


@pytest.mark.parametrize("handler,args", [(agent_tools._flux_recall, {}),
                                         (agent_tools._flux_search, {"query": "x"})])
def test_response_cannot_disclose_another_channel(configured, monkeypatch, handler, args):
    def request(method, path, *a, **kw):
        if path == f"/channels/{CHANNEL}":
            return {"id": CHANNEL, "guild_id": GUILD}
        message = {"id": "444444444444444444", "channel_id": FOREIGN, "content": "foreign-sensitive-content"}
        return {"messages": [message]} if method == "POST" else [message]

    monkeypatch.setattr(agent_tools, "_fluxer_request", request)
    out = handler(args)
    assert "failed" in out or "refused" in out
    assert "foreign-sensitive-content" not in out


@pytest.mark.parametrize("value", ["01", "00001", "9223372036854775808", "9" * 25])
def test_snowflake_rejects_noncanonical_and_out_of_range_ids(value):
    assert not agent_tools._is_snowflake(value)


@pytest.mark.parametrize("value", ["0", "1", "9223372036854775807"])
def test_snowflake_accepts_documented_http_range(value):
    assert agent_tools._is_snowflake(value)
