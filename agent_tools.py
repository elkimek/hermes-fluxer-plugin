"""Agent-facing read tools for the Fluxer API (recall, search, members, audit).

Core's ``tools/discord_tool.py`` is hardwired to ``discord.com`` +
``DISCORD_BOT_TOKEN``, so the ``platform_toolsets.fluxer → hermes-discord``
mapping gives the agent tools aimed at the wrong API on a Fluxer deployment.
This module registers a plugin toolset ``fluxer`` whose tools hit
``FLUXER_BASE_URL`` with ``FLUXER_BOT_TOKEN`` instead. Read-only by design:
every mutation already has a first-class surface (reply, clarify cards).

Append ``fluxer`` to the operator's existing ``platform_toolsets.fluxer`` list;
do not replace that list or enable unrelated toolsets. This toolset contains
only the four read tools. Handlers accept the gateway's single-positional-dict
calling convention via ``_merged``.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("fluxer_platform.agent_tools")

_FLUXER_TOOLSET = "fluxer"
_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_OUTPUT_CHARS = 48_000
_MAX_QUERY_CHARS = 1024  # https://docs.fluxer.app/http-api/search


def _get_secret(key: str, default: str = "") -> str:
    """Profile-scope secret read. Imported lazily: ``agent`` is a Hermes-core
    module that standalone CI does not install, and a missing scope must
    degrade to "unconfigured" instead of breaking collection/registration."""
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return default
    return (get_secret(key, default) or default)


def _api_base(base_url: str) -> str:
    """Normalize a Fluxer URL to the REST API base.

    Same rules as ``adapter._api_base`` — kept local so this module stays
    importable without the gateway stack that ``adapter`` pulls in.
    """
    base = (base_url or "").strip().rstrip("/")
    if not base:
        return ""
    if (
        base.endswith("/api")
        or base.endswith("/api/v1")
        or base.endswith("/v1")
        or "/api/" in base
    ):
        return base
    if base == "https://api.fluxer.app":
        return f"{base}/v1"
    return f"{base}/api"


def _resolve() -> Tuple[str, str, str, str]:
    """(api_base, token, home_channel, guild_id) from the profile secret scope."""
    try:
        # Never fall back to another profile's adapter/config or process env.
        base = _api_base(_get_secret("FLUXER_BASE_URL")) or _api_base("https://api.fluxer.app")
        token = _get_secret("FLUXER_BOT_TOKEN").strip()
        home = _get_secret("FLUXER_HOME_CHANNEL").strip()
        guild = _get_secret("FLUXER_HOME_GUILD_ID").strip()
        if not guild and token:
            guild = _derive_home_guild(base, token, home)
        return base, token, home, guild
    except Exception:
        # Includes the host's UnscopedSecretError in an unbound multiplex turn.
        return "", "", "", ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # urllib copies Authorization to redirects, including other origins.
        # A configured API root should not redirect, even to the same origin.
        raise ValueError("Fluxer API redirects are refused")


def _fluxer_request(
    method: str, path: str, base: str, token: str,
    params: Optional[Dict[str, Any]] = None, body: Optional[Dict[str, Any]] = None,
    timeout: int = 15,
) -> Any:
    # ``base`` is already an API root (``…/v1`` or ``…/api``); do not append again.
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment):
        raise ValueError("invalid Fluxer HTTP(S) API URL")
    req = urllib.request.Request(
        url, data=None if body is None else json.dumps(body).encode(), method=method,
        headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
    )
    # scheme is constrained to http(s) above; URL host comes from config, not the model
    opener = urllib.request.build_opener(_NoRedirect())
    with opener.open(req, timeout=timeout) as resp:  # nosec B310
        raw = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError("Fluxer response too large")
        return json.loads(raw.decode("utf-8")) if raw else None


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(value)))
    except (TypeError, ValueError, OverflowError):
        return default


def _merged(args: tuple, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Gateway calls plugin handlers with the args dict as ONE positional;
    direct kwargs (unit tests / manual calls) merge over it."""
    if len(args) > 1 or (args and not isinstance(args[0], dict)):
        return None
    merged: Dict[str, Any] = dict(args[0]) if args else {}
    merged.update(kwargs)
    return merged


_SNOWFLAKE_RE = re.compile(r"(?:0|[1-9][0-9]{0,18})\Z")


def _is_snowflake(value: Any) -> bool:
    # HTTP snowflakes are nonnegative signed-64-bit decimal values with no leading zeroes.
    # https://docs.fluxer.app/snowflakes/
    return (isinstance(value, str) and bool(_SNOWFLAKE_RE.fullmatch(value))
            and int(value) <= 9223372036854775807)


def _derive_home_guild(base: str, token: str, home_channel: str) -> str:
    """Home guild id derived from the home channel when FLUXER_HOME_GUILD_ID
    is unset, so scoping fails closed even in minimal configs."""
    if not token or not _is_snowflake(home_channel):
        return ""
    try:
        info = _fluxer_request("GET", f"/channels/{home_channel}", base, token)
    except Exception:
        return ""
    if not isinstance(info, dict) or info.get("id") != home_channel:
        return ""
    guild = info.get("guild_id")
    return guild if isinstance(guild, str) and _is_snowflake(guild) else ""


def _channel_allowed(channel_id: str, base: str, token: str, home_guild: str) -> bool:
    """Home-guild restriction for model-supplied channel ids.

    Validate the effective target, including the default home channel. There
    are deliberately no persistent authorization caches: fresh lookups avoid
    cross-profile reuse, stale allow decisions, and unbounded cache growth.
    ``FLUXER_TOOL_SCOPE=global`` explicitly widens the guild boundary only.
    """
    if not token or not _is_snowflake(channel_id):
        return False
    try:
        if _get_secret("FLUXER_TOOL_SCOPE").strip().lower() == "global":
            return True
    except Exception:
        return False
    if not _is_snowflake(home_guild):
        return False
    return _derive_home_guild(base, token, channel_id) == home_guild


def _objects(value: Any) -> list:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("unexpected response shape")
    return value


def _text(record: dict, key: str, default: str = "") -> str:
    value = record.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError("unexpected response shape")
    return value


def _output(lines: list, note: str = "") -> str:
    text = "\n".join(lines) + note
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    marker = "\n[Output truncated: reduce limit or narrow the query/range to retrieve omitted text.]"
    return text[:max(0, _MAX_OUTPUT_CHARS - len(marker))] + marker


def _fmt_message(msg: Dict[str, Any], channel_id: str = "") -> str:
    if channel_id and msg.get("channel_id", channel_id) != channel_id:
        raise ValueError("unexpected response shape")
    author = msg.get("author", {})
    if not isinstance(author, dict):
        raise ValueError("unexpected response shape")
    content = _text(msg, "content")
    extras = []
    for field in ("embeds", "attachments"):
        values = _objects(msg.get(field, []))
        if values:
            extras.append(f"{field}={len(values)} {json.dumps(values, ensure_ascii=False)}")
    suffix = "\n" + "\n".join(extras) if extras else ""
    return (f"{_text(msg, 'id', '?')} [{_text(msg, 'timestamp', '?')}] "
            f"{_text(author, 'username', '?')}: {content or '(no content)'}{suffix}")


def _flux_recall(*args: Any, **kwargs: Any) -> str:
    """Fetch the last messages from a Fluxer channel (defaults to the home channel)."""
    a = _merged(args, kwargs)
    if a is None:
        return "Fluxer recall failed: arguments must be a single object."
    channel_id = a.get("channel_id", "")
    limit = _clamp_int(a.get("limit", 20), 20, 1, 100)
    before = a.get("before", "")
    if not isinstance(channel_id, str) or (channel_id and not _is_snowflake(channel_id)):
        return "Fluxer recall failed: channel_id must be a channel id (digits)."
    if not isinstance(before, str) or (before and not _is_snowflake(before)):
        return "Fluxer recall failed: before must be a message id (digits)."
    base, token, home, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    target = channel_id or home
    if not _is_snowflake(target):
        return "Fluxer recall failed: channel_id must be a channel id (digits)."
    if not _channel_allowed(target, base, token, guild):
        return "Fluxer recall refused: channel is outside the home guild (FLUXER_TOOL_SCOPE=global widens this)."
    params: Dict[str, Any] = {"limit": limit}
    if before:
        params["before"] = str(before)
    try:
        msgs = _fluxer_request(
            "GET", f"/channels/{urllib.parse.quote(target)}/messages",
            base, token, params=params)
    except Exception:
        return "Fluxer recall failed: API request unsuccessful; check connectivity and bot permissions."
    try:
        _objects(msgs)
        lines = [_fmt_message(m, target) for m in msgs[:limit]]
    except (TypeError, ValueError):
        return "Fluxer recall failed: unexpected response shape."
    if not msgs:
        return "No messages in range."
    cursor = _text(msgs[min(limit, len(msgs)) - 1], "id")
    note = f"\n[Page of {len(lines)} messages; use before={cursor} for older messages.]" if _is_snowflake(cursor) else ""
    if len(msgs) > limit:
        note += f"\n[Response truncated to requested limit; {len(msgs) - limit} records omitted.]"
    return _output(lines, note)


def _flux_search(*args: Any, **kwargs: Any) -> str:
    """Search messages in a Fluxer channel (defaults to the home channel)."""
    a = _merged(args, kwargs)
    if a is None:
        return "Fluxer search failed: arguments must be a single object."
    query = a.get("query", "")
    channel_id = a.get("channel_id", "")
    limit = _clamp_int(a.get("limit", 10), 10, 1, 25)
    if not isinstance(query, str) or not query.strip() or len(query) > _MAX_QUERY_CHARS:
        return "Fluxer search failed: query is required and must be text of 1-1024 characters."
    if not isinstance(channel_id, str) or (channel_id and not _is_snowflake(channel_id)):
        return "Fluxer search failed: channel_id must be a channel id (digits)."
    base, token, home, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    target = channel_id or home
    if not _is_snowflake(target):
        return "Fluxer search failed: channel_id must be a channel id (digits)."
    if not _channel_allowed(target, base, token, guild):
        return "Fluxer search refused: channel is outside the home guild (FLUXER_TOOL_SCOPE=global widens this)."
    body: Dict[str, Any] = {
        "content": query,
        "context_channel_id": target,
        "hits_per_page": limit,
        "scope": "current",
    }
    try:
        data = _fluxer_request("POST", "/search/messages", base, token, body=body)
    except Exception:
        return "Fluxer search failed: API request unsuccessful; check search support and bot permissions."
    try:
        if not isinstance(data, dict):
            raise ValueError("unexpected response shape")
        if data.get("indexing") is True:
            return "Fluxer search indexing is in progress; retry the same query later."
        hits = _objects(data.get("hits", data.get("messages")))
        lines = [_fmt_message(h, target) for h in hits[:limit]]
    except (TypeError, ValueError):
        return "Fluxer search failed: unexpected response shape."
    if not hits:
        return "No hits."
    return _output(lines, "\n[Search result page; narrow the query for more specific results.]")


def _flux_members(*args: Any, **kwargs: Any) -> str:
    """List members of the Fluxer home guild."""
    a = _merged(args, kwargs)
    if a is None:
        return "Fluxer members failed: arguments must be a single object."
    limit = _clamp_int(a.get("limit", 50), 50, 1, 1000)
    base, token, _, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    if not _is_snowflake(guild):
        return "Fluxer members unavailable: a valid home guild could not be resolved."
    try:
        members = _fluxer_request(
            "GET", f"/guilds/{urllib.parse.quote(guild)}/members", base, token,
            params={"limit": limit})
    except Exception:
        return "Fluxer members failed: API request unsuccessful; check connectivity and bot permissions."
    try:
        _objects(members)
        lines = []
        for member in members[:limit]:
            user = member.get("user")
            if not isinstance(user, dict):
                raise ValueError("unexpected response shape")
            name = _text(member, "nick") or _text(user, "global_name") or _text(user, "username", "?")
            lines.append(f"{_text(user, 'id', '?')} {name} (@{_text(user, 'username', '?')})")
    except (TypeError, ValueError):
        return "Fluxer members failed: unexpected response shape."
    if not members:
        return "No members returned."
    return _output(lines, f"\n[Page of {len(lines)} members; not a complete guild directory.]")


def _flux_audit(*args: Any, **kwargs: Any) -> str:
    """Recent audit-log entries for the Fluxer home guild (45-day retention)."""
    a = _merged(args, kwargs)
    if a is None:
        return "Fluxer audit log failed: arguments must be a single object."
    limit = _clamp_int(a.get("limit", 25), 25, 1, 100)
    base, token, _, guild = _resolve()
    if not base or not token:
        return "Fluxer tools unavailable: FLUXER_BASE_URL / FLUXER_BOT_TOKEN not set."
    if not _is_snowflake(guild):
        return "Fluxer audit log unavailable: a valid home guild could not be resolved."
    try:
        data = _fluxer_request(
            "GET", f"/guilds/{urllib.parse.quote(guild)}/audit-logs", base, token,
            params={"limit": limit})
    except Exception:
        return "Fluxer audit log failed: API request unsuccessful; check connectivity and bot permissions."
    try:
        if not isinstance(data, dict):
            raise ValueError("unexpected response shape")
        entries = _objects(data.get("audit_log_entries"))
        users = {_text(u, "id"): _text(u, "username", "?")
                 for u in _objects(data.get("users", []))}
        lines = []
        for entry in entries[:limit]:
            actor_id = _text(entry, "user_id", "?")
            actor = users.get(actor_id, actor_id)
            changes = _objects(entry.get("changes", []))
            reason = _text(entry, "reason")
            lines.append(
                f"[{_text(entry, 'id', '?')}] action={entry.get('action_type', '?')} actor={actor}"
                + f" target={_text(entry, 'target_id', '?')}"
                + (f" reason={reason}" if reason else "")
                + (f" changes={json.dumps(changes, ensure_ascii=False)}" if changes else "")
                + (f" options={json.dumps(entry['options'], ensure_ascii=False)}" if 'options' in entry else ""))
    except (TypeError, ValueError):
        return "Fluxer audit log failed: unexpected response shape."
    if not entries:
        return "No audit entries returned."
    return _output(lines, f"\n[Page of {len(lines)} audit entries; not the complete audit log.]")


_TOOLS = [
    {
        "name": "flux_recall",
        "description": "Read recent messages from a Fluxer channel (default: the home channel).",
        "schema": {
            "type": "object",
            "properties": {
                "channel_id": {"type": "string", "description": "Channel id (default home channel)"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20,
                          "description": "1-100, default 20"},
                "before": {"type": "string", "description": "Only messages before this message id"},
            },
        },
        "handler": _flux_recall,
    },
    {
        "name": "flux_search",
        "description": "Full-text search of messages in a Fluxer channel (default: the home channel).",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 1024,
                          "description": "Search text (1-1024 characters)"},
                "channel_id": {"type": "string", "description": "Channel id (default home channel)"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 25, "default": 10,
                          "description": "1-25 hits, default 10"},
            },
            "required": ["query"],
        },
        "handler": _flux_search,
    },
    {
        "name": "flux_members",
        "description": "List members of the Fluxer home guild (id, display name, username).",
        "schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50,
                          "description": "1-1000, default 50"},
            },
        },
        "handler": _flux_members,
    },
    {
        "name": "flux_audit",
        "description": "Recent audit-log entries for the Fluxer home guild (45-day retention).",
        "schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 25,
                          "description": "1-100 entries, default 25"},
            },
        },
        "handler": _flux_audit,
    },
]


def register(ctx, *, register_fn: Optional[Callable] = None) -> int:
    """Register the ``fluxer`` toolset; returns the number of tools registered."""
    register_tool = register_fn or ctx.register_tool
    count = 0
    for tool in _TOOLS:
        try:
            register_tool(
                name=tool["name"],
                toolset=_FLUXER_TOOLSET,
                schema={"name": tool["name"], "description": tool["description"],
                        "parameters": tool["schema"]},
                handler=tool["handler"],
                requires_env=["FLUXER_BOT_TOKEN"],
                description=tool["description"],
            )
            count += 1
        except Exception as exc:
            logger.warning("Fluxer tool %s not registered (%s)", tool["name"], type(exc).__name__)
    return count
