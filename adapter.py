"""Fluxer platform plugin for Hermes Agent.

REST + Gateway adapter for messages, media, interactions, reactions, threads,
delivery recovery, and optional LiveKit voice.
)"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import mimetypes
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
import weakref
from collections import OrderedDict
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Mapping, Optional
from urllib.parse import quote, urljoin, urlparse

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_IMAGE_DOCUMENT_TYPES,
    cache_audio_from_url,
    cache_document_from_bytes,
    cache_image_from_url,
    safe_url_for_log,
    utf16_len,
)


def _ensure_fluxer_platform_registered() -> None:
    """Make sure ``Platform(\"fluxer\")`` is a valid enum member.

    When the adapter is instantiated by the Hermes plugin manager, the manager
    has already registered ``fluxer`` via ``platform_registry`` and
    ``Platform._missing_`` will create the pseudo-member on first access.

    When the adapter is instantiated from standalone scripts (voice auto-join
    sidecar, smoke tests), the plugin manager may not have run yet. In that case
    we register a minimal platform entry ourselves so ``Platform(\"fluxer\")``
    succeeds and the adapter's ``self.platform`` matches the real key used in
    ``GatewayRunner.adapters``.
    """
    try:
        Platform("fluxer")
        return
    except ValueError:
        pass
    try:
        from gateway.platform_registry import platform_registry, PlatformEntry

        if not platform_registry.is_registered("fluxer"):
            platform_registry.register(
                PlatformEntry(
                    name="fluxer",
                    label="Fluxer",
                    adapter_factory=lambda cfg: FluxerAdapter(cfg),
                    check_fn=lambda: True,
                    source="plugin",
                )
            )
    except Exception:
        pass


logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
_DEFAULT_BASE_URL = "https://api.fluxer.app/v1"
_GATEWAY_VERSION = 1
_VOICE_STATE_UPDATE_OPCODE = 4
_VOICE_MESSAGE_FLAG = 1 << 13
_RECONNECT_BASE_DELAY = 2.0
_RECONNECT_MAX_DELAY = 60.0
_HEARTBEAT_ACK_TIMEOUT_FACTOR = 2.5
_REST_RATE_LIMIT_MAX_ATTEMPTS = 3
_REST_RATE_LIMIT_MAX_DELAY = 60.0
_PRESENCE_STATUSES = frozenset({"online", "idle", "dnd", "invisible", "offline"})
_DEFAULT_PRESENCE_STATUS = "online"
_DEFAULT_BACKLOG_LIMIT = 25
_DEFAULT_BACKLOG_BOOTSTRAP_SECONDS = 120
_MENTION_EVERYONE_RE = re.compile(r"@(everyone|here)\b", re.IGNORECASE)
_MENTION_ROLE_RE = re.compile(r"<@&\d+>")
_MENTION_USER_RE = re.compile(r"<@!?\d+>")
_VARIATION_SELECTOR_16 = "\ufe0f"
_EXEC_APPROVAL_REACTIONS = (
    ("✅", "once", "approve once"),
    ("🕒", "session", "approve for this session"),
    ("♾️", "always", "always approve"),
    ("❌", "deny", "deny"),
)
# Emoji per approval choice, so a card built from the host's own ``prompt.actions``
# (which carry only label/choice/style) still gets the matching reaction glyph.
_EA_ACTION_EMOJI = {"once": "✅", "session": "🕒", "always": "♾️", "deny": "❌", "cancel": "❌"}
_SLASH_CONFIRM_REACTIONS = (
    ("✅", "once", "approve once"),
    ("♾️", "always", "always approve"),
    ("❌", "cancel", "cancel"),
)
# A pending approval/confirm card is only answerable while the gateway request it
# belongs to can still be answered. Cards older than the host's ``approvals.timeout``
# (plus a grace margin) are pruned, so a stale card cannot linger in the maps and be
# mistaken for a live one. Minimum 300s mirrors the host default.
_APPROVAL_CARD_TTL_FLOOR = 300.0
_APPROVAL_CARD_TTL_GRACE = 60.0


def _approval_card_ttl() -> float:
    """Seconds a pending card stays registered: the host's approval timeout + grace."""
    try:
        from tools.approval_context import _get_approval_timeout

        timeout = float(_get_approval_timeout())
    except Exception:
        timeout = 300.0
    return max(timeout, _APPROVAL_CARD_TTL_FLOOR) + _APPROVAL_CARD_TTL_GRACE


def _exec_approval_choices(
    *, allow_permanent: bool, allow_session: bool, smart_denied: bool
) -> tuple[tuple[str, str, str], ...]:
    """``(emoji, choice, label)`` rows to offer for an exec approval.

    Mirrors the host's own choice policy (``BasePlatformAdapter._exec_approval_actions``):
    ``once`` and ``deny`` are always offered; ``session`` only when the caller allows it
    and the prompt is not a smart deny (a one-operation owner override); ``always`` only
    when the permanent tier is allowed on top of that.
    """
    offered: list[tuple[str, str, str]] = []
    for row in _EXEC_APPROVAL_REACTIONS:
        _emoji, choice, _label = row
        if choice == "session" and (smart_denied or not allow_session):
            continue
        if choice == "always" and (smart_denied or not allow_session or not allow_permanent):
            continue
        offered.append(row)
    return tuple(offered)


_PLUGIN_ROOT = Path(__file__).resolve().parent


def _fluxer_env(name: str, default: Any = None) -> Any:
    """Read a Fluxer setting without crossing a Hermes profile boundary.

    Newer Hermes releases keep each multiplexed profile's ``.env`` in a
    context-local secret scope. Falling back to raw ``os.getenv`` is safe only
    on older Hermes versions that do not provide that scope API.

    Under multiplexing, the DEFAULT profile is constructed and validated
    *unscoped* (``agent.secret_scope.get_secret`` raises
    ``UnscopedSecretError`` there), so prefer Hermes' own
    ``gateway.platforms._shared.get_scoped_secret`` helper, which mirrors the
    default-profile fallback: a scoped miss remains authoritative; an
    unscoped default-profile read uses ``os.environ`` instead of raising.
    """
    try:
        from gateway.platforms._shared import get_scoped_secret
    except ImportError:
        get_scoped_secret = None
    if get_scoped_secret is not None:
        return get_scoped_secret(name, default)
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError:
        return os.getenv(name, default)
    try:
        return get_secret(name, default)
    except UnscopedSecretError:
        return os.getenv(name, default)


def _voice_child_process_env() -> dict[str, str]:
    """Build the trusted voice sidecar env without cross-profile secrets."""
    try:
        from agent.secret_scope import current_secret_scope, is_multiplex_active
    except ImportError:
        multiplexed = False
        scope = None
    else:
        multiplexed = bool(is_multiplex_active())
        scope = current_secret_scope() if multiplexed else None
    if multiplexed and scope is None:
        raise RuntimeError("Fluxer voice child environment requires an active Hermes profile scope")

    try:
        from tools.environments.local import build_subprocess_env

        env = build_subprocess_env(scrub_secrets=True, inherit_profile_home=True)
    except (ImportError, AttributeError, TypeError):
        env = os.environ.copy()
        if multiplexed:
            sensitive_suffixes = ("_TOKEN", "_KEY", "_SECRET", "_PASSWORD", "_CREDENTIAL")
            for key in list(env):
                if key.endswith(sensitive_suffixes):
                    env.pop(key, None)

    credential_keys = {
        "XAI_API_KEY",
        "GROQ_API_KEY",
        "ELEVENLABS_API_KEY",
        "API_SERVER_KEY",
        "API_SERVER_MODEL_NAME",
        "HERMES_LOCAL_STT_COMMAND",
        "XAI_STT_BASE_URL",
        "GROQ_BASE_URL",
        "ELEVENLABS_STT_BASE_URL",
    }
    if multiplexed:
        # Hermes' generic scrubber cannot know a plugin's full FLUXER_* surface.
        # Remove every inherited Fluxer value before copying the active profile
        # scope, otherwise a secondary voice sidecar can silently use the
        # primary profile's bot token or access policy.
        for key in list(env):
            if key.startswith("FLUXER_"):
                env.pop(key, None)
    source: Mapping[str, Any] = scope if scope is not None else os.environ
    for key, value in source.items():
        if key.startswith("FLUXER_") or key in credential_keys:
            env[key] = str(value)

    profile_home = env.get("HERMES_HOME")
    if profile_home:
        env["HERMES_ENV_FILE"] = str(Path(profile_home) / ".env")
    return env


def _strip_slash(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _api_base(base_url: str) -> str:
    """Normalize a user-provided Fluxer URL to the REST API base.

    The official hosted API is already scoped as ``https://api.fluxer.app/v1``.
    Self-hosted Fluxer may expose either an already-scoped API URL or a plain
    web origin; preserve scoped URLs and append ``/api`` only for a plain origin.
    """
    base = _strip_slash(base_url)
    if not base:
        return ""
    if base.endswith("/api") or base.endswith("/api/v1") or base.endswith("/v1") or "/api/" in base:
        return base
    if base == "https://api.fluxer.app":
        return f"{base}/v1"
    return f"{base}/api"


def _coerce_presence_status(value: Any, default: str = _DEFAULT_PRESENCE_STATUS) -> str:
    """Normalize a presence status string; unknown values fall back to default.

    Fluxer converts ``offline`` to ``invisible`` server-side, so accepting it
    here keeps the payload canonical while still honoring explicit config.
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _PRESENCE_STATUSES:
        return "invisible" if text == "offline" else text
    return default


def _build_identify_payload(
    bot_token: str,
    presence_status: str = _DEFAULT_PRESENCE_STATUS,
    presence_afk: bool = False,
) -> Dict[str, Any]:
    """Build Fluxer's opcode-2 identify payload, including initial presence.

    Opcode 2 optionally carries a ``presence`` object (status + afk) so the
    gateway shows the bot as online/idle/dnd/invisible immediately.
    """
    return {
        "op": 2,
        "d": {
            "token": bot_token,
            "properties": {
                "os": "linux",
                "browser": "hermes",
                "device": "hermes",
            },
            "presence": {
                "status": _coerce_presence_status(presence_status),
                "afk": bool(presence_afk),
            },
        },
    }


def _build_presence_update_payload(
    status: str,
    afk: bool = False,
    mobile: bool = False,
) -> Dict[str, Any]:
    """Build Fluxer's opcode-3 presence update payload.

    Sent after READY to (re)assert the bot status on every gateway session.
    """
    return {
        "op": 3,
        "d": {
            "status": _coerce_presence_status(status),
            "afk": bool(afk),
            "mobile": bool(mobile),
        },
    }


def _build_voice_state_update_payload(
    *,
    channel_id: Optional[str],
    guild_id: Optional[str] = None,
    connection_id: Optional[str] = None,
    self_mute: bool = False,
    self_deaf: bool = True,
    self_video: bool = False,
    self_stream: bool = False,
    viewer_stream_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build Fluxer's opcode-4 voice state update payload.

    Fluxer's LiveKit voice handshake starts on the main gateway: the client
    sends VOICE_STATE_UPDATE, then waits for VOICE_SERVER_UPDATE with a LiveKit
    endpoint/token. This helper intentionally does not touch audio media yet;
    it is the tested seam a future realtime bridge can call.
    """
    return {
        "op": _VOICE_STATE_UPDATE_OPCODE,
        "d": {
            "guild_id": guild_id,
            "channel_id": channel_id,
            "self_mute": self_mute,
            "self_deaf": self_deaf,
            "self_video": self_video,
            "self_stream": self_stream,
            "viewer_stream_keys": list(viewer_stream_keys or []),
            "connection_id": connection_id,
        },
    }


def _voice_join_key(guild_id: Optional[str], channel_id: Optional[str]) -> str:
    return f"{guild_id or ''}:{channel_id or ''}"


def _sanitize_voice_server_update(data: Dict[str, Any], *, matched_pending_join: bool) -> Dict[str, Any]:
    """Return non-secret voice-server metadata safe for logs/state."""
    return {
        "guild_id": data.get("guild_id"),
        "channel_id": data.get("channel_id"),
        "connection_id": data.get("connection_id"),
        "endpoint": data.get("endpoint"),
        "has_token": bool(data.get("token")),
        "matched_pending_join": matched_pending_join,
    }


def _headers(bot_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
        "User-Agent": "Hermes-Fluxer/0.3",
    }


def _auth_headers(bot_token: str) -> Dict[str, str]:
    """Headers for requests where httpx must set Content-Type itself."""
    return {
        "Authorization": f"Bot {bot_token}",
        "User-Agent": "Hermes-Fluxer/0.3",
    }


def _redact_fluxer_error_body(body: str, bot_token: str) -> str:
    """Sanitize REST error bodies before writing them to persistent logs."""
    redacted = body or ""
    if bot_token:
        redacted = redacted.replace(bot_token, "[redacted-token]")
    redacted = re.sub(r"Bot\s+[A-Za-z0-9._~+/=-]+", "Bot [redacted-token]", redacted)
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [redacted-token]", redacted)
    redacted = re.sub(r'("authorization"\s*:\s*")([^"]+)(")', r'\1[redacted-token]\3', redacted, flags=re.IGNORECASE)
    redacted = re.sub(r'("token"\s*:\s*")([^"]+)(")', r'\1[redacted-token]\3', redacted, flags=re.IGNORECASE)
    return redacted[:500]


def _event_seq(payload: Dict[str, Any]) -> Optional[int]:
    seq = payload.get("s")
    try:
        return int(seq) if seq is not None else None
    except (TypeError, ValueError):
        return None


try:  # The host's own classifier; absent in the standalone test shim.
    from gateway.platforms.base import classify_send_error as _host_classify_send_error
except Exception:  # pragma: no cover - shim without the classifier
    _host_classify_send_error = None

# HTTP statuses that a retry can never fix: the request was understood and
# refused. 408/429 are the two 4xx that mean "try again", so they stay out.
_PERMANENT_STATUS_KINDS = {
    400: "unknown",
    401: "forbidden",
    403: "forbidden",
    404: "not_found",
    405: "unknown",
    409: "unknown",
    413: "too_long",
    415: "bad_format",
    422: "bad_format",
}
# httpx transport failures that never reached the server: safe to retry.
_CONNECT_LEVEL_EXC_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "PoolTimeout", "NetworkError",
    "RemoteProtocolError", "ReadError", "WriteError", "TransportError",
})
# Read/write timeouts MAY have been accepted by the server before the response
# was lost, so re-sending a POST can duplicate the message: never retry these.
_LANDED_TIMEOUT_EXC_NAMES = frozenset({"ReadTimeout", "WriteTimeout"})


def _http_status_of(exc: BaseException) -> Optional[int]:
    """Status code off the httpx error the REST helpers raise, else ``None``."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    try:
        return int(status) if status is not None else None
    except (TypeError, ValueError):
        return None


def _classify_send_failure(
    exc: BaseException, error_text: str = ""
) -> tuple[bool, Optional[str]]:
    """``(retryable, error_kind)`` for a failed outbound call.

    The host's ``classify_send_error`` supplies the platform-neutral kind where
    it can; the HTTP status is the authoritative signal for retryability, so a
    permanent 4xx refusal (403 forbidden, 404 gone, 413 too long) is not retried
    while a 5xx/429/connection failure is. A read/write timeout is deliberately
    NOT retryable: the POST may already have landed, and retrying duplicates it.
    """
    status = _http_status_of(exc)
    if status is not None:
        kind = None
        if _host_classify_send_error is not None:
            try:
                kind = _host_classify_send_error(exc, error_text)
            except Exception:
                kind = None
        if not kind or kind == "unknown":
            kind = _PERMANENT_STATUS_KINDS.get(status) or (
                "rate_limited" if status == 429 else
                "transient" if status >= 500 or status == 408 else "unknown"
            )
        retryable = status >= 500 or status in (408, 429)
        return retryable, kind

    name = type(exc).__name__
    if name in _CONNECT_LEVEL_EXC_NAMES:
        return True, "transient"
    if name in _LANDED_TIMEOUT_EXC_NAMES:
        return False, "transient"

    kind = None
    if _host_classify_send_error is not None:
        try:
            kind = _host_classify_send_error(exc, error_text)
        except Exception:
            kind = None
    if kind in {"forbidden", "not_found", "too_long", "bad_format"}:
        return False, kind
    if kind == "rate_limited":
        return True, kind
    # Unknown non-HTTP exception: keep the previous permissive default so a
    # transport-shaped failure the classifier cannot name is still retried.
    return True, (kind or "unknown")


def _error_text_for(exc: BaseException) -> str:
    """An error string that always names the exception class.

    httpx timeouts stringify to ``""``, and the host's own timeout/retryable
    checks match on that text — an empty error would read as an ordinary
    formatting failure and let the host re-send a POST that may already have
    landed.
    """
    detail = str(exc).strip()
    name = type(exc).__name__
    return detail if name in detail else (f"{name}: {detail}" if detail else name)


def _failed_send_result(exc: BaseException, *, message_id: Optional[str] = None) -> "SendResult":
    """A ``SendResult`` failure with the retryability/kind classification applied."""
    retryable, error_kind = _classify_send_failure(exc)
    return SendResult(success=False, message_id=message_id, error=_error_text_for(exc),
                      retryable=retryable, error_kind=error_kind)


def _chunk_without_indicator(chunk: str) -> str:
    """A delivered chunk minus the `` (n/m)`` suffix ``truncate_message`` appends."""
    return re.sub(r" \(\d+/\d+\)$", "", chunk)


class _ThreadGone(Exception):
    """A chosen thread was deleted mid-send: the caller falls back to its parent.

    Carries the ids so ``send`` can abandon the dead thread without re-deriving
    them from mutable channel state.
    """

    def __init__(self, thread_id: str, parent_chat: str, original: BaseException) -> None:
        super().__init__(str(original))
        self.thread_id = thread_id
        self.parent_chat = parent_chat
        self.original = original


def _author_name(author: Dict[str, Any]) -> Optional[str]:
    return (
        author.get("global_name")
        or author.get("display_name")
        or author.get("username")
        or author.get("name")
    )


def _chat_type(raw: Any) -> str:
    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"dm", "direct", "private"}:
            return "dm"
        if lowered in {"group_dm", "group", "group-dm"}:
            return "group"
        if lowered in {"thread", "public_thread", "private_thread", "news_thread"}:
            return "thread"
        if lowered in {"forum", "guild_forum"}:
            return "forum"
        try:
            raw = int(lowered)
        except ValueError:
            return "channel"
    # Discord-like channel types in several codebases:
    # 1 = DM, 3 = group DM, 10/11/12 = thread, 15 = forum.
    if raw == 1:
        return "dm"
    if raw == 3:
        return "group"
    if raw in {10, 11, 12}:
        return "thread"
    if raw == 15:
        return "forum"
    return "channel"


def _quote_id(value: Any) -> str:
    """Quote a Fluxer path segment without letting IDs alter REST routes."""
    return quote(str(value or ""), safe="")


def _attachment_url(att: Dict[str, Any]) -> str:
    return str(att.get("url") or att.get("proxy_url") or "").strip()


def _attachment_filename(att: Dict[str, Any]) -> str:
    return str(att.get("filename") or att.get("title") or att.get("name") or "attachment").strip()


def _attachment_content_type(att: Dict[str, Any]) -> str:
    """Return a normalized attachment MIME type, falling back to filename hints."""
    explicit = str(
        att.get("content_type")
        or att.get("contentType")
        or att.get("mime_type")
        or att.get("mimeType")
        or att.get("mimetype")
        or ""
    ).split(";", 1)[0].strip().lower()
    filename = _attachment_filename(att)
    suffix = Path(filename).suffix.lower()
    voice_shaped = bool(att.get("is_voice_message") or att.get("voice") or att.get("voice_message"))
    voice_shaped = voice_shaped or att.get("waveform") not in (None, "", [], {})
    if explicit:
        if voice_shaped and explicit == "video/webm":
            return "audio/webm"
        return explicit
    guessed, _encoding = mimetypes.guess_type(filename)
    normalized = (guessed or "application/octet-stream").split(";", 1)[0].strip().lower()
    if voice_shaped and suffix == ".webm" and normalized == "video/webm":
        return "audio/webm"
    return normalized


def _extension_for_attachment(att: Dict[str, Any], content_type: str, default: str = ".bin") -> str:
    filename = _attachment_filename(att)
    suffix = Path(filename).suffix.lower()
    if suffix:
        return suffix
    subtype = (content_type or "").split("/", 1)[-1].split(";", 1)[0].lower()
    aliases = {"jpeg": ".jpg", "plain": ".txt", "mpeg": ".mp3", "quicktime": ".mov"}
    if subtype:
        return aliases.get(subtype, f".{subtype}")
    return default


# Fallback cap for an inbound attachment body when the host does not expose its
# configured inbound-media limit. Matches the host default
# (``gateway.max_inbound_media_bytes`` = 128 MiB) so a hostile URL cannot stream an
# unbounded body into memory regardless of which host version is running.
_DEFAULT_ATTACHMENT_MAX_BYTES = 128 * 1024 * 1024
_MAX_ATTACHMENT_REDIRECTS = 4


def _attachment_max_bytes() -> int:
    """The host's configured inbound-media cap, else a safe default.

    ``get_inbound_media_max_bytes`` is the host's public reader for
    ``gateway.max_inbound_media_bytes``; an older host without it, or a disabled
    (``0``) cap, falls back to the module default so the download is always bounded.
    """
    try:
        from gateway.platforms.base import get_inbound_media_max_bytes

        limit = int(get_inbound_media_max_bytes())
        return limit if limit > 0 else _DEFAULT_ATTACHMENT_MAX_BYTES
    except Exception:
        return _DEFAULT_ATTACHMENT_MAX_BYTES


def _redirect_target(response: Any) -> Optional[str]:
    """Absolute redirect target visible on an httpx response, or ``None``.

    Prefers the host's own ``redirect_target_from_response`` (which resolves from
    ``Location`` before the often-unpopulated ``next_request``); falls back to the
    same logic locally so an older host still gets a correct answer.
    """
    try:
        from tools.url_safety import redirect_target_from_response

        return redirect_target_from_response(response)
    except Exception:
        pass
    if not getattr(response, "is_redirect", False):
        return None
    location = (getattr(response, "headers", {}) or {}).get("location")
    if location:
        return urljoin(str(getattr(response, "url", "")), str(location))
    next_request = getattr(response, "next_request", None)
    return str(next_request.url) if next_request else None


async def _fluxer_ssrf_redirect_guard(response: Any) -> None:
    """Re-validate every redirect target (defence in depth behind the manual loop).

    Async because httpx awaits response event hooks. A public URL that 302s to
    ``http://169.254.169.254/`` would otherwise slip past the pre-flight
    ``is_safe_url`` check.
    """
    target = _redirect_target(response)
    if target:
        from tools.url_safety import is_safe_url

        if not is_safe_url(target):
            raise ValueError(
                f"Blocked redirect to private/internal address: {safe_url_for_log(target)}"
            )


async def _read_capped_body(response: Any, max_bytes: int) -> bytes:
    """Read a streaming response body under ``max_bytes``.

    Rejects an oversized ``Content-Length`` early, then re-checks the running total
    per chunk so a lying or absent header cannot smuggle a larger body.
    """
    content_length = (getattr(response, "headers", {}) or {}).get("content-length")
    if content_length:
        try:
            declared = int(content_length)
        except (TypeError, ValueError):
            declared = None
        if declared is not None and declared > max_bytes:
            raise ValueError(f"Fluxer attachment exceeds {max_bytes} bytes (declared {declared})")
    chunks: List[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise ValueError(f"Fluxer attachment exceeds {max_bytes} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _ssrf_safe_async_client_factory() -> tuple[Any, bool]:
    """``(factory, guarded)`` for an async HTTP client.

    Prefers the host's connect-time SSRF-safe client (DNS is resolved, validated and
    dialed by IP at connect time, closing the rebinding gap). An older host without
    it degrades to plain ``httpx.AsyncClient`` — and the caller then keeps
    ``follow_redirects`` off and walks redirects manually, so a redirect is still
    re-validated before it is followed.
    """
    import httpx

    try:
        from tools.url_safety import create_ssrf_safe_async_client

        return create_ssrf_safe_async_client, True
    except Exception:
        return httpx.AsyncClient, False


def _message_has_attachments(data: Dict[str, Any]) -> bool:
    """True when a message carries at least one attachment dict (metadata only).

    Used to keep a media-only message eligible for admission WITHOUT fetching
    anything, so no bytes move before the channel/mention gate has decided.
    """
    attachments = data.get("attachments")
    if not isinstance(attachments, list):
        return False
    return any(isinstance(att, dict) and _attachment_url(att) for att in attachments)


def _message_type_for_media(media_types: List[str]) -> MessageType:
    if not media_types:
        return MessageType.TEXT
    if any(m.startswith("image/") for m in media_types):
        return MessageType.PHOTO
    if any(m.startswith("audio/") for m in media_types):
        return MessageType.AUDIO
    if any(m.startswith("video/") for m in media_types):
        return MessageType.VIDEO
    return MessageType.DOCUMENT


def _is_voice_message(data: Dict[str, Any]) -> bool:
    try:
        flags = int(data.get("flags") or 0)
    except (TypeError, ValueError):
        flags = 0
    if flags & _VOICE_MESSAGE_FLAG:
        return True
    if str(data.get("message_type") or data.get("type") or "").lower() in {"voice", "voice_message"}:
        return True
    attachments = data.get("attachments") or []
    if isinstance(attachments, list):
        for att in attachments:
            if not isinstance(att, dict):
                continue
            if bool(att.get("is_voice_message") or att.get("voice") or att.get("voice_message")):
                return True
            waveform = att.get("waveform")
            if waveform not in (None, "", [], {}):
                return True
            duration = att.get("duration")
            try:
                if duration is not None and float(duration) > 0:
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _audio_duration_seconds(path: Path) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nk=1:nw=1", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        duration = float((result.stdout or "").strip())
        if duration > 0:
            return max(1, int(round(duration)))
    except Exception:
        pass
    return 1


def _coerce_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "y", "*"}:
            return True
        if normalized in {"0", "false", "no", "off", "n", ""}:
            return False
    return default


def _split_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip() for item in value if str(item).strip()}
    return {part.strip() for part in str(value).split(",") if part.strip()}


# Outbound reply-reference placement, mirroring the Discord adapter's
# ``reply_to_mode`` contract so Hermes-side expectations stay portable across
# platforms: ``off`` never references, ``first`` references only the first split
# chunk, ``all`` references every chunk.
_REPLY_TO_MODES = ("off", "first", "all")
_DEFAULT_REPLY_TO_MODE = "first"
_DEFAULT_THREAD_NAME = "Hermes reply"
# Thread names are budgeted in UTF-16 code units (an astral emoji counts two), the
# same way the Discord adapter budgets them, so an emoji-heavy title is truncated
# here rather than rejected by the server.
_THREAD_NAME_LIMIT = 80
# Mention markup renders as a pill in the client, never as text a human reads in a
# thread list: <@123>, <@!123>, <@&123>, <#123>.
_MENTION_MARKUP_RE = re.compile(r"<@[!&]?\d+>|<#\d+>")


def _collapse_thread_name(name: str) -> str:
    """Platform-safe thread name: mention markup stripped, single line, name-budget capped."""
    cleaned = _MENTION_MARKUP_RE.sub("", str(name or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if utf16_len(cleaned) > _THREAD_NAME_LIMIT:
        cleaned = cleaned[: _THREAD_NAME_LIMIT - 3].rstrip() + "..."
        while utf16_len(cleaned) > _THREAD_NAME_LIMIT:
            cleaned = cleaned[:-4].rstrip() + "..."
    return cleaned


def _derive_auto_thread_name(content: str) -> str:
    """Placeholder thread name derived from the message that triggered the turn.

    Mirrors the Discord adapter's first phase: the name comes from the user's own
    words (mention markup stripped, whitespace collapsed, capped to the platform's
    name budget) so the thread is identifiable the moment it appears. The
    semantic rename replaces it with the session title once one exists.
    """
    return _collapse_thread_name(content) or _DEFAULT_THREAD_NAME


# ── session-title rename (plugin-local) ─────────────────────────────────────────
# The gateway owns the auto-title lifecycle, and the only plugin-visible surface
# carrying the result is the session store itself, so the adapter reads the title
# back from it rather than reaching into the host's rename lane. Sources that
# carry meaning for a thread name: the model's title and a title the user set.
# ``derived`` is skipped deliberately — it is a slice of the triggering message,
# i.e. what the placeholder already says.
_TITLE_RENAME_SOURCES = ("llm", "user")
# Bounded wait for the title: it is generated by a background titler, so the first
# turns of a session can finish before the title lands.
_TITLE_RENAME_POLL_INTERVAL = 3.0
_TITLE_RENAME_WINDOW_SECONDS = 180.0
_TITLE_RENAME_MAX_PENDING = 64

# Adapters that are currently connected, so the process-wide plugin hooks can find
# the instance that owns a chat.
_LIVE_ADAPTERS: "weakref.WeakSet[FluxerAdapter]" = weakref.WeakSet()


def _current_session_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except ImportError:
        return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes")


def _state_db_path(home: Optional[Path] = None) -> Optional[Path]:
    """Path of the Hermes session store, or ``None`` when it is not there yet."""
    path = (Path(home) if home is not None else _current_session_home()) / "state.db"
    return path if path.exists() else None


def _open_session_store(home: Optional[Path] = None) -> Optional[Any]:
    """Hermes' own session store, opened read-only, or ``None`` when unavailable.

    Reads the session title back out of ``state.db`` through core's ``SessionDB``
    rather than a hand-rolled ``sqlite3`` connection. ``SessionDB`` owns the
    read-only URI construction (a raw ``file:`` URI assembled with an f-string
    truncates at a ``?`` or ``#`` in the home path and silently opens the wrong
    database), pools a bounded set of read descriptors instead of opening one per
    lookup, and retries the transient ``SQLITE_IOERR`` a warm read-only reader can
    hit during a sibling process's WAL checkpoint.

    Every failure — no store yet, no Hermes runtime, a store this build cannot
    read — is a missing name, never a broken turn.
    """
    path = _state_db_path(home)
    if path is None:
        return None
    try:
        from hermes_state import SessionDB  # type: ignore

        return SessionDB(db_path=path, read_only=True)
    except Exception as exc:
        logger.debug("Fluxer session store unavailable: %s", type(exc).__name__)
        return None


def _is_missing_channel_error(exc: BaseException) -> bool:
    """True when ``exc`` means the target channel or thread no longer exists.

    A thread can be deleted underneath a running turn, so an automatically chosen
    thread has to be abandoned rather than failing the send. The status is read
    off the httpx error the REST helpers raise; any other failure is a real
    delivery problem and must not be silently redirected to the parent channel.
    """
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status == 404:
        return True
    if status == 400:
        body = str(getattr(response, "text", "") or "").lower()
        return "unknown channel" in body or "unknown thread" in body
    return False


def _thread_already_exists_for_message(exc: BaseException) -> bool:
    """True when Fluxer refused a thread start because the message already has one.

    Fluxer anchors a thread to the message it was started from and the thread
    carries that message's id, so this refusal *names* the existing thread. The
    adapter's anchor map lives in memory only -- a restart, an evicted entry, or a
    start that raced another send all reach here -- and reading the refusal as
    "no thread" demotes the whole turn to the parent channel even though the
    thread is already there.
    """
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 400:
        return False
    return "thread_already_created_for_message" in str(getattr(response, "text", "") or "").lower()


def _is_reply_reference_rejected_error(exc: BaseException) -> bool:
    """True when the send failed because the *reply reference* itself was refused.

    Fluxer rejects a reference to a message it types as a system message (the
    gateway's own progress bubbles arrive as ``type: 4``), answering
    ``400 CANNOT_REPLY_TO_SYSTEM_MESSAGE`` with ``path: message_reference``. The
    refusal is about the reference, never about the content, so re-sending the
    identical payload cannot succeed: the reply has to go out without the
    reference instead of being dropped.
    """
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 400:
        return False
    body = str(getattr(response, "text", "") or "").lower()
    return "cannot_reply_to_system_message" in body


def _is_text_command(text: str) -> bool:
    """True when ``text`` is a gateway command typed as a plain message.

    Fluxer registers no native application commands, so ``/new``, ``/help``,
    ``/stop`` and ``/title`` arrive as ordinary text and the adapter's own
    admission gate decides whether the command pipeline ever sees them. A leading
    slash is how the host's ``MessageEvent.is_command()`` reads a command, and
    this mirrors its two extra rules: a ``@botname`` suffix is allowed, and a name
    containing ``/`` is not a command at all (a pasted path stays ordinary text).
    """
    stripped = str(text or "").lstrip()
    if not stripped.startswith("/"):
        return False
    name = stripped.split(maxsplit=1)[0][1:].split("@", 1)[0]
    return bool(name) and "/" not in name


def _thread_id_from_session_key(session_key: Optional[str], platform: str) -> Optional[str]:
    """The thread container a session key is bound to, or ``None``.

    A thread-keyed session names its container in the ``chat_id`` slot
    (``<ns>:<platform>:thread:<thread-id>[:<user>]``), so an operation that has only
    the session key -- an approval prompt, a confirmation -- can still find the
    thread the waiting turn lives in. Anything that is not a thread-keyed key for
    this platform (a channel session, another platform) yields ``None``.
    """
    parts = str(session_key or "").split(":")
    # <ns>:<platform>:<chat_type>:<chat_id>[:<user>]  (ns is always two slots)
    if len(parts) < 5 or parts[2] != str(platform) or parts[3] != "thread":
        return None
    return parts[4] or None


def _coerce_reply_to_mode(value: Any, default: str = _DEFAULT_REPLY_TO_MODE) -> str:
    """Normalize an outbound reply-reference mode; unknown values use ``default``."""
    mode = str(value or "").strip().lower()
    return mode if mode in _REPLY_TO_MODES else default


def _voice_config_dict(extra: Dict[str, Any]) -> Dict[str, Any]:
    raw = extra.get("voice") if isinstance(extra, dict) else None
    return raw if isinstance(raw, dict) else {}


def _voice_setting(extra: Dict[str, Any], key: str, env_name: str, default: Any = None) -> Any:
    env_value = _fluxer_env(env_name)
    if env_value is not None:
        return env_value
    voice = _voice_config_dict(extra)
    if key in voice:
        return voice.get(key)
    return default


def _voice_bool_setting(extra: Dict[str, Any], key: str, env_name: str, default: bool = False) -> bool:
    return _coerce_bool(_voice_setting(extra, key, env_name, default), default)


def _voice_csv_setting(extra: Dict[str, Any], key: str, env_name: str) -> str:
    value = _voice_setting(extra, key, env_name, "")
    if value is None:
        return ""
    return ",".join(sorted(_split_ids(value)))


class FluxerVoiceSupervisorProcess:
    """Plugin-managed wrapper for the optional realtime voice auto-join supervisor."""

    def __init__(
        self,
        *,
        extra: Dict[str, Any],
        plugin_root: Path = _PLUGIN_ROOT,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.extra = extra
        self.plugin_root = plugin_root
        self.popen_factory = popen_factory
        self.process: Optional[subprocess.Popen] = None
        self._watch_task: Optional[asyncio.Task[Any]] = None
        self._watch_process_ref: Optional[subprocess.Popen] = None
        self._restart_delay_seconds = 2.0

    @property
    def enabled(self) -> bool:
        return _voice_bool_setting(self.extra, "enabled", "FLUXER_VOICE_ENABLED", False)

    @property
    def auto_join(self) -> bool:
        return _voice_bool_setting(self.extra, "auto_join", "FLUXER_VOICE_AUTO_JOIN", False)

    @property
    def supervisor_disabled(self) -> bool:
        return _voice_bool_setting(
            self.extra,
            "supervisor_disabled",
            "FLUXER_VOICE_SUPERVISOR_DISABLED",
            False,
        )

    @property
    def configured_channel_ids(self) -> str:
        return _voice_csv_setting(self.extra, "channel_ids", "FLUXER_VOICE_CHANNEL_IDS")

    def should_start(self) -> bool:
        return not self.supervisor_disabled and self.enabled and self.auto_join and bool(self.configured_channel_ids)

    def build_command(self) -> list[str]:
        python = _fluxer_env("FLUXER_VOICE_PYTHON", sys.executable)
        return [python, str(self.plugin_root / "scripts" / "fluxer_voice_auto_join.py")]

    def _child_env(self) -> Dict[str, str]:
        env = _voice_child_process_env()
        # Child scripts create their own FluxerAdapter instances to listen for
        # voice gateway events / perform the LiveKit handshake. Those internal
        # adapters must not start another plugin-managed supervisor from their
        # own connect() call, while the scripts still need ENABLED/AUTO_JOIN to
        # pass their explicit run gates.
        env["FLUXER_VOICE_SUPERVISOR_DISABLED"] = "true"
        for extra_key, env_name in {
            "bot_token": "FLUXER_BOT_TOKEN",
            "base_url": "FLUXER_BASE_URL",
            "gateway_url": "FLUXER_GATEWAY_URL",
            "allowed_users": "FLUXER_ALLOWED_USERS",
            "allow_all_users": "FLUXER_ALLOW_ALL_USERS",
            "backlog_recovery": "FLUXER_BACKLOG_RECOVERY",
            "backlog_limit": "FLUXER_BACKLOG_LIMIT",
            "backlog_bootstrap_seconds": "FLUXER_BACKLOG_BOOTSTRAP_SECONDS",
            "require_mention": "FLUXER_REQUIRE_MENTION",
            "strict_mention": "FLUXER_STRICT_MENTION",
            "free_response_channels": "FLUXER_FREE_RESPONSE_CHANNELS",
            "home_guild_id": "FLUXER_HOME_GUILD_ID",
            "home_guilds": "FLUXER_HOME_GUILDS",
            "auto_free_response_home_guild": "FLUXER_AUTO_FREE_RESPONSE_HOME_GUILD",
            "mention_gated_channels": "FLUXER_MENTION_GATED_CHANNELS",
            "allowed_channels": "FLUXER_ALLOWED_CHANNELS",
            "mention_patterns": "FLUXER_MENTION_PATTERNS",
            "allow_mention_everyone": "FLUXER_ALLOW_MENTION_EVERYONE",
            "allow_mention_roles": "FLUXER_ALLOW_MENTION_ROLES",
            "allow_mention_users": "FLUXER_ALLOW_MENTION_USERS",
            "allow_mention_replied_user": "FLUXER_ALLOW_MENTION_REPLIED_USER",
            "delivery_verification": "FLUXER_DELIVERY_VERIFICATION",
            "application_id": "FLUXER_APPLICATION_ID",
        }.items():
            if env_name not in env and extra_key in self.extra:
                value = self.extra.get(extra_key)
                if value is not None:
                    env[env_name] = _coerce_env_value(value).strip()
        voice = _voice_config_dict(self.extra)
        mappings = {
            "enabled": "FLUXER_VOICE_ENABLED",
            "auto_join": "FLUXER_VOICE_AUTO_JOIN",
            "target_user_ids": "FLUXER_VOICE_TARGET_USER_IDS",
            "channel_ids": "FLUXER_VOICE_CHANNEL_IDS",
            "guild_ids": "FLUXER_VOICE_GUILD_IDS",
            "participant_prefix": "FLUXER_VOICE_PARTICIPANT_PREFIX",
            "brain_provider": "FLUXER_VOICE_BRAIN_PROVIDER",
            "stt_provider": "FLUXER_VOICE_STT_PROVIDER",
            "stt_model": "FLUXER_VOICE_STT_MODEL",
            "elevenlabs_language_code": "FLUXER_VOICE_ELEVENLABS_LANGUAGE_CODE",
            "tts_voice": "FLUXER_VOICE_TTS_VOICE",
            "context_file": "FLUXER_VOICE_CONTEXT_FILE",
            "session_db": "FLUXER_VOICE_SESSION_DB",
            "turn_log_jsonl": "FLUXER_VOICE_TURN_LOG_JSONL",
            "python": "FLUXER_VOICE_PYTHON",
            "hermes_url": "FLUXER_VOICE_HERMES_URL",
            "hermes_model": "FLUXER_VOICE_HERMES_MODEL",
            "hermes_timeout_seconds": "FLUXER_VOICE_HERMES_TIMEOUT_SECONDS",
            "hermes_max_tokens": "FLUXER_VOICE_HERMES_MAX_TOKENS",
            "hermes_temperature": "FLUXER_VOICE_HERMES_TEMPERATURE",
            "hermes_session_id": "FLUXER_VOICE_HERMES_SESSION_ID",
            "hermes_session_key": "FLUXER_VOICE_HERMES_SESSION_KEY",
            "max_turns": "FLUXER_VOICE_MAX_TURNS",
            "initial_settle_seconds": "FLUXER_VOICE_INITIAL_SETTLE_SECONDS",
            "sample_rate": "FLUXER_VOICE_SAMPLE_RATE",
        }
        for key, env_name in mappings.items():
            if env_name not in env and key in voice:
                value = voice.get(key)
                if value is not None:
                    env[env_name] = _coerce_env_value(value)
        for section_name, nested in {
            "vad": {
                "silence_ms": "FLUXER_VOICE_SILENCE_MS",
                "end_padding_ms": "FLUXER_VOICE_END_PADDING_MS",
                "min_segment_ms": "FLUXER_VOICE_MIN_SEGMENT_MS",
                "max_segment_seconds": "FLUXER_VOICE_MAX_SEGMENT_SECONDS",
                "energy_threshold": "FLUXER_VOICE_ENERGY_THRESHOLD",
                "frame_ms": "FLUXER_VOICE_FRAME_MS",
            },
            "timeouts": {
                "capture_seconds": "FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS",
                "connect_seconds": "FLUXER_VOICE_CONNECT_TIMEOUT_SECONDS",
                "xai_seconds": "FLUXER_VOICE_XAI_TIMEOUT_SECONDS",
                "xai_first_audio_seconds": "FLUXER_VOICE_XAI_FIRST_AUDIO_TIMEOUT_SECONDS",
                "max_runtime_seconds": "FLUXER_VOICE_MAX_RUNTIME_SECONDS",
                "start_cooldown_seconds": "FLUXER_VOICE_START_COOLDOWN_SECONDS",
                "stop_timeout_seconds": "FLUXER_VOICE_STOP_TIMEOUT_SECONDS",
            },
            "barge_in": {
                "disable": "FLUXER_VOICE_DISABLE_BARGE_IN",
                "energy_threshold": "FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD",
                "min_ms": "FLUXER_VOICE_BARGE_IN_MIN_MS",
                "window_ms": "FLUXER_VOICE_BARGE_IN_WINDOW_MS",
                "capture_timeout_seconds": "FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS",
                "stop_phrase_energy_threshold": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD",
                "stop_phrase_min_ms": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS",
                "stop_phrase_silence_ms": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS",
                "stop_phrase_max_seconds": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS",
                "after_first_audio_only": "FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY",
            },
        }.items():
            values = voice.get(section_name)
            if isinstance(values, dict):
                for key, env_name in nested.items():
                    if env_name not in env and key in values:
                        value = values[key]
                        if value is not None:
                            env[env_name] = _coerce_env_value(value)
        return env

    def start(self) -> bool:
        if self.process and self.process.poll() is None:
            return True
        if not self.should_start():
            logger.info(
                "Fluxer realtime voice supervisor disabled or not scoped; disabled=%s enabled=%s auto_join=%s channels_configured=%s",
                self.supervisor_disabled,
                self.enabled,
                self.auto_join,
                bool(self.configured_channel_ids),
            )
            return False
        script = self.plugin_root / "scripts" / "fluxer_voice_auto_join.py"
        if not script.exists():
            logger.warning("Fluxer realtime voice supervisor script missing: %s", script)
            return False
        try:
            self.process = self.popen_factory(
                self.build_command(),
                cwd=str(self.plugin_root),
                env=self._child_env(),
                start_new_session=True,
            )
        except Exception as exc:
            self.process = None
            logger.warning(
                "Fluxer realtime voice supervisor failed to start; continuing without voice supervisor: %s",
                exc,
            )
            return False
        logger.info("Fluxer realtime voice supervisor started with PID %s", getattr(self.process, "pid", None))
        proc = self.process
        if proc is not None:
            self._start_watch_task(proc)
        return True

    def _start_watch_task(self, proc: subprocess.Popen) -> None:
        if self._watch_task is not None and not self._watch_task.done():
            if self._watch_process_ref is proc:
                return
            self._watch_task.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._watch_process_ref = proc
        self._watch_task = loop.create_task(self._watch_process(proc), name="fluxer-voice-supervisor-watch")

    async def _watch_process(self, proc: subprocess.Popen) -> None:
        current_task = asyncio.current_task()
        try:
            return_code = await asyncio.to_thread(proc.wait)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer realtime voice supervisor watcher stopped: %s", exc)
            return
        try:
            if self.process is not proc:
                return
            self.process = None
            if not self.should_start():
                logger.info("Fluxer realtime voice supervisor exited with code %s", return_code)
                return
            logger.warning("Fluxer realtime voice supervisor exited with code %s; restarting", return_code)
            await asyncio.sleep(self._restart_delay_seconds)
            if self.process is None and self.should_start():
                if self._watch_task is current_task:
                    self._watch_task = None
                    self._watch_process_ref = None
                self.start()
        finally:
            if self._watch_task is current_task:
                self._watch_task = None
                self._watch_process_ref = None

    async def stop(self) -> None:
        watch_task = self._watch_task
        self._watch_task = None
        self._watch_process_ref = None
        if watch_task is not None and not watch_task.done():
            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await watch_task
        proc = self.process
        self.process = None
        if proc is None or proc.poll() is not None:
            return
        self._terminate_process(proc)
        try:
            await asyncio.to_thread(proc.wait, 8)
        except subprocess.TimeoutExpired:
            self._kill_process(proc)
            try:
                await asyncio.to_thread(proc.wait, 5)
            except subprocess.TimeoutExpired:
                logger.warning("Fluxer realtime voice supervisor did not exit after SIGKILL")
        logger.info("Fluxer realtime voice supervisor stopped")

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        self._signal_process_group(proc, getattr(signal, "SIGTERM", 15), fallback=proc.terminate)

    def _kill_process(self, proc: subprocess.Popen) -> None:
        self._signal_process_group(proc, getattr(signal, "SIGKILL", 9), fallback=proc.kill)

    @staticmethod
    def _signal_process_group(proc: subprocess.Popen, sig: int, *, fallback: Callable[[], None]) -> None:
        pid = getattr(proc, "pid", None)
        if os.name == "posix" and isinstance(pid, int) and pid > 0:
            try:
                os.killpg(os.getpgid(pid), sig)
                return
            except ProcessLookupError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    fallback()
                return
            except Exception:
                pass
        fallback()



def _parse_fluxer_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_message_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("messages", "items", "data", "results"):
            items = payload.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def _message_content(data: Any) -> str:
    if not isinstance(data, dict):
        return ""
    content = data.get("content")
    return content if isinstance(content, str) else ""


def _extract_reply_message(data: Dict[str, Any]) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Return the replied-to message id and any embedded referenced payload.

    Fluxer deployments vary while the API is young: some
    events embed ``referenced_message``, some use ``message_reference`` only,
    and older bridge payloads may call it ``reply_to``. Normalize all obvious
    forms so Hermes gets reply context whenever the information is available.
    """
    candidates = (
        data.get("referenced_message"),
        data.get("referencedMessage"),
        data.get("reply_to_message"),
        data.get("replyToMessage"),
        data.get("reply_to"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            msg_id = candidate.get("id") or candidate.get("message_id")
            if msg_id:
                return str(msg_id), candidate
        elif candidate:
            return str(candidate), None

    reference = data.get("message_reference") or data.get("messageReference") or {}
    if isinstance(reference, dict):
        msg_id = reference.get("message_id") or reference.get("messageId") or reference.get("id")
        if msg_id:
            return str(msg_id), None
    return None, None


def _fluxer_component_row(buttons: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"type": 1, "components": buttons}]


def _fluxer_button(*, custom_id: str, label: str, style: int) -> Dict[str, Any]:
    return {"type": 2, "style": style, "custom_id": custom_id[:100], "label": label, "disabled": False}


def _fluxer_action_buttons(
    *,
    prefix: str,
    specs: tuple[tuple[str, str, str], ...],
    danger_choice: str,
) -> tuple[List[Dict[str, Any]], List[tuple[str, str]]]:
    """Build Fluxer component buttons plus action ids for a native-controls prompt."""
    buttons: List[Dict[str, Any]] = []
    actions: List[tuple[str, str]] = []
    for _emoji, choice, label in specs:
        custom_id = f"{prefix}:{uuid.uuid4().hex}:{choice}"
        actions.append((custom_id, choice))
        buttons.append(_fluxer_button(custom_id=custom_id, label=label, style=4 if choice == danger_choice else 3))
    return buttons, actions


def _attachment_is_voice_shaped(att: Dict[str, Any]) -> bool:
    """Return True when an individual attachment carries Fluxer voice-message shape."""
    if att.get("is_voice_message") or att.get("voice") or att.get("voice_message"):
        return True
    if att.get("waveform") not in (None, "", [], {}):
        return True
    for key in ("duration", "duration_secs", "duration_seconds"):
        value = att.get(key)
        if value is None:
            continue
        try:
            if float(value) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _voice_attachment_metadata(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return safe, normalized metadata for the first Fluxer voice attachment."""
    if not _is_voice_message(data):
        return None
    attachments = data.get("attachments") or []
    if not isinstance(attachments, list):
        return {"is_voice_message": True}
    for att in attachments:
        if not isinstance(att, dict) or not _attachment_is_voice_shaped(att):
            continue
        content_type = _attachment_content_type(att)
        meta: Dict[str, Any] = {
            "is_voice_message": True,
            "attachment_id": str(att.get("id") or "") or None,
            "filename": _attachment_filename(att),
            "content_type": content_type,
        }
        duration = next(
            (att.get(key) for key in ("duration", "duration_secs", "duration_seconds") if att.get(key) is not None),
            None,
        )
        try:
            if duration is not None:
                meta["duration_seconds"] = float(duration)
        except (TypeError, ValueError):
            pass
        waveform = att.get("waveform")
        if waveform not in (None, "", [], {}):
            # Store shape/presence only; waveform blobs can be large and are not
            # useful for Hermes turn context.
            meta["has_waveform"] = True
        return {key: value for key, value in meta.items() if value is not None}
    return {"is_voice_message": True}


def _looks_like_fluxer_id(value: str) -> bool:
    """Return True for path-safe Fluxer IDs, false for tokens/URLs/etc."""
    text = str(value or "").strip()
    return bool(text) and not any(ch in text for ch in ".:/\\?&#\r\n\t ")


def _normalize_reaction_emoji(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name") or value.get("emoji") or value.get("id") or ""
    text = str(value or "").strip()
    # Fluxer/clients may round-trip text emoji with or without VS16. Compare
    # both forms so ♾ and ♾️ resolve the same pending approval.
    return text.replace(_VARIATION_SELECTOR_16, "")


def _normalize_approval_command(command: Any) -> str:
    """Redact a command for comparison against the host's pending-approval snapshot.

    The host stores the approval prompt's command already redacted; the runner hands
    the adapter a *differently* redacted copy of the same string. Normalizing both
    through the force-redactor (idempotent, verified) makes the two comparable without
    ever holding a raw secret.
    """
    text = str(command or "")
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except Exception:
        return text


def _reaction_choice_map(specs: Any) -> Dict[str, str]:
    return {_normalize_reaction_emoji(emoji): choice for emoji, choice, _label in specs}


def _reaction_emoji_from_event(data: Dict[str, Any]) -> str:
    return _normalize_reaction_emoji(
        data.get("emoji")
        or data.get("emoji_name")
        or data.get("reaction")
        or ((data.get("reaction_data") or {}).get("emoji"))
    )


def _directory_channel_type(raw: Any) -> Optional[str]:
    if raw in (0, "0", "text", "guild_text"):
        return "channel"
    if raw in (5, "5", "announcement", "news"):
        return "channel"
    if raw in (10, 11, 12, "10", "11", "12", "thread", "public_thread", "private_thread"):
        return "thread"
    if raw in (15, "15", "forum", "guild_forum"):
        return "forum"
    if raw in (1, "1", "dm", "direct"):
        return "dm"
    if raw in (3, "3", "group_dm", "group"):
        return "group"
    return None


class FluxerAdapter(BasePlatformAdapter):
    """Fluxer adapter using bot REST + Gateway websocket APIs."""

    # Fluxer renders fenced markdown code blocks with syntax highlighting/copy UI.
    # Hermes gateway uses this capability flag to display terminal tool commands
    # as clean fenced blocks in editable tool-progress messages.
    supports_code_blocks = True
    # send() chunks with BasePlatformAdapter.truncate_message(). Advertising
    # this keeps cron delivery from pre-truncating a multi-message report into a
    # single 4,000-codepoint payload. That old path could still exceed Fluxer's
    # 4,000 UTF-16-unit API limit when it contained astral emoji.
    # Fluxer's message API counts UTF-16 code units and caps a message at 4,000
    # of them, so the host's length contract must read this adapter's own cap and
    # unit rather than its 4096-codepoint default.
    MAX_MESSAGE_LENGTH = MAX_MESSAGE_LENGTH
    splits_long_messages = True

    @property
    def message_len_fn(self) -> Callable[[str], int]:
        """Length unit Fluxer counts in (UTF-16 code units, not codepoints)."""
        return utf16_len

    def max_message_length_for_chat(self, chat_id: str) -> int:
        return MAX_MESSAGE_LENGTH

    def message_len_fn_for_chat(self, chat_id: str) -> Callable[[str], int]:
        return utf16_len

    def __init__(self, config: PlatformConfig):
        _ensure_fluxer_platform_registered()
        super().__init__(config, Platform("fluxer"))
        extra = getattr(config, "extra", {}) or {}
        self._gateway_state_updates_enabled = _coerce_bool(extra.get("gateway_state_updates"), True)
        self.base_url = _strip_slash(
            _fluxer_env("FLUXER_BASE_URL") or extra.get("base_url") or _DEFAULT_BASE_URL
        )
        self.api_base_url = _api_base(self.base_url)
        self.bot_token = (
            _fluxer_env("FLUXER_BOT_TOKEN") or extra.get("bot_token") or ""
        ).strip()
        self.gateway_url = _strip_slash(
            _fluxer_env("FLUXER_GATEWAY_URL") or extra.get("gateway_url") or ""
        )
        self.bot_user_id: Optional[str] = str(extra.get("bot_user_id")) if extra.get("bot_user_id") else None
        self._presence_status = _coerce_presence_status(
            _fluxer_env("FLUXER_PRESENCE_STATUS") or extra.get("presence_status")
        )
        self._presence_afk = _coerce_bool(
            _fluxer_env("FLUXER_PRESENCE_AFK", extra.get("presence_afk")),
            False,
        )
        self._allowed_user_ids = _split_ids(_fluxer_env("FLUXER_ALLOWED_USERS") or extra.get("allowed_users"))
        self._allow_all_users = _coerce_bool(_fluxer_env("FLUXER_ALLOW_ALL_USERS", extra.get("allow_all_users")), False)
        self._backlog_enabled = str(
            _fluxer_env("FLUXER_BACKLOG_RECOVERY", extra.get("backlog_recovery", "true"))
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._backlog_limit = min(
            _coerce_int(_fluxer_env("FLUXER_BACKLOG_LIMIT") or extra.get("backlog_limit"), _DEFAULT_BACKLOG_LIMIT),
            100,
        )
        self._backlog_bootstrap_seconds = min(
            _coerce_int(
                _fluxer_env("FLUXER_BACKLOG_BOOTSTRAP_SECONDS") or extra.get("backlog_bootstrap_seconds"),
                _DEFAULT_BACKLOG_BOOTSTRAP_SECONDS,
            ),
            900,
        )
        self._delivery_verification_enabled = str(
            _fluxer_env("FLUXER_DELIVERY_VERIFICATION", extra.get("delivery_verification", "true"))
        ).strip().lower() not in {"0", "false", "no", "off"}
        self._allow_mention_everyone = _coerce_bool(
            _fluxer_env("FLUXER_ALLOW_MENTION_EVERYONE", extra.get("allow_mention_everyone")),
            False,
        )
        self._allow_mention_roles = _coerce_bool(
            _fluxer_env("FLUXER_ALLOW_MENTION_ROLES", extra.get("allow_mention_roles")),
            False,
        )
        self._allow_mention_users = _coerce_bool(
            _fluxer_env("FLUXER_ALLOW_MENTION_USERS", extra.get("allow_mention_users")),
            True,
        )
        self._allow_mention_replied_user = _coerce_bool(
            _fluxer_env("FLUXER_ALLOW_MENTION_REPLIED_USER", extra.get("allow_mention_replied_user")),
            True,
        )
        self._require_mention = _coerce_bool(
            _fluxer_env("FLUXER_REQUIRE_MENTION", extra.get("require_mention")),
            True,
        )
        self._strict_mention = _coerce_bool(
            _fluxer_env("FLUXER_STRICT_MENTION", extra.get("strict_mention")),
            False,
        )
        self._free_response_channels = _split_ids(
            _fluxer_env("FLUXER_FREE_RESPONSE_CHANNELS") or extra.get("free_response_channels")
        )
        self._mention_gated_channels = _split_ids(
            _fluxer_env("FLUXER_MENTION_GATED_CHANNELS") or extra.get("mention_gated_channels")
        )
        # Outbound reply/thread behavior. ``reply_to_mode`` mirrors the Discord
        # adapter's contract (off/first/all) so Hermes-side expectations --
        # progress-reply placement, handoff threads, in-thread continuations --
        # hold the same way on every platform. The shared ``PlatformConfig``
        # field is the canonical source, matching how the Discord adapter reads
        # it; env/extra stay supported for installs that configure per platform.
        # Thread replies are opt-in, per channel or install-wide, because they
        # change where replies land.
        self._reply_to_mode = _coerce_reply_to_mode(
            _fluxer_env("FLUXER_REPLY_TO_MODE")
            or extra.get("reply_to_mode")
            or getattr(config, "reply_to_mode", None)
        )
        self._thread_replies_enabled = _coerce_bool(
            _fluxer_env("FLUXER_THREAD_REPLIES", extra.get("thread_replies")),
            False,
        )
        self._thread_reply_channels = _split_ids(
            _fluxer_env("FLUXER_THREAD_REPLY_CHANNELS") or extra.get("thread_reply_channels")
        )
        # A message sent into a thread must be edited or deleted through that
        # thread, not the channel the caller named, so remember where each sent
        # message actually landed (bounded like the other per-adapter caches).
        self._message_destinations: OrderedDict[str, str] = OrderedDict()
        self._message_destinations_max = 5000
        # Threads already opened for a given (chat, anchor message) pair, so one
        # turn's progress bubble and final reply share a single thread instead of
        # each creating its own.
        self._threads_by_anchor: OrderedDict[tuple[str, str], str] = OrderedDict()
        self._threads_by_anchor_max = 1000
        # One creator per anchor. Sends for a turn interleave, so without this two
        # of them both miss the map above, both POST the same thread, and the
        # loser's turn is demoted to the parent channel by the failure path.
        self._thread_create_locks: Dict[tuple[str, str], asyncio.Lock] = {}
        # Anchors with a send inside, or queued on, their creation lock. The lock
        # map is bounded, and a bound that evicted a lock still in use would hand
        # the next send a second lock for the same anchor -- two creators again,
        # which is the duplicate this lock exists to prevent -- so eviction skips
        # the anchors listed here.
        self._thread_create_lock_users: Dict[tuple[str, str], int] = {}
        # Parent channel of every thread this adapter created, so a thread
        # inherits its parent's allowlist status and can seed mention memory.
        # Only a thread the adapter owns lands here: a thread a human created
        # must stay independent of the parent channel's session and access rules.
        self._thread_parents: Dict[str, str] = {}
        # Whether each known thread is one THIS adapter opened. A bot-owned thread
        # inherits its parent channel's free-response status (it is where the bot
        # was addressed); a thread a human created must not, or the whole channel
        # becomes reachable through a thread the operator never allowlisted.
        self._thread_owners: Dict[str, bool] = {}
        # Thread metadata read back from the API, so the parent/owner of a thread
        # is resolved once per process rather than once per follow-up message.
        # Cleared per thread when the thread is forgotten.
        self._thread_info_cache: Dict[str, Dict[str, Any]] = {}
        # Auto-thread anchors whose thread this adapter has not created yet:
        # ``anchor message id`` -> ``parent channel id``. Fluxer gives a thread the
        # id of the message it was started from, so the session container of the
        # message that opens a thread IS that thread, and the reply, the typing
        # indicator and the approval prompt all have to be routed to it --
        # creating it on first use.
        self._threads_pending: Dict[str, str] = {}
        # The kind of a channel id learned from the API (``thread`` / ``channel``).
        # Fluxer's own MESSAGE_CREATE payload carries no ``channel_type`` and no
        # nested channel object, so the raw event cannot classify itself; without
        # this a follow-up typed inside a thread was read as a channel message and
        # keyed to a different session than the turn that opened the thread.
        self._channel_kinds: OrderedDict[str, str] = OrderedDict()
        # Anchor message ids that must NOT open a thread of their own: a text
        # command typed in a thread-enabled channel. The command addresses the turn
        # already running in that chat, so threading its own message creates a
        # container with no session behind it (live artefact: an active thread
        # named ``/stop``).
        self._no_thread_anchors: set = set()
        # Containers delivery found deleted: the container to use instead. A
        # deleted thread's id never comes back, so the redirect holds for the rest
        # of the process and every later send for that turn lands in the parent
        # instead of failing against the same missing thread.
        self._dead_containers: Dict[str, str] = {}
        # Routing belongs to the executing event, not the channel's newest
        # message or the typing lifetime. Child tasks inherit the context;
        # queued turns rebind it when their processing actually begins.
        self._turn_context: ContextVar[Optional[tuple[Any, str]]] = ContextVar("fluxer_turn", default=None)
        self._turn_anchor: Dict[tuple[str, str], tuple] = {}
        self._turn_anchor_ttl = 1800.0
        # Channels where the automatically chosen thread turned out to be gone.
        # Auto-threading is skipped for the rest of the turn so the fallback to
        # the parent channel sticks instead of re-creating a thread per send.
        self._turn_thread_disabled: set = set()
        self._auto_free_response_home_guild = _coerce_bool(
            _fluxer_env("FLUXER_AUTO_FREE_RESPONSE_HOME_GUILD", extra.get("auto_free_response_home_guild")),
            False,
        )
        self._home_guild_ids = _split_ids(
            _fluxer_env("FLUXER_HOME_GUILDS")
            or _fluxer_env("FLUXER_HOME_GUILD_ID")
            or extra.get("home_guild_ids")
            or extra.get("home_guild_id")
        )
        self._allowed_channel_ids = _split_ids(
            _fluxer_env("FLUXER_ALLOWED_CHANNELS") or extra.get("allowed_channels")
        )
        self._mention_patterns = tuple(
            part.strip()
            for part in _split_ids(_fluxer_env("FLUXER_MENTION_PATTERNS") or extra.get("mention_patterns"))
            if part.strip()
        )
        self._register_native_commands_on_connect = _coerce_bool(
            _fluxer_env("FLUXER_REGISTER_NATIVE_COMMANDS", extra.get("register_native_commands")),
            False,
        )
        self._application_id = str(_fluxer_env("FLUXER_APPLICATION_ID") or extra.get("application_id") or "").strip()
        self._native_command_guild_ids = _split_ids(
            _fluxer_env("FLUXER_NATIVE_COMMAND_GUILDS") or extra.get("native_command_guilds")
        )
        self._mentioned_threads: OrderedDict[str, None] = OrderedDict()
        self._mentioned_threads_max = 5000
        self._home_channel_ids: set[str] = set()
        self._known_channel_ids: set[str] = set()
        home_channel = getattr(config, "home_channel", None)
        if home_channel and getattr(home_channel, "chat_id", None):
            self._home_channel_ids.add(str(home_channel.chat_id))
            self._known_channel_ids.add(str(home_channel.chat_id))
        home_extra = extra.get("home_channel")
        if isinstance(home_extra, dict) and home_extra.get("chat_id"):
            self._home_channel_ids.add(str(home_extra["chat_id"]))
            self._known_channel_ids.add(str(home_extra["chat_id"]))
        home_env = _fluxer_env("FLUXER_HOME_CHANNEL", "").strip()
        if home_env:
            self._home_channel_ids.add(home_env)
            self._known_channel_ids.add(home_env)
        self._last_disconnect_at: Optional[datetime] = None
        self._ws = None
        self._listener_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        self._last_seq: Optional[int] = None
        self._last_heartbeat_sent_at: Optional[float] = None
        self._last_heartbeat_ack_at: Optional[float] = None
        self._awaiting_heartbeat_ack = False
        self._closing = False
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self._last_inbound_by_chat: OrderedDict[str, str] = OrderedDict()
        # Text of the most recent inbound messages, keyed by message id: the thread
        # a turn opens is named from the message that triggered it.
        self._inbound_text_by_message: OrderedDict[str, str] = OrderedDict()
        # Author of those same messages, keyed the same way. A channel can hold one
        # session per participant, so naming a thread from the *triggering* user's
        # session is what keeps a busy channel's threads correctly named; core's
        # lookup refuses to guess between several live users.
        self._inbound_author_by_message: OrderedDict[str, str] = OrderedDict()
        self._inbound_text_by_message_max = 2000
        # ``chat_id`` -> ``(thread_id, initial_name)`` for the thread this adapter
        # most recently opened underneath that chat: the /title command renames it
        # and the session-title poller keeps its guard name in step.
        self._auto_thread_by_chat: OrderedDict[str, tuple[str, str]] = OrderedDict()
        # The same threads keyed by ``(chat_id, user_id)``. A channel can hold one
        # thread per participant, and the per-chat entry above only says which was
        # opened last, so /title needs this to rename the session's own thread
        # rather than whoever happened to post most recently.
        self._auto_thread_by_session: OrderedDict[tuple[str, str], tuple[str, str]] = OrderedDict()
        # Live lookups for a thread that is still waiting for its session title.
        self._title_tasks: Dict[str, asyncio.Task] = {}
        # Hermes' session store, opened lazily and read-only (see
        # ``_open_session_store``). Opened on first use rather than in ``connect``
        # because the store may not exist yet when the gateway starts.
        self._session_db: Optional[Any] = None
        self._title_home = _current_session_home()
        self._session_readers: Dict[Path, Any] = {}
        self._session_retry_after: Dict[Path, float] = {}
        self._session_db_lock = threading.Lock()
        self._session_db_off = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._typing_tasks: Dict[tuple[str, str], asyncio.Task] = {}
        self._pending_exec_approvals: Dict[str, Dict[str, Any]] = {}
        self._pending_component_actions: Dict[str, Dict[str, Any]] = {}
        self._pending_reaction_actions: Dict[str, Dict[str, Any]] = {}
        self._pending_voice_joins: Dict[str, Dict[str, Any]] = {}
        self._last_voice_server_update: Optional[Dict[str, Any]] = None
        self._voice_server_update_handler: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = None
        self._voice_state_update_handler: Optional[Callable[[Dict[str, Any]], Any]] = None
        self._voice_supervisor = FluxerVoiceSupervisorProcess(extra=extra)
        self._gateway_ready_event = asyncio.Event()

    def _mark_connected(self) -> None:
        if self._gateway_state_updates_enabled:
            super()._mark_connected()

    def _mark_disconnected(self) -> None:
        if self._gateway_state_updates_enabled:
            super()._mark_disconnected()

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if not self.bot_token:
            self._set_fatal_error(
                "missing_config",
                "Fluxer requires FLUXER_BOT_TOKEN; FLUXER_BASE_URL is optional and defaults to https://api.fluxer.app/v1",
                retryable=False,
            )
            return False
        try:
            self._closing = False
            try:
                # The /title hook runs on the gateway loop; keep it for the sync
                # hook callback to hand work back to.
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = None
            _LIVE_ADAPTERS.add(self)
            self._mark_connected()
            await self._connect_gateway_once()
            if not is_reconnect:
                await self._maybe_register_native_commands()
            self._voice_supervisor.start()
            return True
        except Exception as exc:
            await self._voice_supervisor.stop()
            self._mark_disconnected()
            logger.warning("Fluxer connect failed: %s", exc)
            self._set_fatal_error("connect_failed", f"Fluxer connect failed: {exc}", retryable=True)
            return False

    async def disconnect(self) -> None:
        await self._voice_supervisor.stop()
        self._closing = True
        self._running = False
        _LIVE_ADAPTERS.discard(self)
        self._close_session_store()
        for task in list(self._title_tasks.values()):
            if task and not task.done():
                task.cancel()
        self._title_tasks.clear()
        for task in (self._heartbeat_task, self._listener_task, self._reconnect_task):
            if task and not task.done():
                task.cancel()
        for task in list(self._typing_tasks.values()):
            if task and not task.done():
                task.cancel()
        self._typing_tasks.clear()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._reconnect_task = None
        self._gateway_ready_event.clear()
        self._mark_disconnected()

    async def _connect_gateway_once(self) -> None:
        if not self.gateway_url:
            info = await self._request("GET", "/gateway/bot")
            self.gateway_url = _strip_slash(str(info.get("url") or ""))
        if not self.gateway_url:
            raise RuntimeError("Fluxer gateway URL missing from /gateway/bot")

        import websockets

        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
        current = asyncio.current_task()
        if self._listener_task and self._listener_task is not current and not self._listener_task.done():
            self._listener_task.cancel()
        self._ws = None
        self._heartbeat_task = None
        self._listener_task = None
        self._awaiting_heartbeat_ack = False
        self._last_heartbeat_sent_at = None
        self._last_heartbeat_ack_at = None
        self._gateway_ready_event.clear()

        sep = "&" if "?" in self.gateway_url else "?"
        ws_url = f"{self.gateway_url}{sep}v={_GATEWAY_VERSION}&encoding=json"
        reconnect_cutoff = self._last_disconnect_at or (
            datetime.now(tz=timezone.utc) - timedelta(seconds=self._backlog_bootstrap_seconds)
        )
        self._ws = await websockets.connect(ws_url, open_timeout=15, close_timeout=5, max_size=None)
        self._listener_task = asyncio.create_task(self._listen_loop(), name="fluxer-listen")
        logger.info("Fluxer gateway websocket connected")
        await self._recover_backlog(cutoff=reconnect_cutoff)

    def _schedule_reconnect(self, reason: str) -> None:
        if self._closing:
            return
        if self._reconnect_task and not self._reconnect_task.done():
            return
        logger.warning("Fluxer gateway disconnected (%s); scheduling reconnect", reason)
        self._reconnect_task = asyncio.create_task(self._reconnect_loop(reason), name="fluxer-reconnect")

    async def _reconnect_loop(self, reason: str) -> None:
        delay = _RECONNECT_BASE_DELAY
        while not self._closing:
            try:
                await asyncio.sleep(delay)
                await self._connect_gateway_once()
                self._mark_connected()
                self._voice_supervisor.start()
                logger.info("Fluxer gateway reconnected after %s", reason)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Fluxer reconnect failed: %s; retrying in %.1fs", exc, delay)
                delay = min(delay * 2, _RECONNECT_MAX_DELAY)

    def _sanitize_outbound_mentions(self, content: str) -> str:
        """Fail closed for global/role pings in model-generated Fluxer output.

        Fluxer message readback exposes mention fields on current deployments,
        but the hosted API is still young. We send an ``allowed_mentions``
        object for compatible servers and also neutralize dangerous ping syntax
        in the content itself so unsupported deployments cannot accidentally
        @everyone a channel from LLM output or echoed user text.
        """
        sanitized = content
        if not self._allow_mention_everyone:
            sanitized = _MENTION_EVERYONE_RE.sub(lambda m: "@\u200b" + m.group(1), sanitized)
        if not self._allow_mention_roles:
            sanitized = _MENTION_ROLE_RE.sub(lambda m: m.group(0).replace("<@&", "<@\u200b&", 1), sanitized)
        if not self._allow_mention_users:
            sanitized = _MENTION_USER_RE.sub(lambda m: m.group(0).replace("<@", "<@\u200b", 1), sanitized)
        return sanitized

    def _allowed_mentions_payload(self) -> Dict[str, Any]:
        parse: List[str] = []
        if self._allow_mention_everyone:
            parse.append("everyone")
        if self._allow_mention_roles:
            parse.append("roles")
        if self._allow_mention_users:
            parse.append("users")
        return {"parse": parse, "replied_user": self._allow_mention_replied_user}

    def _outbound_message_payload(self, content: str, **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "content": self._sanitize_outbound_mentions(content),
            "allowed_mentions": self._allowed_mentions_payload(),
        }
        payload.update(extra)
        return payload

    async def _add_reaction(self, chat_id: str, message_id: str, emoji: str) -> None:
        await self._request(
            "PUT",
            f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}/reactions/{_quote_id(emoji)}/@me",
        )

    async def add_reaction(
        self,
        chat_id: str,
        emoji: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add an explicit agent-requested reaction to a Fluxer message."""
        if not str(emoji).strip():
            return {"success": False, "error": "emoji is required"}
        target = message_id or self._last_inbound_by_chat.get(str(chat_id))
        if not target:
            return {
                "success": False,
                "error": "no message to react to — pass message_id",
            }
        try:
            await self._add_reaction(chat_id, target, emoji)
        except Exception as exc:
            logger.debug("Fluxer agent reaction failed (%s)", type(exc).__name__)
            return {
                "success": False,
                "error": "reaction failed (see gateway debug log)",
            }
        return {"success": True, "message_id": target}

    async def remove_reaction(
        self,
        chat_id: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Remove every reaction this bot owns on a Fluxer message."""
        target = message_id or self._last_inbound_by_chat.get(str(chat_id))
        if not target:
            return {
                "success": False,
                "error": "no message to unreact — pass message_id",
            }
        try:
            message = await self._request(
                "GET",
                f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(target)}",
            )
            owned_emojis: List[str] = []
            for reaction in message.get("reactions") or []:
                if not isinstance(reaction, dict) or reaction.get("me") is not True:
                    continue
                emoji = reaction.get("emoji") or {}
                if not isinstance(emoji, dict):
                    continue
                name = str(emoji.get("name") or "")
                emoji_id = str(emoji.get("id") or "")
                if name:
                    owned_emojis.append(f"{name}:{emoji_id}" if emoji_id else name)
            for emoji in owned_emojis:
                await self._request(
                    "DELETE",
                    f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(target)}/reactions/{_quote_id(emoji)}/@me",
                )
        except Exception as exc:
            logger.debug("Fluxer agent unreact failed (%s)", type(exc).__name__)
            return {
                "success": False,
                "error": "unreact failed (see gateway debug log)",
            }
        return {
            "success": True,
            "message_id": target,
            "removed": len(owned_emojis),
        }

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            # Resolve the destination before anything is sent: an explicit
            # thread_id wins, then opt-in thread replies may start a thread from
            # the triggering message, then the channel itself.
            target_chat_id, payload, auto_thread_id = await self._resolve_outbound_target(chat_id, reply_to, metadata)
            self._known_channel_ids.add(str(target_chat_id))

            # Mention neutralization inserts zero-width characters. Apply it
            # before measuring so the final wire payload cannot cross Fluxer's
            # 4,000 UTF-16-unit limit after sanitization.
            formatted = self._sanitize_outbound_mentions(self.format_message(content))
            chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH, len_fn=utf16_len)
            message_ids: List[str] = []
            responses: List[Dict[str, Any]] = []

            for index, chunk in enumerate(chunks):
                chunk_payload = self._outbound_message_payload(chunk, **payload)
                if index > 0 and self._reply_to_mode != "all":
                    # "first" (default) and "off" keep the reference on the first
                    # split chunk only; repeating it on every continuation
                    # creates noisy threads and can make partial retries nastier.
                    # "all" mirrors the Discord adapter's every-chunk reference.
                    chunk_payload.pop("message_reference", None)

                try:
                    data = await self._post_chunk(
                        chunk_payload, payload, target_chat_id, index,
                        chat_id, reply_to, metadata, auto_thread_id,
                    )
                except _ThreadGone as gone:
                    # A thread this adapter chose can be deleted between the
                    # progress bubble and the reply. Abandon it and post to the
                    # parent channel rather than losing the message. For a
                    # prospective thread the caller's chat_id IS the thread, so
                    # the parent channel is the one recorded when it was opened.
                    logger.info(
                        "Fluxer thread %s is gone; falling back to channel %s",
                        gone.thread_id, gone.parent_chat,
                    )
                    self._forget_thread(gone.thread_id)
                    self._disable_turn_thread(chat_id, reply_to, metadata)
                    self._remember_dead_container(gone.thread_id, gone.parent_chat)
                    self._remember_dead_container(chat_id, gone.parent_chat)
                    auto_thread_id = None
                    target_chat_id = str(gone.parent_chat)
                    self._known_channel_ids.add(target_chat_id)
                    if reply_to and self._reply_to_mode != "off":
                        # Restore the reference in the shared payload as well: the
                        # remaining chunks of a split reply are rebuilt from it, so
                        # patching only this chunk would leave every later one
                        # unlinked after a fallback. The mode still decides where
                        # the reference lands.
                        payload["message_reference"] = {"message_id": str(reply_to)}
                        chunk_payload["message_reference"] = {"message_id": str(reply_to)}
                        if index > 0 and self._reply_to_mode != "all":
                            chunk_payload.pop("message_reference", None)
                    data = await self._post_chunk(
                        chunk_payload, payload, target_chat_id, index,
                        chat_id, reply_to, metadata, None,
                    )
                except Exception as exc:
                    # Any other failure: if earlier chunks of this split already
                    # landed, the failure is a PARTIAL delivery — the host must
                    # never re-send the visible head, only the missing tail.
                    if message_ids:
                        logger.warning(
                            "Fluxer split send stopped after %d/%d chunks: %s",
                            len(message_ids), len(chunks), exc,
                        )
                        return self._partial_split_result(exc, chunks, message_ids, responses)
                    raise
                responses.append(data)
                if data.get("id"):
                    message_id = str(data["id"])
                    message_ids.append(message_id)
                    # Edits/deletes must target the thread, not the caller's channel.
                    self._remember_message_destination(message_id, target_chat_id)
                    verified = await self._verify_delivery(
                        target_chat_id,
                        message_id,
                        expected_content=chunk_payload["content"],
                    )
                    if verified is not None:
                        responses[-1] = {**data, "delivery_verified": True, "delivery_readback": verified}
                    else:
                        responses[-1] = {**data, "delivery_verified": False}

            # Split-payload contract: ``message_id`` is the LAST chunk (later edits
            # target the newest bubble) and the earlier ones ride in
            # ``continuation_message_ids`` in send order.
            return SendResult(
                success=True,
                message_id=message_ids[-1] if message_ids else None,
                raw_response={"message_ids": message_ids, "responses": responses},
                continuation_message_ids=tuple(message_ids[:-1]),
            )
        except Exception as exc:
            logger.warning("Fluxer send failed: %s", exc)
            return _failed_send_result(exc)

    async def _post_chunk(
        self,
        chunk_payload: Dict[str, Any],
        shared_payload: Dict[str, Any],
        target_chat_id: str,
        index: int,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
        auto_thread_id: Optional[str],
    ) -> Dict[str, Any]:
        """POST one split chunk, absorbing the two recoverable refusals.

        - A refused reply reference (``400 CANNOT_REPLY_TO_SYSTEM_MESSAGE``) names
          the reference, never the content, so the identical retry can only fail
          again: drop the reference — from the SHARED payload too, so no later
          chunk re-sends it — and send on.
        - A missing container (404 / unknown thread) on a thread this adapter
          chose raises :class:`_ThreadGone` so the caller falls back to the parent
          channel. Any other failure propagates unchanged: it is a real delivery
          problem, not a redirect.
        """
        try:
            return await self._request(
                "POST",
                f"/channels/{_quote_id(target_chat_id)}/messages",
                json=chunk_payload,
            )
        except Exception as exc:
            if _is_reply_reference_rejected_error(exc) and chunk_payload.get("message_reference"):
                logger.info(
                    "Fluxer refused the reply reference (%s); sending without it",
                    type(exc).__name__,
                )
                shared_payload.pop("message_reference", None)
                chunk_payload.pop("message_reference", None)
                return await self._request(
                    "POST",
                    f"/channels/{_quote_id(target_chat_id)}/messages",
                    json=chunk_payload,
                )
            if auto_thread_id and _is_missing_channel_error(exc):
                parent_chat = (
                    self._thread_parents.get(str(chat_id))
                    or self._threads_pending.get(str(chat_id))
                    or str(chat_id)
                )
                raise _ThreadGone(auto_thread_id, str(parent_chat), exc) from exc
            raise

    def _partial_split_result(
        self,
        exc: BaseException,
        chunks: List[str],
        message_ids: List[str],
        responses: List[Dict[str, Any]],
    ) -> SendResult:
        """A failure after some split chunks landed, marked ``partial_overflow``.

        The host's retry ladder and stream consumer both read
        ``raw_response["partial_overflow"]`` to avoid re-sending the visible head,
        so the delivered ids and the prefix actually on screen travel with the
        failure. ``message_id`` stays the LAST delivered chunk.
        """
        retryable, error_kind = _classify_send_failure(exc)
        delivered_prefix = "".join(_chunk_without_indicator(c) for c in chunks[: len(message_ids)])
        return SendResult(
            success=False,
            message_id=message_ids[-1],
            error=_error_text_for(exc),
            retryable=retryable,
            error_kind=error_kind,
            continuation_message_ids=tuple(message_ids[:-1]),
            raw_response={
                "partial_overflow": True,
                "delivered_chunks": len(message_ids),
                "total_chunks": len(chunks),
                "last_message_id": message_ids[-1],
                "delivered_prefix": delivered_prefix,
                "continuation_message_ids": tuple(message_ids[:-1]),
                "message_ids": list(message_ids),
                "responses": responses,
            },
        )

    def _request_id_for_command(self, session_key: str, command: str) -> Optional[str]:
        """The gateway request id this approval prompt belongs to, or ``None``.

        ``send_exec_approval`` receives the runner's redacted command but NOT the
        gateway ``request_id`` (the host's ``ExecApprovalPrompt`` carries no such
        field). The host's public ``list_gateway_approvals(session_key)`` returns
        snapshots of the session's unresolved approvals, each carrying its
        ``request_id``. Match on the normalized command, but ONLY when exactly one
        pending request matches: a unique match is safe, while an ambiguous one (two
        identical commands queued) or an unavailable reader yields ``None``, and the
        caller then falls back to the host's FIFO resolution rather than ever
        resolving the wrong request.
        """
        normalized = _normalize_approval_command(command)
        try:
            from tools.approval import list_gateway_approvals

            pending = list_gateway_approvals(session_key) or []
        except Exception:
            return None
        matches = [
            entry
            for entry in pending
            if isinstance(entry, dict)
            and entry.get("request_id")
            and _normalize_approval_command(entry.get("command")) == normalized
        ]
        if len(matches) == 1:
            return str(matches[0]["request_id"])
        if len(matches) > 1:
            logger.warning(
                "Fluxer approval prompt has %d pending requests with an identical command; "
                "not binding a request id (resolution stays FIFO)",
                len(matches),
            )
        return None

    async def _send_exec_approval_prompt(self, prompt: Any) -> SendResult:
        """Render an exec-approval card natively (the host's contract override).

        The host's ``BasePlatformAdapter.send_exec_approval`` builds an
        ``ExecApprovalPrompt`` and calls this hook, which is what makes
        ``supports_exec_approval_buttons()`` true for this adapter. Taking the prompt
        object means the choice set is derived from ``prompt.actions`` — the host's
        own allow_permanent / allow_session / smart_denied policy — instead of the
        adapter inventing a full four-button row.
        """
        actions = list(getattr(prompt, "actions", None) or [])
        if not actions:
            actions = [("approve once", "once", "primary"), ("deny", "deny", "danger")]
        return await self._post_approval_card(
            chat_id=prompt.chat_id,
            session_key=prompt.session_key,
            command=prompt.command,
            description=prompt.description,
            metadata=getattr(prompt, "metadata", None),
            reaction_specs=tuple(
                (_EA_ACTION_EMOJI.get(choice, "•"), choice, label)
                for label, choice, _style in actions
            ),
        )

    async def send_exec_approval(
        self,
        chat_id: str,
        command: str,
        session_key: str,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        allow_permanent: bool = True,
        allow_session: bool = True,
        smart_denied: bool = False,
    ) -> SendResult:
        """Send a dangerous-command approval prompt with clickable reactions.

        Signature matches ``BasePlatformAdapter.send_exec_approval`` so the runner's
        ``**flags`` (``allow_permanent`` / ``allow_session`` / ``smart_denied``) are
        honoured instead of raising ``TypeError``. The offered choices follow the
        host's own policy, so a smart deny or a permanent-incapable prompt cannot be
        escalated to a permanent approval from this surface.
        """
        specs = _exec_approval_choices(
            allow_permanent=allow_permanent,
            allow_session=allow_session,
            smart_denied=smart_denied,
        )
        return await self._post_approval_card(
            chat_id=chat_id,
            session_key=session_key,
            command=command,
            description=description or "dangerous command",
            metadata=metadata,
            reaction_specs=specs,
        )

    async def _post_approval_card(
        self,
        *,
        chat_id: str,
        session_key: str,
        command: str,
        description: str,
        metadata: Optional[Dict[str, Any]],
        reaction_specs: tuple[tuple[str, str, str], ...],
    ) -> SendResult:
        """Post an exec-approval card and register its components + reactions.

        The card is bound to the gateway ``request_id`` when exactly one pending
        request matches its command, so a tap resolves THAT request and never another
        via FIFO. When the id cannot be determined safely it is left unbound and the
        host's FIFO resolution applies — never a wrong-request approval.
        """
        target_id = await self._resolve_thread_container(chat_id, metadata, session_key)
        cmd_display = command if len(command) <= 3200 else command[:3197] + "..."
        reaction_hint = " • ".join(f"{emoji} {label}" for emoji, _choice, label in reaction_specs)
        content = (
            "**Command approval required**\n"
            f"```\n{cmd_display}\n```\n"
            f"Reason: {description}\n\n"
            f"React: {reaction_hint}\n"
            "Reply with `/approve once`, `/approve session`, `/approve always`, or `/deny`."
        )
        try:
            self._known_channel_ids.add(target_id)
            buttons, component_actions = _fluxer_action_buttons(
                prefix="fluxer_exec",
                specs=reaction_specs,
                danger_choice="deny",
            )
            data, components_enabled = await self._post_message_with_optional_components(
                target_id,
                content,
                _fluxer_component_row(buttons),
            )
            message_id = str(data.get("id") or "")
            if not message_id:
                return SendResult(success=False, error="Fluxer approval message missing id", retryable=True)

            request_id = self._request_id_for_command(session_key, command)
            pending = {
                "session_key": session_key,
                "channel_id": target_id,
                "created_at": time.time(),
                "content": content,
                "resolved": False,
                "command": command,
                "request_id": request_id,
            }
            self._pending_exec_approvals[message_id] = pending
            self._prune_pending_cards()
            if components_enabled:
                self._register_component_actions(
                    component_actions,
                    kind="exec_approval",
                    message_id=message_id,
                    session_key=session_key,
                    channel_id=target_id,
                    request_id=request_id,
                )
            for emoji, choice, _label in reaction_specs:
                self._pending_reaction_actions[f"{message_id}:{_normalize_reaction_emoji(emoji)}"] = {
                    "kind": "exec_approval",
                    "message_id": message_id,
                    "session_key": session_key,
                    "channel_id": target_id,
                    "choice": choice,
                    "request_id": request_id,
                    "created_at": time.time(),
                }
                await self._add_reaction(target_id, message_id, emoji)
            return SendResult(success=True, message_id=message_id, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer exec approval prompt failed: %s", exc)
            return _failed_send_result(exc)

    async def send_slash_confirm(
        self,
        chat_id: str,
        title: str,
        message: str,
        session_key: str,
        confirm_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a slash-command confirmation prompt with clickable reactions."""
        target_id = await self._resolve_thread_container(chat_id, metadata, session_key)
        content_message = (message or "").rstrip()
        if not re.search(r"/(?:approve|always|cancel)", content_message, flags=re.IGNORECASE):
            content_message = f"{content_message}\n\n_Text fallback: reply `/approve`, `/always`, or `/cancel`._"
        reaction_hint = " • ".join(f"{emoji} {label}" for emoji, _choice, label in _SLASH_CONFIRM_REACTIONS)
        content_message = f"{content_message}\n\nReact: {reaction_hint}"
        content = f"**{title}**\n{content_message}"
        try:
            buttons, component_actions = _fluxer_action_buttons(
                prefix="fluxer_confirm",
                specs=_SLASH_CONFIRM_REACTIONS,
                danger_choice="cancel",
            )
            data, components_enabled = await self._post_message_with_optional_components(
                target_id,
                content,
                _fluxer_component_row(buttons),
            )
            message_id = str(data.get("id") or "")
            if not message_id:
                return SendResult(success=False, error="Fluxer slash confirm message missing id", retryable=True)
            if components_enabled:
                self._register_component_actions(
                    component_actions,
                    kind="slash_confirm",
                    message_id=message_id,
                    session_key=session_key,
                    confirm_id=confirm_id,
                    channel_id=target_id,
                )
            for emoji, choice, _label in _SLASH_CONFIRM_REACTIONS:
                self._pending_reaction_actions[f"{message_id}:{_normalize_reaction_emoji(emoji)}"] = {
                    "kind": "slash_confirm",
                    "message_id": message_id,
                    "session_key": session_key,
                    "confirm_id": confirm_id,
                    "channel_id": target_id,
                    "choice": choice,
                    "created_at": time.time(),
                }
                await self._add_reaction(target_id, message_id, emoji)
            self._prune_pending_cards()
            return SendResult(success=True, message_id=message_id, raw_response=data)
        except Exception as exc:
            logger.warning("Fluxer slash confirm prompt failed: %s", exc)
            return _failed_send_result(exc)

    def _interaction_user_allowed(self, user_id: str) -> bool:
        if not user_id:
            return False
        if self.bot_user_id and user_id == str(self.bot_user_id):
            return False
        if self._allow_all_users:
            return True
        return user_id in self._allowed_user_ids

    async def edit_message(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        *,
        finalize: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Edit a previously sent Fluxer message.

        Gateway tool-progress bubbles are only enabled for adapters that
        override ``BasePlatformAdapter.edit_message``. Fluxer exposes a PATCH
        message endpoint, so implement it to unlock visible tool
        calling/progress in chat instead of only final responses and status
        callbacks.
        """
        try:
            # A message this adapter sent may live in a thread; route the edit
            # there rather than the caller's channel.
            target_chat_id = self._destination_for_message(message_id, chat_id)
            # Sanitize before measuring: neutralized mentions are one UTF-16
            # unit longer than their source syntax.
            formatted = self._sanitize_outbound_mentions(self.format_message(content))
            if utf16_len(formatted) > MAX_MESSAGE_LENGTH:
                if finalize:
                    # The FINAL edit holds the whole answer, so it must be split
                    # across the preview plus continuations rather than truncated.
                    # Truncating here delivered only the first chunk and reported
                    # success, silently hiding the rest of the reply.
                    return await self._edit_overflow_split(
                        target_chat_id, message_id, formatted
                    )
                # Mid-stream: truncate instead of splitting. Splitting creates
                # continuation messages whose ids become the new edit target, and
                # the next token chunk re-edits the full text into the continuation,
                # triggering another split -> infinite duplication loop.
                formatted = self.truncate_message(
                    formatted,
                    MAX_MESSAGE_LENGTH,
                    len_fn=utf16_len,
                )[0]
            data = await self._request(
                "PATCH",
                f"/channels/{_quote_id(target_chat_id)}/messages/{_quote_id(message_id)}",
                json=self._outbound_message_payload(formatted),
            )
            # Intermediate streaming edits are intentionally not content-checked:
            # a later PATCH can land before this edit's GET read-back, making the
            # read-back newer than ``formatted`` and producing a false mismatch.
            # The final edit is stable and still gets exact visible-content
            # verification; intermediate edits retain ID/author verification.
            verified = await self._verify_delivery(
                target_chat_id,
                str(data.get("id") or message_id),
                expected_content=(self._sanitize_outbound_mentions(formatted) if finalize else None),
            )
            raw_response = dict(data)
            raw_response["delivery_verified"] = verified is not None
            if verified is not None:
                raw_response["delivery_readback"] = verified
            return SendResult(
                success=True,
                message_id=str(data.get("id") or message_id),
                raw_response=raw_response,
            )
        except Exception as exc:
            logger.warning("Fluxer edit failed for message %s: %s", message_id, exc)
            return _failed_send_result(exc)

    async def _edit_overflow_split(
        self, target_chat_id: str, message_id: str, formatted: str
    ) -> SendResult:
        """Deliver an oversized FINAL edit: chunk 1 edits ``message_id``, the rest
        are sent as continuations. Returns ``message_id=<last-id>`` +
        ``continuation_message_ids``.

        A continuation failure returns ``success=False`` with
        ``raw_response["partial_overflow"]`` / ``delivered_prefix`` /
        ``last_message_id`` so the host delivers only the missing tail instead of
        treating the truncated preview as the whole answer.
        """
        chunks = self.truncate_message(formatted, MAX_MESSAGE_LENGTH, len_fn=utf16_len)
        if len(chunks) <= 1:
            chunks = [formatted]  # defensive: a single chunk just edits normally
        try:
            data = await self._request(
                "PATCH",
                f"/channels/{_quote_id(target_chat_id)}/messages/{_quote_id(message_id)}",
                json=self._outbound_message_payload(chunks[0]),
            )
        except Exception as exc:
            logger.error("Fluxer overflow split: first-chunk edit failed: %s", exc)
            return _failed_send_result(exc)
        first_id = str(data.get("id") or message_id)
        delivered_ids = [first_id]
        delivered_chunks = [chunks[0]]
        for chunk in chunks[1:]:
            try:
                data = await self._request(
                    "POST",
                    f"/channels/{_quote_id(target_chat_id)}/messages",
                    json=self._outbound_message_payload(chunk),
                )
            except Exception as exc:
                logger.warning(
                    "Fluxer overflow split: stopped at %d/%d chunks delivered: %s",
                    len(delivered_ids), len(chunks), exc,
                )
                retryable, error_kind = _classify_send_failure(exc)
                delivered_prefix = "".join(
                    _chunk_without_indicator(c) for c in delivered_chunks
                )
                last_id = delivered_ids[-1]
                return SendResult(
                    success=False,
                    message_id=last_id,
                    error=_error_text_for(exc),
                    retryable=retryable,
                    error_kind=error_kind,
                    continuation_message_ids=tuple(delivered_ids[:-1]),
                    raw_response={
                        "partial_overflow": True,
                        "delivered_chunks": len(delivered_ids),
                        "total_chunks": len(chunks),
                        "last_message_id": last_id,
                        "delivered_prefix": delivered_prefix,
                        "continuation_message_ids": tuple(delivered_ids[:-1]),
                    },
                )
            new_id = str(data.get("id") or "")
            if new_id:
                self._remember_message_destination(new_id, target_chat_id)
                delivered_ids.append(new_id)
            delivered_chunks.append(chunk)
        last_id = delivered_ids[-1]
        return SendResult(
            success=True,
            message_id=last_id,
            continuation_message_ids=tuple(delivered_ids[:-1]),
            raw_response={"message_ids": delivered_ids},
        )

    async def delete_message(self, chat_id: str, message_id: str) -> bool:
        """Delete a Fluxer message when gateway cleanup/ephemeral flows ask."""
        try:
            # Route through the thread when that is where the message landed.
            target_chat_id = self._destination_for_message(message_id, chat_id)
            await self._request(
                "DELETE", f"/channels/{_quote_id(target_chat_id)}/messages/{_quote_id(message_id)}"
            )
            return True
        except Exception as exc:
            logger.debug("Fluxer delete failed for message %s: %s", message_id, exc)
            return False

    async def list_pinned_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Return pinned messages for a Fluxer channel.

        Fluxer's pin routes are exposed at:
        ``/channels/{channel_id}/messages/pins``. The hosted API currently
        returns ``{"items": [...], "has_more": bool}``, while some deployments
        may return a bare list, so accept both.
        """
        payload = await self._request("GET", f"/channels/{_quote_id(chat_id)}/messages/pins")
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("messages") or []
            return list(items) if isinstance(items, list) else []
        if isinstance(payload, list):
            return payload
        return []

    async def pin_message(self, chat_id: str, message_id: str) -> bool:
        """Pin a Fluxer message when the server supports message pins."""
        try:
            await self._request("PUT", f"/channels/{_quote_id(chat_id)}/pins/{_quote_id(message_id)}")
            return True
        except Exception as exc:
            logger.debug("Fluxer pin failed for message %s: %s", message_id, exc)
            return False

    async def unpin_message(self, chat_id: str, message_id: str) -> bool:
        """Unpin a Fluxer message when the server supports message pins."""
        try:
            await self._request("DELETE", f"/channels/{_quote_id(chat_id)}/pins/{_quote_id(message_id)}")
            return True
        except Exception as exc:
            logger.debug("Fluxer unpin failed for message %s: %s", message_id, exc)
            return False

    async def _maybe_register_native_commands(self) -> None:
        if not self._register_native_commands_on_connect:
            return
        application_id = self._application_id or self.bot_token.split(".", 1)[0]
        if not _looks_like_fluxer_id(application_id):
            logger.warning("Fluxer native command registration enabled but no valid application id is available")
            return
        guild_ids = sorted(self._native_command_guild_ids)
        try:
            if guild_ids:
                for guild_id in guild_ids:
                    await self.register_native_commands(application_id, guild_id=guild_id)
            else:
                await self.register_native_commands(application_id)
        except Exception as exc:
            logger.warning("Fluxer native command registration failed: %s", exc)

    def _native_command_payload(self, command: Any) -> Dict[str, Any]:
        description = str(getattr(command, "description", "Run Hermes command") or "Run Hermes command")[:100]
        payload: Dict[str, Any] = {
            "name": str(getattr(command, "name", "")).replace("_", "-")[:32],
            "description": description,
            "type": 1,
        }
        args_hint = str(getattr(command, "args_hint", "") or "").strip()
        if args_hint:
            payload["options"] = [{"type": 3, "name": "args", "description": args_hint[:100], "required": False}]
        return payload

    async def register_native_commands(
        self,
        application_id: str,
        *,
        guild_id: Optional[str] = None,
        commands: Optional[List[Any]] = None,
    ) -> Any:
        """Bulk upsert Fluxer application commands using Discord/JFA routes."""
        if commands is None:
            from hermes_cli.commands import COMMAND_REGISTRY

            commands = [cmd for cmd in COMMAND_REGISTRY if not getattr(cmd, "cli_only", False)]
        payloads = [self._native_command_payload(command) for command in commands if getattr(command, "name", None)]
        if guild_id:
            path = f"/applications/{_quote_id(application_id)}/guilds/{_quote_id(guild_id)}/commands"
        else:
            path = f"/applications/{_quote_id(application_id)}/commands"
        return await self._request("PUT", path, json=payloads)

    async def list_channels(self) -> List[Dict[str, Any]]:
        """Enumerate reachable Fluxer guild channels and active threads."""
        try:
            guilds_payload = await self._request("GET", "/users/@me/guilds", params={"limit": 200, "with_counts": True})
        except Exception as exc:
            logger.debug("Fluxer guild enumeration failed: %s", exc)
            return []
        guilds = guilds_payload if isinstance(guilds_payload, list) else (guilds_payload.get("guilds") or guilds_payload.get("items") or [])
        channels: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for guild in guilds:
            if not isinstance(guild, dict):
                continue
            guild_id = str(guild.get("id") or "")
            if not guild_id:
                continue
            guild_name = str(guild.get("name") or guild_id)
            try:
                guild_channels = await self._request("GET", f"/guilds/{_quote_id(guild_id)}/channels")
            except Exception as exc:
                logger.debug("Fluxer channel enumeration failed for guild %s: %s", guild_id, exc)
                guild_channels = []
            for item in guild_channels if isinstance(guild_channels, list) else []:
                if not isinstance(item, dict):
                    continue
                channel_id = str(item.get("id") or "")
                name = str(item.get("name") or "")
                channel_type = _directory_channel_type(item.get("type"))
                if not channel_id or not name or channel_type not in {"channel", "forum", "thread"} or channel_id in seen:
                    continue
                seen.add(channel_id)
                entry: Dict[str, Any] = {"id": channel_id, "name": name, "guild": guild_name, "guild_id": guild_id, "type": channel_type}
                if item.get("parent_id"):
                    entry["parent_id"] = str(item["parent_id"])
                channels.append(entry)
            try:
                threads_payload = await self._request("GET", f"/guilds/{_quote_id(guild_id)}/threads/active", warn_on_error=False)
            except Exception as exc:
                logger.debug("Fluxer active thread enumeration unavailable for guild %s: %s", guild_id, exc)
                threads_payload = {}
            threads = threads_payload.get("threads") if isinstance(threads_payload, dict) else []
            for thread in threads if isinstance(threads, list) else []:
                if not isinstance(thread, dict):
                    continue
                thread_id = str(thread.get("id") or "")
                name = str(thread.get("name") or "")
                if not thread_id or not name or thread_id in seen:
                    continue
                seen.add(thread_id)
                entry = {"id": thread_id, "name": name, "guild": guild_name, "guild_id": guild_id, "type": "thread"}
                if thread.get("parent_id"):
                    entry["parent_id"] = str(thread["parent_id"])
                channels.append(entry)
        for channel_id in self._known_channel_ids:
            if channel_id not in seen:
                channels.append({"id": channel_id, "name": channel_id, "type": "channel"})
        return channels

    def _thread_replies_enabled_for(self, chat_id: str) -> bool:
        """True when replies addressed to ``chat_id`` belong in a thread.

        ``FLUXER_THREAD_REPLIES`` enables thread replies install-wide; an
        explicit id in ``FLUXER_THREAD_REPLY_CHANNELS`` enables it for that
        channel only (the same channel-list convention as
        ``FLUXER_FREE_RESPONSE_CHANNELS``).
        """
        return bool(self._thread_replies_enabled or str(chat_id) in self._thread_reply_channels)

    def _remember_message_destination(self, message_id: Optional[str], chat_id: str) -> None:
        """Remember which channel or thread a sent message actually landed in."""
        if not message_id:
            return
        key = str(message_id)
        self._message_destinations[key] = str(chat_id)
        self._message_destinations.move_to_end(key)
        while len(self._message_destinations) > self._message_destinations_max:
            self._message_destinations.popitem(last=False)

    def _destination_for_message(self, message_id: str, fallback: str) -> str:
        """Channel/thread a previously sent message lives in.

        Falls back to the caller's ``chat_id`` for messages this adapter did not
        send (e.g. restored after a restart), so behaviour is unchanged there.
        """
        return self._message_destinations.get(str(message_id), str(fallback))

    def _forget_thread(self, thread_id: Optional[str]) -> None:
        """Drop every trace of ``thread_id`` so it is never chosen again.

        A thread can be deleted while a turn is running. The cached entry has to
        go with it, otherwise a retry would keep targeting the thread that just
        failed and reuse the same missing id forever.
        """
        tid = str(thread_id or "").strip()
        if not tid:
            return
        # Release only the turn anchors that pointed at the thread being
        # invalidated. A channel whose own thread has not been created yet has no
        # entry in ``_threads_by_anchor`` either, so reading "no entry" as "this
        # anchor pointed here" would drop an unrelated channel's anchor and let a
        # later message become that channel's reply anchor. Run before the
        # ``_threads_by_anchor`` cleanup, which is what the check reads.
        for turn_key, (anchor, _pinned_at) in list(self._turn_anchor.items()):
            if self._threads_by_anchor.get((turn_key[0], str(anchor))) == tid:
                self._turn_anchor.pop(turn_key, None)
        self._known_channel_ids.discard(tid)
        self._thread_parents.pop(tid, None)
        self._thread_owners.pop(tid, None)
        self._thread_info_cache.pop(tid, None)
        self._threads_pending.pop(tid, None)
        self._channel_kinds.pop(tid, None)
        self._mentioned_threads.pop(tid, None)
        for key in [k for k, v in self._threads_by_anchor.items() if v == tid]:
            self._threads_by_anchor.pop(key, None)
        for chat_key in [k for k, v in self._auto_thread_by_chat.items() if v[0] == tid]:
            self._auto_thread_by_chat.pop(chat_key, None)
        for session_key in [k for k, v in self._auto_thread_by_session.items() if v[0] == tid]:
            self._auto_thread_by_session.pop(session_key, None)

    def _pinned_turn_anchor(self, chat_id: str) -> str:
        """Anchor message pinned when the current turn started, else ``""``."""
        key = self._turn_key(chat_id)
        if key[1]:
            return key[1]
        entry = self._turn_anchor.get(key)
        if not entry:
            return ""
        anchor, pinned_at = entry
        if time.monotonic() - pinned_at > self._turn_anchor_ttl:
            self._turn_anchor.pop(key, None)
            return ""
        return str(anchor or "")

    def _turn_key(self, chat_id: str, reply_to=None, metadata=None) -> tuple[str, str]:
        """Only explicit anchors or the executing event can own a delivery."""
        anchor = str(reply_to or (metadata or {}).get("reply_to_message_id") or "")
        context = self._turn_context.get()
        if not anchor and context and str(context[0].source.chat_id) == str(chat_id):
            anchor = str(context[0].message_id or "")
        return str(chat_id), anchor

    def _disable_turn_thread(self, chat_id: str, reply_to=None, metadata=None) -> None:
        key = self._turn_key(chat_id, reply_to, metadata)
        if len(self._turn_thread_disabled) >= self._threads_by_anchor_max:
            self._turn_thread_disabled.pop()
        self._turn_thread_disabled.add(key)

    async def handle_message(self, event: MessageEvent) -> None:
        # Inline commands execute here; spawned work inherits the context.
        marker = self._turn_context.set((event, ""))
        try:
            await super().handle_message(event)
        finally:
            self._turn_context.reset(marker)

    async def _process_message_background(self, event: MessageEvent, session_key: str) -> None:
        # Pending drains can run in the previous turn's task. Rebind to their
        # actual event, keeping ownership until final/footer/media delivery ends.
        marker = self._turn_context.set((event, session_key))
        key = self._turn_key(event.source.chat_id)
        try:
            await super()._process_message_background(event, session_key)
        finally:
            try:
                await self.stop_typing(event.source.chat_id)
            finally:
                self._turn_anchor.pop(key, None)
                self._turn_thread_disabled.discard(key)
                self._turn_context.reset(marker)

    def _pin_turn_anchor(self, chat_id: str, metadata=None) -> None:
        """Remember only the executing event or an explicit trigger."""
        key = self._turn_key(chat_id, metadata=metadata)
        if key in self._turn_anchor:
            return
        anchor = key[1]
        if anchor:
            self._turn_anchor[key] = (anchor, time.monotonic())
            if len(self._turn_anchor) > self._threads_by_anchor_max:
                self._turn_anchor.pop(next(iter(self._turn_anchor)))

    def _remember_thread_id(self, data: Any, parent_chat_id: Optional[str] = None) -> Optional[str]:
        """Record a thread returned by the API and return its id, else ``None``."""
        thread_id = str((data or {}).get("id") or "").strip()
        if not thread_id:
            return None
        self._known_channel_ids.add(thread_id)
        if parent_chat_id:
            self._thread_parents[thread_id] = str(parent_chat_id)
        # This thread now exists, so it is no longer waiting to be created, and it
        # is by definition one this adapter opened.
        self._threads_pending.pop(thread_id, None)
        self._remember_thread_owner(thread_id, True)
        # A thread this adapter created continues a conversation the bot was
        # already addressed in, so ordinary follow-ups in it belong to the bot.
        # Strict mention mode opts out and keeps requiring a fresh mention.
        if not self._strict_mention:
            self._remember_mentioned_thread(thread_id)
        return thread_id

    async def _start_thread_from_message(self, chat_id: str, message_id: str) -> Optional[str]:
        """Best-effort thread start from ``message_id``; ``None`` keeps the channel.

        Thread creation must never cost a reply, so a failure that leaves no thread
        behind falls back to the parent channel and the message still lands. A
        thread already opened for this (chat, message) pair is reused, so one
        turn's progress bubble and its final reply share a thread rather than
        creating two -- across interleaved sends, and across a restart, when the
        thread exists on the server but this process no longer remembers it.
        """
        key = (str(chat_id), str(message_id))
        # An anchor this adapter recorded as a command never opens a thread: the
        # command addresses the turn already running in that chat, and a thread on
        # its own message id would key a container with no session behind it.
        if key in self._no_thread_anchors:
            return None
        existing = self._threads_by_anchor.get(key)
        if existing:
            return existing
        async with self._thread_creation_slot(key):
            # Re-read under the lock: a concurrent send may have opened the thread
            # while this one waited, and creation must not be repeated.
            existing = self._threads_by_anchor.get(key)
            if existing:
                return existing
            # Two-phase naming, same as Discord: the placeholder comes from the message
            # that triggered the turn; the thread is renamed to the session title as
            # soon as the session has one. The title is usually already there (the
            # titler runs early in the turn, the thread is opened at send time), in
            # which case the thread is born with its final name and no rename follows.
            placeholder = _derive_auto_thread_name(self._inbound_text_by_message.get(str(message_id), ""))
            author_id = self._inbound_author_by_message.get(str(message_id)) or None
            title = await asyncio.to_thread(self._session_title_for_chat, str(chat_id), author_id)
            thread_name = _collapse_thread_name(title or "") or placeholder
            # Whether the thread came from the server already existing rather than
            # from a create here: only a thread this adapter created carries a name
            # this adapter chose.
            adopted = False
            try:
                data = await self.create_thread(
                    str(chat_id), thread_name, message_id=str(message_id)
                )
            except Exception as exc:
                if not _thread_already_exists_for_message(exc):
                    logger.debug("Fluxer thread start failed for %s/%s: %s", chat_id, message_id, exc)
                    return None
                # The refusal names the thread: Fluxer anchors a thread to its
                # message and gives it that message's id. Adopt it -- falling back
                # to the channel here is what drops replies out of their thread.
                adopted = True
                logger.info(
                    "Fluxer thread %s already exists for anchor %s; adopting it", message_id, message_id
                )
                data = {"id": str(message_id)}
                server_name = await self._fetch_channel_name(str(message_id))
                if server_name:
                    data["name"] = server_name
            thread_id = self._remember_thread_id(data, chat_id)
            if thread_id:
                self._threads_by_anchor[key] = thread_id
                self._threads_by_anchor.move_to_end(key)
                while len(self._threads_by_anchor) > self._threads_by_anchor_max:
                    self._threads_by_anchor.popitem(last=False)
                created_name = str((data or {}).get("name") or thread_name)
                self._record_auto_thread(chat_id, thread_id, created_name, author_id)
                if created_name != placeholder:
                    self._note_renamed_thread(thread_id, created_name)
                if not title:
                    # Guard with the name this adapter would have given the thread,
                    # never with the one an adopted thread already carries: there
                    # ``created_name`` is the server's current name, which a human
                    # may have chosen, and the guard is exactly what authorises the
                    # overwrite -- so passing it would let the session title clobber
                    # that human's choice.
                    self._schedule_title_rename(
                        chat_id, thread_id, thread_name if adopted else created_name, author_id
                    )
            return thread_id

    @contextlib.asynccontextmanager
    async def _thread_creation_slot(self, key: tuple[str, str]) -> AsyncIterator[None]:
        """Hold the per-anchor creation lock with the anchor marked in use.

        The mark is taken *before* the lock is awaited, so a send still queued on
        it pins the lock's entry as well, and released only when the critical
        section ends. Without the mark the bounded map can evict the lock a send
        is holding, handing the next send for that anchor a second lock.
        """
        users = self._thread_create_lock_users
        users[key] = users.get(key, 0) + 1
        try:
            async with self._thread_create_lock(key):
                yield
        finally:
            remaining = users.get(key, 1) - 1
            if remaining > 0:
                users[key] = remaining
            else:
                users.pop(key, None)

    def _thread_create_lock(self, key: tuple[str, str]) -> asyncio.Lock:
        """A stable per-anchor creation lock, bounded like the anchor map."""
        lock = self._thread_create_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._thread_create_locks[key] = lock
            self._evict_idle_thread_create_locks()
        return lock

    def _evict_idle_thread_create_locks(self) -> None:
        """Drop the oldest idle per-anchor locks until the map is within its bound.

        The map is insertion-ordered, so a bound that evicts purely on age can drop
        the lock a send is still holding (or waiting on): the next send for that
        anchor then builds a second lock and POSTs a second thread -- the duplicate
        this lock exists to prevent. Anchors with a send in flight are skipped, so
        the map may overshoot its bound while every candidate is busy and shrinks
        again as those sends finish.
        """
        for key in list(self._thread_create_locks):
            if len(self._thread_create_locks) <= self._threads_by_anchor_max:
                return
            if key in self._thread_create_lock_users:
                continue
            self._thread_create_locks.pop(key, None)

    # ── session-title naming for a thread this adapter opened ───────────────────

    def _record_auto_thread(
        self, chat_id: str, thread_id: str, name: str, user_id: Optional[str] = None
    ) -> None:
        """Remember the thread opened for ``chat_id`` (last one wins per chat)."""
        chat_key = str(chat_id)
        tid, label = str(thread_id), str(name or "")
        self._auto_thread_by_chat[chat_key] = (tid, label)
        self._auto_thread_by_chat.move_to_end(chat_key)
        while len(self._auto_thread_by_chat) > 1000:
            self._auto_thread_by_chat.popitem(last=False)
        session_key = (chat_key, str(user_id or ""))
        self._auto_thread_by_session[session_key] = (tid, label)
        self._auto_thread_by_session.move_to_end(session_key)
        while len(self._auto_thread_by_session) > 1000:
            self._auto_thread_by_session.popitem(last=False)

    def auto_thread_info_for_chat(self, chat_id: str) -> Optional[tuple[str, str]]:
        """``(thread_id, name)`` of the thread this adapter opened for ``chat_id``, if any."""
        return self._auto_thread_by_chat.get(str(chat_id))

    def _auto_thread_for_session(
        self, chat_id: str, user_id: Optional[str]
    ) -> Optional[tuple[str, str]]:
        """The thread this session owns, or ``None`` when the chat is ambiguous.

        ``/title`` sets ONE session's title, so the rename target has to be that
        session's thread. The per-chat entry is the shared "latest thread"
        contract and carries no participant, so in a channel where two people each
        have a thread it points at whichever was opened last — renaming the other
        person's thread, with the placeholder guard still passing because it
        compares that thread's own cached name. Trust the per-chat entry only when
        it is the chat's sole auto-thread; otherwise require an exact
        ``(chat, user)`` match, and decline rather than rename someone else's.
        """
        chat_key = str(chat_id)
        user_key = str(user_id or "")
        exact = self._auto_thread_by_session.get((chat_key, user_key))
        if exact:
            return exact
        # No entry for this user: only rename the chat's sole auto-thread, and only
        # when nothing records it as someone else's. An unattributed thread (the
        # triggering author was never seen) stays eligible, which keeps the
        # best-effort behavior; an attributed one must match.
        candidates: Dict[str, tuple[tuple[str, str], str]] = {}
        owners = {
            entry[0]: entry_user
            for (entry_chat, entry_user), entry in self._auto_thread_by_session.items()
            if entry_chat == chat_key
        }
        latest = self._auto_thread_by_chat.get(chat_key)
        if latest:
            candidates[str(latest[0])] = (latest, owners.get(str(latest[0]), ""))
        for (entry_chat, entry_user), entry in self._auto_thread_by_session.items():
            if entry_chat == chat_key:
                candidates[str(entry[0])] = (entry, entry_user)
        if len(candidates) == 1:
            entry, owner = next(iter(candidates.values()))
            if not owner or owner == user_key:
                return entry
        if candidates:
            logger.debug(
                "Fluxer /title rename skipped: %d auto-thread(s) in channel %s and none for user %s",
                len(candidates), chat_key, user_id,
            )
        return None

    def _session_title_for_chat(self, chat_id: str, user_id: Optional[str] = None) -> Optional[str]:
        """The triggering session's title for a chat, or ``None``.

        Reads the title back through Hermes' own session store: the gateway's
        auto-title runs in its own thread and is not surfaced to platform plugins,
        so the title is read rather than pushed. ``user_id`` narrows the lookup to
        the participant whose message opened the thread, which is what a channel
        with several live sessions needs.

        Only a title the model wrote or the user set counts. ``derived`` is skipped
        deliberately: it is a slice of the triggering message, i.e. what the
        placeholder already says. Acquisition, reads and normalization are all
        best-effort: optional naming must never prevent message delivery.
        """
        try:
            store = self._get_session_store()
            chat_key = str(chat_id or "")
            if store is None or not chat_key:
                return None
            session_id = store.find_session_by_origin(
                platform=str(self.platform.value), chat_id=chat_key,
                user_id=str(user_id) if user_id else None,
            )
            if not session_id:
                return None
            if store.get_session_title_source(session_id) not in _TITLE_RENAME_SOURCES:
                return None
            title = store.get_session_title(session_id)
            return str(title).strip() or None
        except Exception as exc:
            logger.debug("Fluxer session title lookup failed (%s)", type(exc).__name__)
            return None

    def _session_origin_for_key(self, session_key: str) -> Optional[tuple[str, str]]:
        """Newest session origin, or ``None`` when any part of lookup fails.

        Keep optional ``/title`` thread renames best-effort, including reader
        acquisition and conversion of the stored chat/participant identifiers.
        """
        try:
            parts = str(session_key or "").split(":", 2)
            profile = parts[1] if len(parts) == 3 and parts[0] == "agent" else None
            store = self._get_session_store(profile=None if profile == "main" else profile)
            key = str(session_key or "").strip()
            if store is None or not key:
                return None
            rows = store.list_sessions_rich(
                source=str(self.platform.value), session_key=key, limit=1
            )
            if not rows or not isinstance(rows[0], Mapping):
                return None
            chat_id = rows[0].get("chat_id")
            if not chat_id:
                return None
            user_id = rows[0].get("user_id")
            return str(chat_id), str(user_id) if user_id else ""
        except Exception as exc:
            logger.debug("Fluxer session origin lookup failed (%s)", type(exc).__name__)
            return None

    def _chat_id_for_session(self, session_key: str) -> Optional[str]:
        """Chat id of the newest session row for a ``session_key``, or ``None``."""
        origin = self._session_origin_for_key(session_key)
        return origin[0] if origin else None

    def _get_session_store(self, profile: Optional[str] = None) -> Optional[Any]:
        """An owned reader for the routed profile, not the host-injected field."""
        context = self._turn_context.get()
        source = context[0].source if context else None
        profile = profile or getattr(source, "profile", None) or getattr(self, "_owner_profile", None)
        home = self._title_home
        if profile:
            try:
                from hermes_cli.profiles import get_profile_dir

                home = Path(get_profile_dir(profile))
            except (ImportError, ValueError):
                return None  # Never fall back to another profile's titles.
        with self._session_db_lock:
            if time.monotonic() < self._session_retry_after.get(home, 0.0):
                return None
            if home not in self._session_readers:
                store = _open_session_store(home)
                if store is None:
                    self._session_retry_after[home] = time.monotonic() + _TITLE_RENAME_POLL_INTERVAL
                    return None
                self._session_readers[home] = store
                self._session_retry_after.pop(home, None)
            self._session_db = self._session_readers[home]
            return self._session_db

    def _close_session_store(self) -> None:
        """Release the session store handle; a later use reopens it."""
        with self._session_db_lock:
            stores = list(self._session_readers.values())
            self._session_readers.clear()
            self._session_retry_after.clear()
            self._session_db = None
            self._session_db_off = False
            for store in stores:
                try:
                    store.close()
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Fluxer session store close failed: %s", type(exc).__name__)

    def _schedule_title_rename(
        self, chat_id: str, thread_id: str, placeholder: str, user_id: Optional[str] = None
    ) -> None:
        """Watch for this session's title and rename ``thread_id`` when it appears."""
        key = str(thread_id)
        pending = self._title_tasks.get(key)
        if pending is not None and not pending.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._loop = loop
        task = loop.create_task(
            self._rename_thread_when_titled(str(chat_id), key, str(placeholder), user_id)
        )
        self._title_tasks[key] = task
        task.add_done_callback(lambda _t, k=key: self._title_tasks.pop(k, None))
        while len(self._title_tasks) > _TITLE_RENAME_MAX_PENDING:
            # Cancel the evicted task, don't merely drop the reference: an
            # orphaned task keeps polling its lookup for the rest of the window,
            # and ``disconnect`` cannot cancel what ``_title_tasks`` no longer
            # holds, so it could reopen the session store and rename a thread
            # after shutdown.
            evicted = self._title_tasks.pop(next(iter(self._title_tasks)))
            evicted.cancel()

    async def _rename_thread_when_titled(
        self, chat_id: str, thread_id: str, placeholder: str, user_id: Optional[str] = None
    ) -> None:
        """Rename a thread to its session title, bounded by a wall-clock window.

        Best-effort by design: a session that never gets a title, or a thread a
        human renamed in the meantime (declined by the placeholder guard), simply
        keeps the name it has.
        """
        deadline = time.monotonic() + _TITLE_RENAME_WINDOW_SECONDS
        while True:
            title = await asyncio.to_thread(self._session_title_for_chat, chat_id, user_id)
            if title and title != placeholder:
                try:
                    applied = await self.rename_thread(
                        thread_id, title, only_if_current_name=placeholder or None
                    )
                except Exception as exc:
                    logger.debug("Fluxer title rename failed for %s: %s", thread_id, exc)
                    return
                if applied:
                    logger.info(
                        "Fluxer thread %s named from session title %r", thread_id, title
                    )
                return
            if time.monotonic() >= deadline:
                return
            await asyncio.sleep(_TITLE_RENAME_POLL_INTERVAL)

    def rename_thread_from_session_title(self, session_key: str, title: str) -> None:
        """Rename this adapter's thread for a session, from a sync entry point.

        Used by the ``pre_command`` hook so ``/title`` reaches the thread as well as
        the session record. Renames only a thread still carrying its placeholder.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            coro = self._rename_thread_for_session(session_key, title)
        except Exception:
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception as exc:
            coro.close()
            logger.debug("Fluxer /title rename could not be scheduled: %s", exc)

    async def _rename_thread_for_session(self, session_key: str, title: str) -> None:
        origin = await asyncio.to_thread(self._session_origin_for_key, str(session_key))
        if not origin:
            return
        entry = self._auto_thread_for_session(origin[0], origin[1])
        if not entry:
            return
        thread_id, initial_name = entry
        try:
            applied = await self.rename_thread(thread_id, title, only_if_current_name=initial_name or None)
        except Exception as exc:
            logger.debug("Fluxer /title rename failed for %s: %s", thread_id, exc)
            return
        if applied:
            logger.info("Fluxer thread %s named from /title", thread_id)

    async def _fetch_channel_name(self, channel_id: str) -> Optional[str]:
        """Current server-side name of a channel or thread, else ``None``."""
        try:
            data = await self._request(
                "GET", f"/channels/{_quote_id(channel_id)}", warn_on_error=False
            )
        except Exception as exc:
            logger.debug("Fluxer channel lookup failed for %s: %s", channel_id, exc)
            return None
        if isinstance(data, Mapping):
            return str(data.get("name") or "")
        return None

    def _note_renamed_thread(self, thread_id: str, name: str) -> None:
        """Keep the cached auto-thread name in step with a rename this adapter applied."""
        tid = str(thread_id)
        for chat_key, (cached_id, _cached_name) in list(self._auto_thread_by_chat.items()):
            if cached_id == tid:
                self._auto_thread_by_chat[chat_key] = (cached_id, name)
        for session_key, (cached_id, _cached_name) in list(self._auto_thread_by_session.items()):
            if cached_id == tid:
                self._auto_thread_by_session[session_key] = (cached_id, name)


    async def _fetch_thread_info(self, thread_id: str) -> Dict[str, Any]:
        """Read a channel's parent, owner and kind from the API.

        Fluxer returns a thread as a channel whose ``parent_id`` is the channel it
        hangs under; the thread's creator is the message author, so a thread the
        bot opened can be told apart from one a human created. ``type`` is the same
        channel-type integer ``_chat_type`` reads. The id itself is the anchor
        message id, which the caller already knows, so only these fields are read.
        """
        data = await self._request("GET", f"/channels/{_quote_id(thread_id)}", warn_on_error=False)
        info: Dict[str, Any] = {}
        if isinstance(data, Mapping):
            if data.get("type") is not None:
                info["type"] = data["type"]
            parent = data.get("parent_id") or data.get("parent_chat_id")
            if parent:
                info["parent_id"] = str(parent)
            owner = data.get("owner_id") or data.get("ownerId")
            if owner is not None:
                info["owner_id"] = str(owner)
        return info

    def _remember_channel_kind(self, channel_id: str, info: Mapping[str, Any]) -> None:
        """Cache a channel id's kind from an API read, bounded like the other maps."""
        cid = str(channel_id or "").strip()
        if not cid or info.get("type") is None:
            return
        self._channel_kinds[cid] = _chat_type(info.get("type"))
        self._channel_kinds.move_to_end(cid)
        while len(self._channel_kinds) > self._threads_by_anchor_max:
            self._channel_kinds.popitem(last=False)

    async def _resolve_channel_kind(self, channel_id: str) -> Optional[str]:
        """The kind of ``channel_id`` (``thread`` / ``channel`` / ``dm``), or ``None``.

        An event that classifies itself (``channel_type``, or a nested channel
        object) never reaches here. Fluxer's real MESSAGE_CREATE carries neither, so
        the kind is learned from the API once and remembered; a lookup that fails
        returns ``None`` and the caller keeps the event exactly as it arrived rather
        than guessing "channel" and splitting the session.
        """
        cid = str(channel_id or "").strip()
        if not cid:
            return None
        known = self._channel_kinds.get(cid)
        if known is not None:
            return known
        # A thread this adapter opened already names its kind and its parent.
        if cid in self._thread_parents or self._thread_is_bot_owned(cid) is not None:
            self._remember_channel_kind(cid, {"type": 11})
            return "thread"
        if self._threads_pending.get(cid):
            return "thread"
        info = await self._resolve_thread_info(cid)
        if info.get("type") is None:
            return None
        return self._channel_kinds.get(cid)

    async def _resolve_thread_info(self, thread_id: str, *, expect_thread: bool = False) -> Dict[str, Any]:
        """Cached ``_fetch_thread_info``; a failed lookup is never cached.

        ``expect_thread`` states that the caller already knows the channel is a
        thread (its message arrived in one). Only then is a parent recorded when the
        API read omits ``type``: the plain kind-resolution path must not mistake a
        text channel's category for a parent channel.
        """
        tid = str(thread_id or "").strip()
        if not tid:
            return {}
        cached = self._thread_info_cache.get(tid)
        if cached is not None:
            return cached
        try:
            info = await self._fetch_thread_info(tid)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A lookup failure must not invent a parent, and must not cost the
            # message: leave the thread exactly as it arrived and try again later.
            logger.debug("Fluxer thread info lookup failed for %s (%s)", tid, type(exc).__name__)
            return {}
        if not isinstance(info, Mapping):
            return {}
        info = dict(info)
        # The kind comes first: a thread this adapter did not open must be decided
        # BEFORE its parent is recorded, or an unknown-owner thread would inherit
        # the parent channel's hard allowlist the moment the parent is cached.
        self._remember_channel_kind(tid, info)
        owner = info.get("owner_id")
        owned = self._thread_is_bot_owned(tid)
        if owner is not None and self.bot_user_id is not None:
            owned = str(owner) == str(self.bot_user_id)
            self._thread_owners[tid] = owned
            # A thread this adapter opened is one the bot was addressed in, so its
            # ordinary follow-ups belong to the bot -- the same rule the create path
            # applies. Learning it here is what keeps a cold restart from turning the
            # bot's own thread into a mention-gated drop zone.
            if owned and not self._strict_mention:
                self._remember_mentioned_thread(tid)
        # The parent is recorded for a THREAD only, and only when the API read says
        # the channel is one. This method is also reached from channel *kind*
        # resolution, where ``parent_id`` is a category rather than a parent
        # channel; caching that in the map the thread fallbacks read would send a
        # fallback to a category. Inheritance of the parent's admission rules is a
        # separate decision, gated on confirmed ownership in
        # ``_should_process_message``.
        parent = info.get("parent_id")
        known_thread = expect_thread or self._channel_kinds.get(tid) == "thread"
        if parent and known_thread:
            self._thread_parents[tid] = str(parent)
        elif known_thread:
            self._thread_parents.pop(tid, None)
        self._thread_info_cache[tid] = info
        # Bounded like the other per-adapter caches: a long-lived process must not
        # grow one entry per thread it has ever seen.
        if len(self._thread_info_cache) > self._threads_by_anchor_max:
            self._thread_info_cache.pop(next(iter(self._thread_info_cache)), None)
        return info

    def _remember_thread_owner(self, thread_id: str, owned: bool) -> None:
        """Record whether ``thread_id`` is a thread this adapter opened."""
        tid = str(thread_id or "").strip()
        if tid:
            self._thread_owners[tid] = bool(owned)

    def _remember_no_thread_anchor(self, chat_id: str, anchor: str) -> None:
        """Mark ``(chat_id, anchor)`` as a turn that must never open a thread.

        A text command addresses the turn already running in its chat, so the
        container it resolves to is that chat -- not a thread opened on the
        command's own message id. The anchor is recorded because the typing
        indicator and the reply both derive their container from the turn anchor,
        and each would otherwise open a thread of its own for the command.
        """
        key = (str(chat_id or ""), str(anchor or ""))
        if not key[0] or not key[1]:
            return
        self._no_thread_anchors.add(key)
        while len(self._no_thread_anchors) > self._threads_by_anchor_max:
            self._no_thread_anchors.pop()

    def _remember_dead_container(self, container_id: str, fallback: str) -> None:
        """Redirect every later send for a deleted container to ``fallback``.

        A deleted thread's id never comes back, so recording the redirect is what
        keeps the rest of the turn (and the typing indicator) from re-addressing
        the same missing thread and failing.
        """
        cid = str(container_id or "").strip()
        target = str(fallback or "").strip()
        if not cid or not target or cid == target:
            return
        self._dead_containers[cid] = target
        while len(self._dead_containers) > self._threads_by_anchor_max:
            self._dead_containers.pop(next(iter(self._dead_containers)), None)

    def _thread_is_bot_owned(self, thread_id: str) -> Optional[bool]:
        """Whether ``thread_id`` is the bot's own thread, or ``None`` if unknown."""
        return self._thread_owners.get(str(thread_id or ""))

    async def _resolve_thread_source(
        self, data: Dict[str, Any], channel_id: str, chat_type: str
    ) -> tuple[str, str, Optional[str], Optional[str]]:
        """Normalize an inbound message to one routing contract.

        Returns ``(chat_id, chat_type, parent_chat_id, pending_parent)``.

        A message that arrives in a thread is keyed on that thread's own id in the
        ``chat_id`` slot with no thread slot -- the shape the host derives for a
        platform whose threads are separate chat containers (Discord), and the shape
        this deployment's persisted Fluxer thread sessions already use. The parent
        channel travels in ``parent_chat_id`` for admission and profile routing.

        A channel message that will open a thread is keyed on the anchor message id
        -- the id Fluxer will give the thread -- so the turn that opens a thread and
        every follow-up inside it derive ONE session key. The thread itself is not
        created here: creating it is a send-time decision, and a turn that never
        delivers must not leave an empty thread behind. ``pending_parent`` names the
        channel the thread must be created under when the first send happens, and is
        only registered once the message has been admitted.
        """
        if chat_type == "dm":
            return channel_id, chat_type, None, None
        if chat_type == "thread":
            # Parent/owner come from the create that opened the thread, or are read
            # back once after a restart; an unresolvable thread keeps its own id.
            # The parent is reported whatever the owner is -- the host uses it for
            # profile routing and branching -- while inheritance of the parent's
            # admission rules stays gated on confirmed ownership in
            # ``_should_process_message``.
            parent = self._thread_parents.get(str(channel_id))
            if parent is None:
                info = await self._resolve_thread_info(channel_id, expect_thread=True)
                parent = info.get("parent_id")
            return channel_id, chat_type, (str(parent) if parent else None), None
        if not self._thread_replies_enabled_for(channel_id):
            return channel_id, chat_type, None, None
        # A plain-text command addresses the turn already running in this chat, so
        # it keeps the channel's session: auto-threading it would key it on the
        # command's own message id and the stop would never reach its target. The
        # anchor is recorded as never-threadable so the typing indicator and the
        # reply -- which both resolve their container from the turn anchor -- do not
        # open a thread on the command's own message either.
        if _is_text_command(str(data.get("content") or "")):
            anchor = str(data.get("id") or "").strip()
            if anchor:
                self._remember_no_thread_anchor(channel_id, anchor)
            return channel_id, chat_type, None, None
        anchor = str(data.get("id") or "").strip()
        if not anchor:
            return channel_id, chat_type, None, None
        # Bounded: a channel that never delivers must not leak one entry per message.
        if len(self._threads_pending) > self._threads_by_anchor_max:
            self._threads_pending.pop(next(iter(self._threads_pending)), None)
        return anchor, "thread", str(channel_id), str(channel_id)

    def _classify_untyped_event(self, channel_id: str) -> Optional[str]:
        """Kind of ``channel_id`` from state already in hand, else ``None``.

        Fluxer's own MESSAGE_CREATE payload carries no ``channel_type`` and no nested
        channel object, so a raw event cannot classify itself. Everything the adapter
        has *learned* about the id -- a thread it opened, a thread it is about to
        open, a kind a previous lookup recorded -- decides it here with no I/O.
        """
        cid = str(channel_id or "").strip()
        if not cid:
            return None
        known = self._channel_kinds.get(cid)
        if known is not None:
            return known
        if cid in self._thread_parents or self._thread_is_bot_owned(cid) is not None:
            return "thread"
        if self._threads_pending.get(cid):
            return "thread"
        return None

    def _typing_target_sync(self, chat_id: str, metadata=None) -> tuple[str, bool]:
        """``(container, needs_creation)`` for a typing indicator.

        A turn whose session is a thread delivers into that thread, so the indicator
        has to refresh the thread -- a parent-channel indicator is what a user
        watching the thread sees as "nothing happening". A prospective thread that
        does not exist yet reports ``needs_creation``; the caller opens it first.

        Precedence matches the delivery resolver: an explicit container wins, then a
        container this adapter already knows, then the prospective thread this turn
        will open, then the caller's chat id.
        """
        cid = str(chat_id or "")
        if not cid:
            return "", False
        # A container delivery already found deleted is never addressed again.
        redirect = self._dead_containers.get(cid)
        if redirect:
            return str(redirect), False
        explicit = str((metadata or {}).get("thread_id") or "").strip()
        if explicit:
            return explicit, False
        # Already a container of its own: a thread the adapter opened, or a thread
        # whose parent/owner is known. Post straight into it.
        if cid in self._thread_parents or self._thread_is_bot_owned(cid) is not None:
            return cid, False
        # The chat id IS a not-yet-created thread's anchor: the thread to open is
        # this container, under the parent recorded when the message was admitted.
        if self._threads_pending.get(cid):
            return cid, True
        # A channel: the thread this turn will deliver into, if thread replies apply.
        anchor = self._turn_key(cid, metadata=metadata)[1]
        if not anchor:
            return cid, False
        existing = self._threads_by_anchor.get((cid, anchor))
        if existing:
            return existing, False
        if self._no_thread_anchors and (cid, anchor) in self._no_thread_anchors:
            return cid, False
        if self._threads_pending.get(cid) == anchor or self._thread_replies_enabled_for(cid):
            return anchor, True
        return cid, False

    async def _typing_target(self, chat_id: str, metadata=None) -> str:
        """The container a typing indicator for this turn belongs in."""
        target, needs_creation = self._typing_target_sync(chat_id, metadata)
        if not needs_creation:
            return target or str(chat_id)
        parent = (
            self._threads_pending.get(target)
            or self._threads_pending.get(str(chat_id))
            or str(chat_id)
        )
        created = await self._start_thread_from_message(str(parent), target)
        # A thread that could not be opened is not a container: refresh the parent
        # channel rather than posting a typing indicator at a channel that is not
        # there (the prospective id names a thread that does not exist yet).
        return created or str(parent)

    async def _resolve_thread_container(
        self, chat_id: str, metadata=None, session_key: Optional[str] = None
    ) -> str:
        """The container a non-message operation (approval, confirm) belongs in.

        An explicit ``thread_id`` wins. A container delivery already found deleted
        is redirected to its parent. Otherwise a session keyed on a thread names
        its container in the key itself, so an approval prompt posted while a
        thread-keyed turn is waiting for input lands in that thread -- and a thread
        that does not exist yet is created under its parent first. A creation that
        fails falls back to the parent channel: aiming the prompt at a container
        that is not there 404s, and the waiting turn can then never be answered.
        """
        explicit = str((metadata or {}).get("thread_id") or "").strip()
        if explicit:
            return explicit
        redirect = self._dead_containers.get(str(chat_id))
        if redirect:
            return str(redirect)
        candidate = _thread_id_from_session_key(session_key, str(self.platform.value))
        if candidate:
            redirect = self._dead_containers.get(str(candidate))
            if redirect:
                return str(redirect)
            pending_parent = self._threads_pending.get(candidate)
            if pending_parent:
                created = await self._start_thread_from_message(str(pending_parent), candidate)
                # The parent is where the prompt has to land when the thread could
                # not be opened: a prospective id names a channel that does not exist.
                return created or str(pending_parent)
            return candidate
        pending_parent = self._threads_pending.get(str(chat_id))
        if pending_parent:
            created = await self._start_thread_from_message(str(pending_parent), str(chat_id))
            return created or str(pending_parent)
        return str(chat_id)

    async def _resolve_outbound_target(
        self,
        chat_id: str,
        reply_to: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any], Optional[str]]:
        """Resolve ``(target_chat_id, reference_payload, auto_thread_id)`` for one send.

        Precedence mirrors the shared platform contract: an explicit
        ``metadata['thread_id']`` wins, then opt-in thread replies may start a
        thread from the triggering message, then the channel itself. The reply
        reference honours ``FLUXER_REPLY_TO_MODE`` (``off`` suppresses it;
        ``first``/``all`` place it on the first or every split chunk).
        ``auto_thread_id`` names the thread this adapter chose for the turn (so a
        caller can abandon it if it turns out to be gone); it is ``None`` when the
        destination was explicit.
        """
        payload: Dict[str, Any] = {}
        target_chat_id = str(chat_id)
        auto_thread_id: Optional[str] = None
        explicit_thread_id = str((metadata or {}).get("thread_id") or "").strip()
        started_thread = False
        # A container delivery already found deleted is never addressed again: the
        # redirect is what keeps the rest of the turn in the parent channel instead
        # of failing against the same missing thread.
        dead_redirect = self._dead_containers.get(str(chat_id))
        if dead_redirect:
            target_chat_id = str(dead_redirect)
        elif explicit_thread_id:
            target_chat_id = explicit_thread_id
        elif str(chat_id) in self._thread_parents or self._thread_is_bot_owned(str(chat_id)) is not None:
            # The caller's chat id IS a thread this adapter already opened. It has to
            # count as an auto-chosen container, not a plain channel: a thread opened
            # earlier in the turn (by the typing indicator or a progress send) can be
            # deleted before the reply, and only an ``auto_thread_id`` gets the
            # parent-channel fallback instead of a failed send.
            target_chat_id = str(chat_id)
            auto_thread_id = str(chat_id)
        else:
            turn_key = self._turn_key(chat_id, reply_to, metadata)
            anchor = turn_key[1]
            # A prospective thread: the session container IS the anchor message, and
            # Fluxer will give the thread that same id, so creating it here is what
            # puts the reply inside the thread the session was keyed on. A creation
            # failure still has to deliver, so it falls back to the parent channel.
            pending_parent = self._threads_pending.get(str(chat_id))
            if pending_parent:
                if turn_key in self._turn_thread_disabled:
                    target_chat_id = str(pending_parent)
                else:
                    new_thread_id = await self._start_thread_from_message(str(pending_parent), str(chat_id))
                    if new_thread_id:
                        target_chat_id = new_thread_id
                        auto_thread_id = new_thread_id
                        started_thread = True
                    else:
                        self._disable_turn_thread(chat_id, reply_to, metadata)
                        target_chat_id = str(pending_parent)
            elif self._thread_replies_enabled_for(chat_id) and turn_key not in self._turn_thread_disabled:
                # The gateway sets ``reply_to`` only for a few platforms and supplies
                # no thread id for Fluxer. Prefer the trigger in hand, then the anchor
                # pinned when this turn started -- the channel's latest inbound message
                # can move while a reply is still being produced, and using it here
                # would split one turn across two threads. An anchor this adapter
                # recorded as a command keeps the channel: a thread opened on the
                # command's own message keys the reply to a session with no run.
                if anchor and (chat_id, anchor) not in self._no_thread_anchors:
                    new_thread_id = await self._start_thread_from_message(chat_id, anchor)
                    if new_thread_id:
                        target_chat_id = new_thread_id
                        auto_thread_id = new_thread_id
                        started_thread = True
                    else:
                        # Choosing the parent is a turn-level decision, including
                        # a failed/empty creation response, not just deleted threads.
                        self._disable_turn_thread(chat_id, reply_to, metadata)
        # A thread started from the triggering message already carries the parent
        # context, so a second reference inside it is redundant.
        if reply_to and self._reply_to_mode != "off" and not started_thread:
            payload["message_reference"] = {"message_id": str(reply_to)}
        return target_chat_id, payload, auto_thread_id

    async def create_handoff_thread(self, parent_chat_id: str, name: str) -> Optional[str]:
        """Create a fresh Fluxer thread for a CLI→platform handoff or ``/branch``.

        Implements the shared ``BasePlatformAdapter.create_handoff_thread``
        contract (the base returns ``None``), so a handoff's scrollback gets its
        own thread instead of landing in the parent channel. Returns the new
        thread id, or ``None`` when the server refuses -- callers then fall back
        to ``parent_chat_id``.
        """
        try:
            data = await self.create_thread(
                str(parent_chat_id), _collapse_thread_name(name) or _DEFAULT_THREAD_NAME
            )
        except Exception as exc:
            logger.warning("Fluxer handoff thread creation failed for %s: %s", parent_chat_id, exc)
            return None
        return self._remember_thread_id(data, parent_chat_id)

    async def create_thread(
        self,
        chat_id: str,
        name: str,
        *,
        message_id: Optional[str] = None,
        auto_archive_duration: int = 1440,
        thread_type: int = 11,
    ) -> Dict[str, Any]:
        """Create a Fluxer thread, optionally from an existing message."""
        if message_id:
            payload = {"name": name, "auto_archive_duration": auto_archive_duration, "rate_limit_per_user": 0}
            return await self._request(
                "POST",
                f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}/threads",
                json=payload,
            )
        payload = {
            "name": name,
            "type": thread_type,
            "auto_archive_duration": auto_archive_duration,
            "rate_limit_per_user": 0,
            "invitable": True,
        }
        return await self._request("POST", f"/channels/{_quote_id(chat_id)}/threads", json=payload)

    async def rename_thread(
        self,
        thread_id: str,
        name: str,
        *,
        only_if_current_name: Optional[str] = None,
        **_unused: Any,
    ) -> bool:
        """Rename a thread (or channel); ``True`` when the new name was applied.

        ``only_if_current_name`` protects a thread a human already renamed: when
        the server's current name no longer matches the placeholder this adapter
        created, the rename is declined instead of clobbering the human's choice.
        Extra keyword arguments are accepted and ignored so the gateway's rename
        lanes can drive every platform adapter through one call shape.
        """
        tid = str(thread_id or "").strip()
        cleaned = _collapse_thread_name(name)
        if not tid or not cleaned:
            return False
        if only_if_current_name is not None:
            current = await self._fetch_channel_name(tid)
            if current is None or current.strip() != str(only_if_current_name).strip():
                logger.debug(
                    "Fluxer thread %s rename declined: current name %r does not match %r",
                    tid, current, only_if_current_name,
                )
                return False
        try:
            await self._request("PATCH", f"/channels/{_quote_id(tid)}", json={"name": cleaned})
        except Exception as exc:
            logger.debug("Fluxer thread rename failed for %s: %s", tid, exc)
            return False
        self._note_renamed_thread(tid, cleaned)
        return True

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Start a persistent Fluxer typing indicator loop.

        Fluxer typing indicators expire after a short window. Hermes calls
        ``send_typing`` when a turn starts, so keep refreshing until the runner
        calls ``stop_typing`` after the final response. This preserves a
        clean typing-only "I am working" mode when ``display.tool_progress`` is
        set to ``off``.
        """
        chat_key = self._turn_key(chat_id, metadata=metadata)
        self._pin_turn_anchor(chat_id, metadata)
        if chat_key in self._typing_tasks:
            return
        # The indicator belongs to the container the turn delivers into, so resolve
        # it through the same resolver as delivery -- opening the prospective thread
        # if it does not exist yet, rather than refreshing the parent channel.
        target = await self._typing_target(chat_id, metadata)

        async def _typing_loop() -> None:
            try:
                while True:
                    try:
                        await self._request("POST", f"/channels/{_quote_id(target)}/typing", json={})
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.debug("Fluxer typing indicator failed for %s: %s", chat_key, exc)
                        return
                    await asyncio.sleep(8)
            except asyncio.CancelledError:
                pass
            finally:
                self._typing_tasks.pop(chat_key, None)

        self._typing_tasks[chat_key] = asyncio.create_task(_typing_loop(), name=f"fluxer-typing-{chat_key}")

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        chat_key = self._turn_key(chat_id, metadata=metadata)
        # Typing can stop before trailing output; only delivery completion
        # releases ownership. One user's stop cannot cancel another user's loop.
        task = self._typing_tasks.pop(chat_key, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        try:
            cached = await cache_image_from_url(image_url, Path(image_url.split("?", 1)[0]).suffix or ".jpg")
            return await self.send_image_file(chat_id, cached, caption=caption, reply_to=reply_to, metadata=metadata)
        except Exception as exc:
            logger.warning("Fluxer image URL upload failed: %s", exc)
            return _failed_send_result(exc)

    async def send_image_file(
        self,
        chat_id: str,
        image_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            image_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_document(
        self,
        chat_id: str,
        file_path: str,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            file_path,
            caption=caption,
            file_name=file_name,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_video(
        self,
        chat_id: str,
        video_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            video_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            title=kwargs.get("title"),
        )

    async def send_voice(
        self,
        chat_id: str,
        audio_path: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SendResult:
        return await self._send_file_message(
            chat_id,
            audio_path,
            caption=caption,
            reply_to=reply_to,
            metadata=metadata,
            flags=_VOICE_MESSAGE_FLAG,
            is_voice=True,
            duration=kwargs.get("duration"),
            waveform=kwargs.get("waveform"),
            title=kwargs.get("title"),
        )

    async def _send_file_message(
        self,
        chat_id: str,
        file_path: str,
        *,
        caption: Optional[str] = None,
        file_name: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        flags: int = 0,
        is_voice: bool = False,
        duration: Optional[int] = None,
        waveform: Optional[str] = None,
        title: Optional[str] = None,
    ) -> SendResult:
        resolved = self.validate_media_delivery_path(file_path)
        if not resolved:
            return SendResult(success=False, error=f"Unsafe or missing file path: {file_path}", retryable=False)

        path = Path(resolved)
        filename = file_name or path.name
        # Uploads must honour the same destination and reference rules as text:
        # an explicit thread_id wins, thread replies may start a thread, and
        # FLUXER_REPLY_TO_MODE=off suppresses the reference here too.
        target_chat_id, reference, auto_thread_id = await self._resolve_outbound_target(chat_id, reply_to, metadata)
        payload: Dict[str, Any] = {
            "nonce": str(int(time.time() * 1000)),
            "allowed_mentions": self._allowed_mentions_payload(),
        }
        if caption and not is_voice:
            payload["content"] = self._sanitize_outbound_mentions(caption)
        payload.update(reference)
        if flags:
            payload["flags"] = flags
        self._known_channel_ids.add(str(target_chat_id))

        attachment: Dict[str, Any] = {"id": 0, "filename": filename, "title": title or filename}
        if is_voice:
            # Fluxer's schema requires duration + waveform for VOICE_MESSAGE uploads.
            attachment["duration"] = int(duration) if duration is not None else _audio_duration_seconds(path)
            attachment["waveform"] = str(waveform or "AAAA")
        payload["attachments"] = [attachment]

        try:
            try:
                data = await self._multipart_request(
                    "POST",
                    f"/channels/{_quote_id(target_chat_id)}/messages",
                    payload=payload,
                    files=[("files[0]", path, filename)],
                )
            except Exception as exc:
                if _is_reply_reference_rejected_error(exc) and payload.get("message_reference"):
                    # Same refusal as text sends, and the same answer: the
                    # reference is what Fluxer rejected, so the upload goes out
                    # without it rather than being lost.
                    logger.info(
                        "Fluxer refused the reply reference (%s); uploading without it",
                        type(exc).__name__,
                    )
                    payload.pop("message_reference", None)
                    data = await self._multipart_request(
                        "POST",
                        f"/channels/{_quote_id(target_chat_id)}/messages",
                        payload=payload,
                        files=[("files[0]", path, filename)],
                    )
                elif auto_thread_id and _is_missing_channel_error(exc):
                    # Same deleted-thread fallback as text sends: an upload must
                    # not fail just because the thread this adapter chose was
                    # removed. The parent channel is the fallback for a prospective
                    # thread, whose caller-side chat_id IS the thread.
                    fallback_chat = (
                        self._thread_parents.get(str(chat_id))
                        or self._threads_pending.get(str(chat_id))
                        or str(chat_id)
                    )
                    logger.info(
                        "Fluxer thread %s is gone; uploading to channel %s instead",
                        auto_thread_id,
                        fallback_chat,
                    )
                    self._forget_thread(auto_thread_id)
                    self._disable_turn_thread(chat_id, reply_to, metadata)
                    self._remember_dead_container(auto_thread_id, fallback_chat)
                    self._remember_dead_container(chat_id, fallback_chat)
                    auto_thread_id = None
                    target_chat_id = str(fallback_chat)
                    self._known_channel_ids.add(target_chat_id)
                    payload.pop("message_reference", None)
                    if reply_to and self._reply_to_mode != "off":
                        payload["message_reference"] = {"message_id": str(reply_to)}
                    data = await self._multipart_request(
                        "POST",
                        f"/channels/{_quote_id(target_chat_id)}/messages",
                        payload=payload,
                        files=[("files[0]", path, filename)],
                    )
                else:
                    raise
            message_id = str(data.get("id")) if data.get("id") else None
            # Edits/deletes must target the thread, not the caller's channel.
            self._remember_message_destination(message_id, target_chat_id)
            verified = await self._verify_delivery(
                target_chat_id,
                message_id,
                expected_content=payload.get("content"),
                expected_attachment_count=1,
            )
            raw_response = dict(data)
            raw_response["delivery_verified"] = verified is not None
            if verified is not None:
                raw_response["delivery_readback"] = verified
            return SendResult(success=True, message_id=message_id, raw_response=raw_response)
        except Exception as exc:
            logger.warning("Fluxer file upload failed: %s", exc)
            return _failed_send_result(exc)

    async def _verify_delivery(
        self,
        chat_id: str,
        message_id: Optional[str],
        *,
        expected_content: Optional[str] = None,
        expected_attachment_count: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read back a sent/edited message and validate the visible payload.

        Verification is deliberately non-fatal for the caller: a REST timeout or
        stale read after a successful POST may still mean the message landed,
        and retrying could duplicate messages. We attach verification state to
        ``raw_response`` and log mismatches for diagnostics.
        """
        if not self._delivery_verification_enabled or not message_id:
            return None
        try:
            data = await self._request("GET", f"/channels/{_quote_id(chat_id)}/messages/{_quote_id(message_id)}")
        except Exception as exc:
            logger.warning("Fluxer delivery read-back failed for message %s: %s", message_id, exc)
            return None

        problems: List[str] = []
        if str(data.get("id") or "") != str(message_id):
            problems.append("id_mismatch")
        if expected_content is not None and str(data.get("content") or "") != expected_content:
            problems.append("content_mismatch")
        if expected_attachment_count is not None:
            attachments = data.get("attachments") or []
            actual_count = len(attachments) if isinstance(attachments, list) else 0
            if actual_count < expected_attachment_count:
                problems.append(f"attachment_count_mismatch:{actual_count}<{expected_attachment_count}")
        if self.bot_user_id:
            author = data.get("author") or {}
            author_id = str(author.get("id") or data.get("author_id") or "")
            if author_id and author_id != self.bot_user_id:
                problems.append("author_mismatch")
        if problems:
            logger.warning("Fluxer delivery verification warning for %s: %s", message_id, ",".join(problems))
            data = dict(data)
            data["delivery_verification_warnings"] = problems
        return data

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        try:
            data = await self._request("GET", f"/channels/{_quote_id(chat_id)}")
            return {
                "id": str(data.get("id") or chat_id),
                "name": data.get("name") or str(chat_id),
                "type": _chat_type(data.get("type")),
                "raw": data,
            }
        except Exception:
            return {"id": str(chat_id), "name": str(chat_id), "type": "channel"}

    def _remember_mentioned_thread(self, thread_key: str) -> None:
        """Track mentioned threads in insertion order and evict oldest entries."""
        if not thread_key:
            return
        self._mentioned_threads[thread_key] = None
        self._mentioned_threads.move_to_end(thread_key)
        while len(self._mentioned_threads) > self._mentioned_threads_max:
            self._mentioned_threads.popitem(last=False)

    def _remember_message_id(self, message_id: str) -> bool:
        """Return False for duplicate message IDs; otherwise remember newest ID."""
        if not message_id:
            return True
        if message_id in self._seen_message_ids:
            self._seen_message_ids.move_to_end(message_id)
            return False
        self._seen_message_ids[message_id] = None
        while len(self._seen_message_ids) > 1000:
            self._seen_message_ids.popitem(last=False)
        return True

    def _register_component_actions(
        self,
        actions: List[tuple[str, str]],
        *,
        kind: str,
        message_id: str,
        channel_id: str,
        session_key: str,
        confirm_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> None:
        """Register Fluxer native button custom IDs for a pending prompt."""
        for custom_id, choice in actions:
            state: Dict[str, Any] = {
                "kind": kind,
                "message_id": message_id,
                "session_key": session_key,
                "channel_id": channel_id,
                "choice": choice,
                "created_at": time.time(),
            }
            if confirm_id is not None:
                state["confirm_id"] = confirm_id
            if request_id is not None:
                state["request_id"] = request_id
            self._pending_component_actions[custom_id] = state
        self._prune_pending_cards()

    def _prune_pending_cards(self) -> None:
        """Drop pending approval/confirm cards past the host's approval TTL.

        A card left registered after its gateway request can no longer be answered
        (turn ended, request timed out) must not linger: it would keep a stale button
        live and, worse, a later tap could resolve an unrelated pending request. This
        only prunes the adapter's own maps — it never resolves or denies anything, so
        the host's own approval timeout remains the sole authority on the request.
        """
        ttl = _approval_card_ttl()
        now = time.time()
        for mapping in (
            self._pending_exec_approvals,
            self._pending_component_actions,
            self._pending_reaction_actions,
        ):
            for key, state in list(mapping.items()):
                if not isinstance(state, dict):
                    continue
                created = state.get("created_at")
                try:
                    created_at = float(created) if created is not None else None
                except (TypeError, ValueError):
                    created_at = None
                if created_at is not None and now - created_at > ttl:
                    mapping.pop(key, None)

    async def _post_message_with_optional_components(
        self,
        channel_id: str,
        content: str,
        components: List[Dict[str, Any]],
    ) -> tuple[Dict[str, Any], bool]:
        """POST a message with components, falling back only on explicit unsupported-component responses."""
        payload = self._outbound_message_payload(self.format_message(content))
        payload["components"] = components
        try:
            data = await self._request(
                "POST",
                f"/channels/{_quote_id(channel_id)}/messages",
                json=payload,
            )
            return data, True
        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code not in {400, 404, 415, 422}:
                logger.warning("Fluxer component message send failed without safe fallback: %s", exc)
                raise
            logger.debug("Fluxer components unsupported by deployment; retrying without components: %s", exc)
            data = await self._request(
                "POST",
                f"/channels/{_quote_id(channel_id)}/messages",
                json=self._outbound_message_payload(self.format_message(content)),
            )
            return data, False

    @staticmethod
    def _rate_limit_delay(response: Any, attempt: int) -> float:
        """Resolve Fluxer's 429 retry window from headers or JSON body."""
        candidates: list[Any] = [response.headers.get("Retry-After")]
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            candidates.extend((body.get("retry_after"), body.get("retryAfter")))
        for candidate in candidates:
            try:
                value = float(candidate)
            except (TypeError, ValueError):
                continue
            if value >= 0:
                return min(_REST_RATE_LIMIT_MAX_DELAY, max(0.05, value))
        return min(_REST_RATE_LIMIT_MAX_DELAY, float(2**attempt))

    async def _request(self, method: str, path: str, *, warn_on_error: bool = True, **kwargs) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        async with httpx.AsyncClient(timeout=20) as client:
            response = None
            for attempt in range(_REST_RATE_LIMIT_MAX_ATTEMPTS):
                response = await client.request(method, url, headers=_headers(self.bot_token), **kwargs)
                if response.status_code != 429 or attempt + 1 >= _REST_RATE_LIMIT_MAX_ATTEMPTS:
                    break
                delay = self._rate_limit_delay(response, attempt)
                # Paths can contain deployment-specific identifiers. Keep them out
                # of logs just as we keep the Authorization header out of logs.
                logger.info("Fluxer REST %s rate limited; retrying in %.3fs", method, delay)
                await asyncio.sleep(delay)
            assert response is not None
            if response.status_code >= 400 and warn_on_error:
                logger.warning(
                    "Fluxer REST %s failed: status=%s body=%s",
                    method,
                    response.status_code,
                    _redact_fluxer_error_body(response.text, self.bot_token),
                )
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()

    async def _multipart_request(
        self,
        method: str,
        path: str,
        *,
        payload: Dict[str, Any],
        files: List[tuple[str, Path, str]],
    ) -> Dict[str, Any]:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for Fluxer adapter") from exc

        url = urljoin(self.api_base_url + "/", path.lstrip("/"))
        multipart_files = []
        handles = []
        try:
            for field_name, file_path, filename in files:
                handle = file_path.open("rb")
                handles.append(handle)
                content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
                multipart_files.append((field_name, (filename, handle, content_type)))
            data = {"payload_json": json.dumps(payload, separators=(",", ":"))}
            async with httpx.AsyncClient(timeout=60) as client:
                response = None
                for attempt in range(_REST_RATE_LIMIT_MAX_ATTEMPTS):
                    if attempt:
                        for handle in handles:
                            handle.seek(0)
                    response = await client.request(
                        method,
                        url,
                        headers=_auth_headers(self.bot_token),
                        data=data,
                        files=multipart_files,
                    )
                    if response.status_code != 429 or attempt + 1 >= _REST_RATE_LIMIT_MAX_ATTEMPTS:
                        break
                    delay = self._rate_limit_delay(response, attempt)
                    logger.info("Fluxer REST %s %s rate limited; retrying upload in %.3fs", method, path, delay)
                    await asyncio.sleep(delay)
                assert response is not None
                response.raise_for_status()
                if not response.content:
                    return {}
                return response.json()
        finally:
            for handle in handles:
                try:
                    handle.close()
                except Exception:
                    pass

    async def _download_attachment_bytes(self, url: str) -> bytes:
        """Fetch attachment bytes with SSRF, redirect and size protection.

        A pre-flight ``is_safe_url`` alone is not enough: the previous version then
        used a plain ``httpx.AsyncClient(follow_redirects=True)`` and returned
        ``response.content`` unbounded, so (a) a public URL that redirected to a
        private/internal address was followed without re-validation, (b) DNS could
        be rebound between the check and the connect, and (c) a hostile or merely
        huge body was buffered fully into memory. Now the host's connect-time
        SSRF-safe client is used, every redirect is re-validated (manually, so it
        holds even on a host without the guard) and the body is read under the
        host's inbound-media cap.
        """
        try:
            import httpx  # noqa: F401
            from tools.url_safety import is_safe_url
        except ImportError as exc:
            raise RuntimeError("httpx and url safety helpers are required for Fluxer attachments") from exc

        if not is_safe_url(url):
            raise ValueError(f"Blocked unsafe Fluxer attachment URL: {safe_url_for_log(url)}")
        headers = {"User-Agent": "Hermes-Fluxer/0.3", "Accept": "*/*"}
        attachment_host = urlparse(url).netloc.lower()
        api_host = urlparse(self.api_base_url).netloc.lower()
        if attachment_host and api_host and attachment_host == api_host:
            headers["Authorization"] = f"Bot {self.bot_token}"

        max_bytes = _attachment_max_bytes()
        factory, _guarded = _ssrf_safe_async_client_factory()
        # ``follow_redirects`` stays OFF: redirects are walked here so each hop is
        # re-validated against ``is_safe_url`` before it is followed.
        async with factory(
            timeout=30.0,
            follow_redirects=False,
            event_hooks={"response": [_fluxer_ssrf_redirect_guard]},
        ) as client:
            download_url = url
            redirect_count = 0
            while True:
                async with client.stream("GET", download_url, headers=headers) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        target = _redirect_target(response)
                        if not target:
                            response.raise_for_status()
                            raise ValueError("Fluxer attachment redirect without a target")
                        if not is_safe_url(target):
                            raise ValueError(
                                f"Blocked unsafe Fluxer attachment redirect: {safe_url_for_log(target)}"
                            )
                        if redirect_count >= _MAX_ATTACHMENT_REDIRECTS:
                            raise ValueError("Too many Fluxer attachment redirects")
                        download_url = target
                        redirect_count += 1
                        continue
                    response.raise_for_status()
                    return await _read_capped_body(response, max_bytes)

    async def _cache_attachment(self, att: Dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
        url = _attachment_url(att)
        if not url:
            return None, None
        try:
            from tools.url_safety import is_safe_url
        except ImportError as exc:
            raise RuntimeError("url safety helpers are required for Fluxer attachments") from exc
        if not is_safe_url(url):
            logger.warning("Fluxer blocked unsafe attachment URL: %s", safe_url_for_log(url))
            return None, None
        content_type = _attachment_content_type(att)
        filename = _attachment_filename(att)
        ext = _extension_for_attachment(att, content_type)

        try:
            if content_type.startswith("image/") or ext in SUPPORTED_IMAGE_DOCUMENT_TYPES:
                image_type = content_type if content_type.startswith("image/") else SUPPORTED_IMAGE_DOCUMENT_TYPES.get(ext, "image/jpeg")
                image_ext = ext if ext in SUPPORTED_IMAGE_DOCUMENT_TYPES else _extension_for_attachment(att, image_type, ".jpg")
                return await cache_image_from_url(url, image_ext), image_type
            if content_type.startswith("audio/"):
                return await cache_audio_from_url(url, ext if ext != ".bin" else ".ogg"), content_type

            data = await self._download_attachment_bytes(url)
            return cache_document_from_bytes(data, filename), content_type or "application/octet-stream"
        except Exception as exc:
            logger.warning("Fluxer failed to cache attachment %s; dropping attachment: %s", filename, exc)
            return None, None

    async def _extract_attachments(self, data: Dict[str, Any]) -> tuple[List[str], List[str]]:
        media_urls: List[str] = []
        media_types: List[str] = []
        attachments = data.get("attachments") or []
        if not isinstance(attachments, list):
            return media_urls, media_types
        for att in attachments:
            if not isinstance(att, dict):
                continue
            cached, mtype = await self._cache_attachment(att)
            if cached:
                media_urls.append(cached)
                media_types.append(mtype or "application/octet-stream")
        return media_urls, media_types

    async def _resolve_reply_context(
        self,
        data: Dict[str, Any],
        channel_id: str,
    ) -> tuple[Optional[str], Optional[str]]:
        reply_to_message_id, referenced = _extract_reply_message(data)
        if not reply_to_message_id:
            return None, None

        reply_to_text = _message_content(referenced)
        if reply_to_text:
            return reply_to_message_id, reply_to_text

        # Some gateway events only include ``message_reference``. One REST
        # lookup recovers the text for context injection; failures are non-fatal
        # because the triggering message itself is still valid.
        try:
            payload = await self._request("GET", f"/channels/{_quote_id(channel_id)}/messages/{_quote_id(reply_to_message_id)}")
            reply_to_text = _message_content(payload) or None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer reply context lookup failed for %s: %s", reply_to_message_id, exc)
            reply_to_text = None
        return reply_to_message_id, reply_to_text

    def _message_mentions_bot(self, data: Dict[str, Any], text: str) -> bool:
        if self.bot_user_id:
            uid = re.escape(str(self.bot_user_id))
            if re.search(rf"<@!?{uid}>", text or ""):
                return True
            mentions = data.get("mentions") or []
            if isinstance(mentions, list):
                for mention in mentions:
                    if isinstance(mention, dict) and str(mention.get("id") or "") == str(self.bot_user_id):
                        return True
                    if str(mention) == str(self.bot_user_id):
                        return True

        # Human-friendly direct-address fallback for Fluxer instances that do
        # not expose bot mention markup yet. Keep it anchored to the start so
        # "Hermes said..." in the middle of a discussion doesn't wake the bot.
        default_patterns = (r"^\s*@?hermes\b[,:]?",)
        for pattern in (*default_patterns, *self._mention_patterns):
            try:
                if re.search(pattern, text or "", flags=re.IGNORECASE):
                    return True
            except re.error:
                logger.debug("Ignoring invalid Fluxer mention pattern: %s", pattern)
        return False

    def _strip_bot_mention(self, text: str) -> str:
        stripped = text or ""
        if self.bot_user_id:
            stripped = re.sub(rf"<@!?{re.escape(str(self.bot_user_id))}>\s*", "", stripped).strip()
        stripped = re.sub(r"^\s*@?hermes\b[,:]?\s*", "", stripped, flags=re.IGNORECASE)
        for pattern in self._mention_patterns:
            try:
                stripped = re.sub(pattern, "", stripped, count=1, flags=re.IGNORECASE).strip()
            except re.error:
                pass
        return stripped

    def _should_process_message(
        self,
        *,
        channel_id: str,
        chat_type: str,
        text: str,
        data: Dict[str, Any],
        reply_to_message_id: Optional[str],
    ) -> tuple[bool, str]:
        if chat_type == "dm":
            return True, text
        if self._allowed_channel_ids and channel_id not in self._allowed_channel_ids:
            # A thread this adapter created inherits its parent channel's
            # allowlist status, so an allowlisted parent does not silently
            # exclude the thread the bot just opened underneath it. A thread a
            # human created inherits nothing: the parent's allowlist entry was
            # never consent for a conversation the operator did not start. Only a
            # CONFIRMED bot-owned thread inherits -- an ownership that is unknown
            # (cold cache, failed lookup) fails closed rather than widening the
            # allowlist on a parent that was merely recorded alongside it.
            parent_id = self._thread_parents.get(str(channel_id))
            if self._thread_is_bot_owned(str(channel_id)) is not True:
                parent_id = None
            if not (parent_id and parent_id in self._allowed_channel_ids):
                logger.debug("Fluxer ignoring message in a channel outside the allowed set")
                return False, text
        parent_channel_id = self._thread_parents.get(str(channel_id))
        if self._thread_is_bot_owned(str(channel_id)) is not True:
            parent_channel_id = None
        if (
            channel_id in self._free_response_channels
            or channel_id in self._home_channel_ids
            or (parent_channel_id and parent_channel_id in self._free_response_channels)
        ):
            return True, text
        guild_id = str(
            data.get("guild_id")
            or (data.get("guild") or {}).get("id")
            or (data.get("channel") or {}).get("guild_id")
            or ""
        )
        if (
            self._auto_free_response_home_guild
            and guild_id
            and guild_id in self._home_guild_ids
            and channel_id not in self._mention_gated_channels
        ):
            return True, text
        if not self._require_mention:
            return True, text
        # A command typed as plain text is an address in its own right. Fluxer
        # registers no native application commands, so /new, /help, /stop and
        # /title only ever arrive this way, and a mention gate that drops them
        # makes the whole command pipeline unreachable in exactly the channels
        # the gate protects -- silently, with no reply to tell the user why.
        if _is_text_command(text):
            return True, text

        is_mentioned = self._message_mentions_bot(data, text)
        thread_key = str(data.get("thread_id") or (channel_id if chat_type == "thread" else "") or reply_to_message_id or "")
        in_mentioned_thread = bool(thread_key and thread_key in self._mentioned_threads)
        if self._strict_mention and not is_mentioned:
            return False, text
        if not is_mentioned and not in_mentioned_thread:
            return False, text
        if is_mentioned:
            text = self._strip_bot_mention(text)
            if thread_key:
                self._remember_mentioned_thread(thread_key)
        return True, text

    async def _recover_backlog(self, *, cutoff: datetime) -> None:
        if not self._backlog_enabled or not self._known_channel_ids:
            return
        cutoff = cutoff.astimezone(timezone.utc)
        for channel_id in sorted(self._known_channel_ids):
            try:
                payload = await self._request(
                    "GET",
                    f"/channels/{_quote_id(channel_id)}/messages",
                    params={"limit": self._backlog_limit},
                )
                messages = _extract_message_list(payload)
                recovered = 0
                for message in reversed(messages):
                    ts = _parse_fluxer_timestamp(message.get("timestamp") or message.get("created_at"))
                    if ts is not None and ts < cutoff:
                        continue
                    if not message.get("channel_id"):
                        message = {**message, "channel_id": channel_id}
                    msg_id = str(message.get("id") or "")
                    if msg_id and msg_id in self._seen_message_ids:
                        continue
                    await self._handle_message_create(
                        message,
                        {"op": 0, "t": "MESSAGE_CREATE", "d": message, "backfill": True},
                    )
                    recovered += 1
                if recovered:
                    logger.info("Fluxer recovered %d backlog message(s)", recovered)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # HTTP exception strings can include request URLs or headers. The
                # exception class supplies useful context without exposing either.
                logger.warning("Fluxer backlog recovery failed (%s)", type(exc).__name__)

    async def _listen_loop(self) -> None:
        assert self._ws is not None
        reason = "closed"
        try:
            async for raw in self._ws:
                payload = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode("utf-8"))
                await self._handle_gateway_dispatch(payload)
            reason = "websocket closed"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._running:
                logger.warning("Fluxer listener stopped: %s", exc)
            reason = str(exc)
        finally:
            if not self._closing:
                self._last_disconnect_at = datetime.now(tz=timezone.utc)
                self._mark_disconnected()
                self._schedule_reconnect(reason)

    async def _send_heartbeat(self) -> None:
        if self._ws is None:
            return
        # Only the FIRST un-acked heartbeat starts the watchdog window. Resetting
        # the timestamp on every tick made the elapsed time never exceed one
        # interval, so a dead gateway was never noticed.
        if not self._awaiting_heartbeat_ack:
            self._last_heartbeat_sent_at = time.monotonic()
        self._awaiting_heartbeat_ack = True
        await self._ws.send(json.dumps({"op": 1, "d": self._last_seq}))

    async def _send_gateway_payload(self, payload: Dict[str, Any]) -> bool:
        """Send a raw Fluxer gateway payload if the websocket is available."""
        if self._ws is None:
            return False
        await self._ws.send(json.dumps(payload))
        return True

    async def _send_presence_update(self) -> None:
        """Assert bot presence via opcode 3 after a successful READY.

        Identify already carries the initial presence; this re-sends it on
        every session so reconnects cannot leave the bot stuck as offline.
        Failure is logged and tolerated — presence is best-effort.
        """
        payload = _build_presence_update_payload(
            self._presence_status,
            afk=self._presence_afk,
        )
        try:
            success = await self._send_gateway_payload(payload)
            if success:
                logger.info("Fluxer presence update sent")
            else:
                logger.debug("Fluxer presence update skipped: websocket unavailable")
        except Exception as exc:
            logger.warning("Fluxer presence update failed (continuing): %s", type(exc).__name__)

    async def wait_until_gateway_ready(self, timeout: float = 10.0) -> bool:
        """Wait until Fluxer gateway identification has completed with READY."""
        if self._gateway_ready_event.is_set():
            return True
        try:
            await asyncio.wait_for(self._gateway_ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def send_voice_state_update(
        self,
        channel_id: Optional[str],
        *,
        guild_id: Optional[str] = None,
        connection_id: Optional[str] = None,
        self_mute: bool = False,
        self_deaf: bool = True,
        self_video: bool = False,
        self_stream: bool = False,
        viewer_stream_keys: Optional[List[str]] = None,
    ) -> bool:
        """Ask Fluxer to join/update/leave a voice channel on the main gateway.

        This is the first half of Fluxer's LiveKit flow. A successful send should
        be followed by gateway VOICE_SERVER_UPDATE containing endpoint/token;
        the future realtime bridge can then connect to LiveKit and publish audio.
        """
        payload = _build_voice_state_update_payload(
            channel_id=channel_id,
            guild_id=guild_id,
            connection_id=connection_id,
            self_mute=self_mute,
            self_deaf=self_deaf,
            self_video=self_video,
            self_stream=self_stream,
            viewer_stream_keys=viewer_stream_keys,
        )
        sent = await self._send_gateway_payload(payload)
        if sent and channel_id:
            self._pending_voice_joins[_voice_join_key(guild_id, channel_id)] = {
                "guild_id": guild_id,
                "channel_id": channel_id,
                "connection_id": connection_id,
            }
        elif sent and channel_id is None:
            stale_join_keys = [
                key
                for key, pending in self._pending_voice_joins.items()
                if guild_id is None or pending.get("guild_id") == guild_id
            ]
            for key in stale_join_keys:
                self._pending_voice_joins.pop(key, None)
        return sent

    async def _heartbeat_loop(self, interval_ms: int) -> None:
        interval_s = max(interval_ms, 1000) / 1000
        ack_timeout_s = interval_s * _HEARTBEAT_ACK_TIMEOUT_FACTOR
        try:
            while not self._closing and self._ws is not None:
                await asyncio.sleep(interval_s)
                if (
                    self._awaiting_heartbeat_ack
                    and self._last_heartbeat_sent_at is not None
                    and time.monotonic() - self._last_heartbeat_sent_at > ack_timeout_s
                ):
                    logger.warning("Fluxer heartbeat ACK timed out; closing websocket to trigger reconnect")
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    return
                await self._send_heartbeat()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Fluxer heartbeat stopped: %s", exc)

    def _interaction_user_id(self, data: Dict[str, Any]) -> str:
        user = data.get("user") or ((data.get("member") or {}).get("user") or {})
        return str(user.get("id") or "")

    async def _respond_to_interaction(self, data: Dict[str, Any], *, content: str, components: Optional[List[Dict[str, Any]]] = None) -> None:
        interaction_id = str(data.get("id") or "")
        token = str(data.get("token") or "")
        if not interaction_id or not token:
            return
        payload: Dict[str, Any] = {"type": 7, "data": self._outbound_message_payload(content)}
        if components is not None:
            payload["data"]["components"] = components
        await self._request("POST", f"/interactions/{_quote_id(interaction_id)}/{_quote_id(token)}/callback", json=payload)

    async def _handle_interaction_create(self, data: Dict[str, Any]) -> None:
        interaction_type = data.get("type")
        interaction_data = data.get("data") or {}
        if interaction_type == 3 or interaction_data.get("custom_id"):
            await self._handle_component_interaction(data)
            return
        if interaction_type == 2:
            await self._handle_application_command_interaction(data)

    async def _handle_reaction_add(self, data: Dict[str, Any]) -> None:
        message_id = str(data.get("message_id") or ((data.get("message") or {}).get("id")) or "")
        channel_id = str(data.get("channel_id") or ((data.get("channel") or {}).get("id")) or "")
        emoji = _reaction_emoji_from_event(data)
        if not message_id or not emoji:
            return
        action = self._pending_reaction_actions.get(f"{message_id}:{emoji}")
        if not action:
            return
        user = data.get("user") or ((data.get("member") or {}).get("user") or {})
        user_id = str(data.get("user_id") or user.get("id") or "")
        if user.get("bot") or (self.bot_user_id and user_id == str(self.bot_user_id)):
            return
        if not self._interaction_user_allowed(user_id):
            logger.info("Fluxer reaction action ignored for unauthorized user %s", user_id or "<missing>")
            return
        action_channel_id = str(action.get("channel_id") or "")
        if action_channel_id and channel_id and channel_id != action_channel_id:
            logger.warning("Fluxer reaction action ignored for mismatched channel %s", channel_id)
            return
        if action.get("kind") == "exec_approval":
            await self._resolve_exec_approval_action(action, user_id=user_id)
        elif action.get("kind") == "slash_confirm":
            await self._resolve_slash_confirm_action(action, user_id=user_id)

    async def _handle_component_interaction(self, data: Dict[str, Any]) -> None:
        interaction_data = data.get("data") or {}
        custom_id = str(interaction_data.get("custom_id") or "")
        action = self._pending_component_actions.get(custom_id)
        if not action:
            return
        user_id = self._interaction_user_id(data)
        if not self._interaction_user_allowed(user_id):
            logger.info("Fluxer component action ignored for unauthorized user %s", user_id or "<missing>")
            return
        interaction_message_id = str(((data.get("message") or {}).get("id")) or "")
        action_message_id = str(action.get("message_id") or "")
        if action_message_id and interaction_message_id and interaction_message_id != action_message_id:
            logger.warning("Fluxer component action %s ignored for mismatched message %s", custom_id, interaction_message_id)
            return
        interaction_channel_id = str(data.get("channel_id") or ((data.get("channel") or {}).get("id")) or "")
        action_channel_id = str(action.get("channel_id") or "")
        if action_channel_id and interaction_channel_id and interaction_channel_id != action_channel_id:
            logger.warning("Fluxer component action %s ignored for mismatched channel %s", custom_id, interaction_channel_id)
            return
        self._pending_component_actions.pop(custom_id, None)
        if action.get("kind") == "exec_approval":
            await self._resolve_exec_approval_action(action, user_id=user_id, interaction=data)
        elif action.get("kind") == "slash_confirm":
            await self._resolve_slash_confirm_action(action, user_id=user_id, interaction=data)

    async def _resolve_exec_approval_action(self, action: Dict[str, Any], *, user_id: str, interaction: Optional[Dict[str, Any]] = None) -> None:
        message_id = str(action.get("message_id") or (((interaction or {}).get("message") or {}).get("id")) or "")
        pending = self._pending_exec_approvals.pop(message_id, None)
        if not pending or pending.get("resolved"):
            return
        pending["resolved"] = True
        session_key = str(action.get("session_key") or pending.get("session_key") or "")
        choice = str(action.get("choice") or "deny")
        # Bind the resolution to the exact gateway request this card belongs to.
        # ``request_id`` is captured when the card was posted; the action carries it
        # too. Passing it to ``resolve_gateway_approval`` resolves THAT request only.
        # When no id could be determined, resolution falls back to the host's FIFO
        # order — the pre-existing behaviour — and never to a guessed id, so a card
        # can never approve a request it does not belong to.
        request_id = str(action.get("request_id") or pending.get("request_id") or "") or None
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, choice, request_id=request_id)
            logger.info(
                "Fluxer approval action resolved %d approval(s) for session %s (choice=%s request_id=%s user=%s)",
                count,
                session_key,
                choice,
                request_id or "<fifo>",
                user_id,
            )
            if request_id and count == 0:
                # The card was live but its request is gone (already answered, timed
                # out, or withdrawn). Do not fall back to FIFO: that could resolve an
                # unrelated pending request for the same session.
                logger.info(
                    "Fluxer approval card %s: request %s no longer pending; not falling back to FIFO",
                    message_id,
                    request_id,
                )
        except Exception as exc:
            logger.error("Fluxer component approval resolve failed: %s", exc)
        for cid, state in list(self._pending_component_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_component_actions.pop(cid, None)
        for rid, state in list(self._pending_reaction_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_reaction_actions.pop(rid, None)
        label = {"once": "approved once", "session": "approved for session", "always": "approved permanently", "deny": "denied"}.get(choice, choice)
        resolved_content = f"{pending.get('content') or 'Command approval required'}\n\nResolved: {label} by <@{user_id}>."
        try:
            if interaction:
                await self._respond_to_interaction(interaction, content=resolved_content, components=[])
            else:
                await self.edit_message(str(pending.get("channel_id") or action.get("channel_id") or ""), message_id, resolved_content)
        except Exception as exc:
            logger.debug("Fluxer approval resolution response failed: %s", exc)

    async def _resolve_slash_confirm_action(self, action: Dict[str, Any], *, user_id: str, interaction: Optional[Dict[str, Any]] = None) -> None:
        session_key = str(action.get("session_key") or "")
        confirm_id = str(action.get("confirm_id") or "")
        choice = str(action.get("choice") or "cancel")
        try:
            from tools.slash_confirm import resolve

            await resolve(session_key, confirm_id, choice)
        except Exception as exc:
            logger.error("Fluxer slash confirm resolve failed: %s", exc)
        message_id = str(action.get("message_id") or (((interaction or {}).get("message") or {}).get("id")) or "")
        for rid, state in list(self._pending_reaction_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_reaction_actions.pop(rid, None)
        try:
            content = f"Slash confirmation: {choice} by <@{user_id}>."
            if interaction:
                await self._respond_to_interaction(interaction, content=content, components=[])
            elif message_id:
                await self.edit_message(str(action.get("channel_id") or ""), message_id, content)
        except Exception:
            pass

    async def _handle_application_command_interaction(self, data: Dict[str, Any]) -> None:
        interaction_data = data.get("data") or {}
        name = str(interaction_data.get("name") or "").strip()
        if not name:
            return
        options = interaction_data.get("options") or []
        parts = [f"/{name}"]
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict) and option.get("value") is not None:
                    parts.append(str(option.get("value")))
        text = " ".join(parts).strip()
        user = data.get("user") or ((data.get("member") or {}).get("user") or {})
        user_id = str(user.get("id") or "")
        if not self._interaction_user_allowed(user_id):
            logger.info("Fluxer ignoring application command from non-allowed user %s", user_id)
            try:
                await self._request(
                    "POST",
                    f"/interactions/{_quote_id(data.get('id'))}/{_quote_id(data.get('token'))}/callback",
                    json={"type": 4, "data": {"content": "You are not allowed to use this bot.", "flags": 64}},
                    warn_on_error=False,
                )
            except Exception as exc:
                logger.debug("Fluxer unauthorized application-command response failed: %s", exc)
            return
        channel = data.get("channel") or {}
        channel_id = str(data.get("channel_id") or channel.get("id") or "")
        if not channel_id:
            return
        # The native path is routed through the SAME contract as a text message, so
        # a native command cannot key a different container than the message path
        # (Fluxer registers no native application commands on this deployment, so
        # the path is cold -- but a second, divergent routing contract would grow
        # here otherwise).
        interaction_chat_type = _chat_type(data.get("channel_type") or channel.get("type"))
        if not (data.get("channel_type") or channel.get("type")):
            known_kind = self._classify_untyped_event(channel_id)
            if known_kind is None:
                known_kind = await self._resolve_channel_kind(channel_id)
            if known_kind:
                interaction_chat_type = known_kind
        # A native command is a command: it addresses the turn already running in
        # this chat, so it keeps the channel's session. Presenting the resolved text
        # to the resolver lets the SAME command rule that guards the text path apply
        # here, instead of a second rule that would drift from it.
        source_chat_id, source_chat_type, parent_chat_id, pending_parent = await self._resolve_thread_source(
            {**data, "content": text, "id": ""}, channel_id, interaction_chat_type
        )
        if pending_parent:
            self._threads_pending[source_chat_id] = pending_parent
        source = self.build_source(
            chat_id=source_chat_id,
            chat_name=channel.get("name"),
            chat_type=source_chat_type,
            user_id=user_id or None,
            user_name=_author_name(user),
            guild_id=data.get("guild_id"),
            parent_chat_id=parent_chat_id,
            message_id=str(data.get("id") or "") or None,
        )
        try:
            await self._request(
                "POST",
                f"/interactions/{_quote_id(data.get('id'))}/{_quote_id(data.get('token'))}/callback",
                json={"type": 5, "data": {"flags": 64}},
                warn_on_error=False,
            )
        except Exception as exc:
            logger.debug("Fluxer application-command defer response failed: %s", exc)
        await self.handle_message(MessageEvent(text=text, message_type=MessageType.TEXT, source=source, raw_message={"interaction": data}, message_id=str(data.get("id") or "") or None))

    def set_voice_server_update_handler(
        self,
        handler: Optional[Callable[[Dict[str, Any], Dict[str, Any]], Any]],
    ) -> None:
        """Register an in-memory LiveKit bridge handoff for VOICE_SERVER_UPDATE.

        The handler receives `(raw_update, safe_update)`. `raw_update` may contain
        the ephemeral LiveKit token and must not be stored by the adapter.
        """
        self._voice_server_update_handler = handler

    def set_voice_state_update_handler(
        self,
        handler: Optional[Callable[[Dict[str, Any]], Any]],
    ) -> None:
        """Register a callback for user VOICE_STATE_UPDATE events.

        The handler receives the raw, non-token voice-state payload so a local
        supervisor can react to a target user joining/leaving voice channels.
        """
        self._voice_state_update_handler = handler

    async def _handle_voice_state_update(self, data: Dict[str, Any]) -> None:
        if self._voice_state_update_handler is None:
            return
        try:
            result = self._voice_state_update_handler(dict(data))
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            logger.exception("Fluxer voice-state update handler failed")

    async def _handle_voice_server_update(self, data: Dict[str, Any]) -> None:
        """Capture Fluxer LiveKit server metadata without retaining the token."""
        guild_id = data.get("guild_id")
        channel_id = data.get("channel_id")
        key = _voice_join_key(
            str(guild_id) if guild_id is not None else None,
            str(channel_id) if channel_id is not None else None,
        )
        pending = self._pending_voice_joins.pop(key, None)
        matched = pending is not None
        safe_update = _sanitize_voice_server_update(data, matched_pending_join=matched)
        self._last_voice_server_update = safe_update
        logger.info(
            "Fluxer voice server update captured endpoint=%s has_token=%s matched_pending_join=%s channel=%s guild=%s connection=%s",
            safe_url_for_log(str(safe_update.get("endpoint") or "")),
            safe_update.get("has_token"),
            matched,
            safe_update.get("channel_id") or "<none>",
            safe_update.get("guild_id") or "<dm>",
            safe_update.get("connection_id") or "<none>",
        )
        if matched and self._voice_server_update_handler is not None:
            try:
                result = self._voice_server_update_handler(dict(data), dict(safe_update))
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Fluxer voice server update bridge handler failed channel=%s guild=%s connection=%s",
                    safe_update.get("channel_id") or "<none>",
                    safe_update.get("guild_id") or "<dm>",
                    safe_update.get("connection_id") or "<none>",
                )

    async def _handle_gateway_dispatch(self, payload: Dict[str, Any]) -> None:
        op = payload.get("op")
        self._last_seq = _event_seq(payload) or self._last_seq

        if op == 10:  # HELLO
            interval = int(((payload.get("d") or {}).get("heartbeat_interval") or 41250))
            if self._ws is not None:
                await self._ws.send(
                    json.dumps(
                        _build_identify_payload(
                            self.bot_token,
                            presence_status=self._presence_status,
                            presence_afk=self._presence_afk,
                        )
                    )
                )
                # Fluxer's hosted gateway also tolerates/expects an
                # immediate heartbeat; waiting a full interval can trip hosted
                # gateway 4009 heartbeat-timeout closes during dogfood sessions.
                await self._send_heartbeat()
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop(interval), name="fluxer-heartbeat")
            return
        if op == 1:  # server heartbeat request
            await self._send_heartbeat()
            return
        if op == 11:  # HEARTBEAT_ACK
            self._awaiting_heartbeat_ack = False
            self._last_heartbeat_ack_at = time.monotonic()
            return
        if op in {7, 9, 12}:  # RECONNECT / INVALID_SESSION / GATEWAY_ERROR
            logger.warning("Fluxer gateway requested reconnect: op=%s payload=%s", op, payload.get("d"))
            if self._ws is not None:
                await self._ws.close()
            return
        if op != 0:  # not DISPATCH
            return

        event_name = payload.get("t")
        data = payload.get("d") or {}
        if event_name == "READY":
            user = data.get("user") or data.get("bot") or {}
            if user.get("id"):
                self.bot_user_id = str(user["id"])
            self._gateway_ready_event.set()
            await self._send_presence_update()
            return
        if event_name == "INTERACTION_CREATE":
            await self._handle_interaction_create(data)
            return
        if event_name == "VOICE_SERVER_UPDATE":
            await self._handle_voice_server_update(data)
            return
        if event_name == "VOICE_STATE_UPDATE":
            await self._handle_voice_state_update(data)
            return
        if event_name in {"MESSAGE_REACTION_ADD", "REACTION_ADD", "MESSAGE_REACTION_CREATE"}:
            await self._handle_reaction_add(data)
            return
        if event_name in {"THREAD_CREATE", "THREAD_UPDATE", "CHANNEL_CREATE", "CHANNEL_UPDATE"}:
            channel_id = str(data.get("id") or "")
            if channel_id:
                self._known_channel_ids.add(channel_id)
            return
        if event_name == "MESSAGE_DELETE":
            await self._handle_message_delete(data)
            return
        if event_name == "MESSAGE_UPDATE":
            await self._handle_message_update(data)
            return
        if event_name != "MESSAGE_CREATE":
            return
        await self._handle_message_create(data, payload)

    async def _handle_message_update(self, data: Dict[str, Any]) -> None:
        """Track Fluxer edits that affect adapter-owned state.

        We intentionally do not re-dispatch edited user messages as fresh user
        turns; that would surprise users and can duplicate agent runs. For now
        MESSAGE_UPDATE only keeps pending approval bookkeeping in sync so a
        later reaction edits the current prompt text rather than stale content.
        """
        message_id = str(data.get("id") or data.get("message_id") or "")
        if not message_id:
            return
        pending = self._pending_exec_approvals.get(message_id)
        if pending is None:
            return
        content = data.get("content")
        if isinstance(content, str) and content:
            pending["content"] = content

    async def _handle_message_delete(self, data: Dict[str, Any]) -> None:
        """Fail closed if an outstanding approval or slash-confirm prompt is deleted."""
        message_id = str(data.get("id") or data.get("message_id") or "")
        if not message_id:
            return
        self._seen_message_ids.pop(message_id, None)
        for chat_id, latest_id in list(self._last_inbound_by_chat.items()):
            if latest_id == message_id:
                self._last_inbound_by_chat.pop(chat_id, None)
        pending = self._pending_exec_approvals.pop(message_id, None)
        slash_cancel_action: Optional[Dict[str, Any]] = None
        for cid, state in list(self._pending_component_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_component_actions.pop(cid, None)
                if slash_cancel_action is None and state.get("kind") == "slash_confirm":
                    slash_cancel_action = state
        for rid, state in list(self._pending_reaction_actions.items()):
            if state.get("message_id") == message_id:
                self._pending_reaction_actions.pop(rid, None)
                if slash_cancel_action is None and state.get("kind") == "slash_confirm":
                    slash_cancel_action = state
        if slash_cancel_action:
            try:
                from tools.slash_confirm import resolve

                await resolve(
                    str(slash_cancel_action.get("session_key") or ""),
                    str(slash_cancel_action.get("confirm_id") or ""),
                    "cancel",
                )
                logger.info("Fluxer deleted slash-confirm prompt cancelled confirm %s", slash_cancel_action.get("confirm_id"))
            except Exception as exc:
                logger.error("Fluxer deleted slash-confirm prompt cancel failed: %s", exc)
        if not pending or pending.get("resolved"):
            return
        pending["resolved"] = True
        session_key = str(pending.get("session_key") or "")
        request_id = str(pending.get("request_id") or "") or None
        try:
            from tools.approval import resolve_gateway_approval

            count = resolve_gateway_approval(session_key, "deny", request_id=request_id)
            logger.info(
                "Fluxer deleted approval prompt denied %d approval(s) for session %s (request_id=%s)",
                count,
                session_key,
                request_id or "<fifo>",
            )
        except Exception as exc:
            logger.error("Fluxer deleted approval prompt deny failed: %s", exc)

    async def _handle_message_create(self, data: Dict[str, Any], raw_payload: Dict[str, Any]) -> None:
        msg_id = str(data.get("id") or "")
        if msg_id and not self._remember_message_id(msg_id):
            return

        author = data.get("author") or data.get("user") or {}
        author_id = str(author.get("id") or data.get("author_id") or "")
        if author.get("bot") or (self.bot_user_id and author_id == str(self.bot_user_id)):
            return
        if not self._interaction_user_allowed(author_id):
            logger.info("Fluxer ignoring message from non-allowed user %s", author_id)
            return

        text = data.get("content") or ""
        # Attachment BYTES are not fetched here. A message's admission is decided
        # by its channel and mention, and fetching before that gate meant every
        # dropped message still triggered outbound network I/O (and, on a
        # mention-gated channel, an unauthenticated fetch of a stranger's link).
        # A media-only message stays eligible using metadata alone; the download
        # happens after the gate below.
        has_attachments = _message_has_attachments(data)
        if not text and not has_attachments:
            return

        channel_id = str(data.get("channel_id") or (data.get("channel") or {}).get("id") or "")
        if not channel_id:
            return
        self._known_channel_ids.add(channel_id)
        reply_to_message_id, reply_to_text = await self._resolve_reply_context(data, channel_id)
        chat_type = _chat_type(data.get("channel_type") or (data.get("channel") or {}).get("type"))
        # Fluxer's own MESSAGE_CREATE carries no ``channel_type`` and no nested
        # channel object -- only ``channel_id`` -- so a raw event classified itself
        # as a plain channel even when it arrived inside a thread. That keyed an
        # in-thread follow-up to a different session than the turn that opened the
        # thread, which is the split this routing contract exists to remove. The
        # kind is settled from what the adapter already knows, then from one API
        # read; a lookup that fails leaves the event exactly as it arrived.
        untyped = not (data.get("channel_type") or (data.get("channel") or {}).get("type"))
        if untyped:
            known_kind = self._classify_untyped_event(channel_id)
            if known_kind is None:
                known_kind = await self._resolve_channel_kind(channel_id)
            if known_kind:
                chat_type = known_kind
        # A follow-up in a thread the bot opened has to be admitted by the same rules
        # as the channel it hangs under, so the parent/owner are resolved BEFORE the
        # admission gate rather than after it. A cold process has neither in memory;
        # one lookup restores both, and it is cached for the rest of the process.
        if chat_type == "thread" and self._thread_parents.get(str(channel_id)) is None:
            await self._resolve_thread_info(channel_id, expect_thread=True)
        should_process, text = self._should_process_message(
            channel_id=channel_id,
            chat_type=chat_type,
            text=text,
            data=data,
            reply_to_message_id=reply_to_message_id,
        )
        if not should_process:
            return
        # Now that the message is admitted, fetch its attachment bytes. An
        # attachment that fails to cache (unsafe URL, SSRF-blocked redirect,
        # oversize body, network error) is dropped and the turn proceeds on text
        # alone, exactly as before — the move only changes WHEN the fetch happens.
        media_urls, media_types = await self._extract_attachments(data)
        # One resolver decides the session container for this message: an in-thread
        # follow-up is keyed on the thread, a channel message that will open a thread
        # is keyed on the anchor that names it. Registered only now that the message
        # is admitted, so a rejected message never leaves a pending thread behind.
        source_chat_id, source_chat_type, parent_chat_id, pending_parent = await self._resolve_thread_source(
            data, channel_id, chat_type
        )
        if pending_parent:
            self._threads_pending[source_chat_id] = pending_parent
        if msg_id:
            self._last_inbound_by_chat[channel_id] = msg_id
            self._last_inbound_by_chat.move_to_end(channel_id)
            while len(self._last_inbound_by_chat) > 1000:
                self._last_inbound_by_chat.popitem(last=False)
            # Keep the trigger text so the thread this turn opens can be named
            # from the user's own words (mention already stripped above), and the
            # trigger's author so the title is read from that participant's session
            # rather than guessed out of a channel's several.
            self._inbound_text_by_message[msg_id] = text or ""
            self._inbound_text_by_message.move_to_end(msg_id)
            self._inbound_author_by_message[msg_id] = str(author_id or "")
            self._inbound_author_by_message.move_to_end(msg_id)
            while len(self._inbound_text_by_message) > self._inbound_text_by_message_max:
                self._inbound_text_by_message.popitem(last=False)
            while len(self._inbound_author_by_message) > self._inbound_text_by_message_max:
                self._inbound_author_by_message.popitem(last=False)
        source = self.build_source(
            chat_id=source_chat_id,
            chat_name=(data.get("channel") or {}).get("name"),
            chat_type=source_chat_type,
            user_id=author_id or None,
            user_name=_author_name(author),
            guild_id=data.get("guild_id"),
            parent_chat_id=parent_chat_id,
            message_id=msg_id or None,
        )

        timestamp = datetime.now(tz=timezone.utc)
        ts_raw = data.get("timestamp") or data.get("created_at")
        if isinstance(ts_raw, str):
            try:
                timestamp = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                pass

        voice_metadata = _voice_attachment_metadata(data)
        event_raw_payload = dict(raw_payload)
        if voice_metadata:
            event_raw_payload["fluxer_voice_message"] = voice_metadata

        event = MessageEvent(
            text=text,
            message_type=MessageType.VOICE if voice_metadata else _message_type_for_media(media_types),
            source=source,
            raw_message=event_raw_payload,
            message_id=msg_id or None,
            media_urls=media_urls,
            media_types=media_types,
            reply_to_message_id=reply_to_message_id,
            reply_to_text=reply_to_text,
            timestamp=timestamp,
        )
        await self.handle_message(event)


def check_requirements() -> bool:
    """Passively report whether the adapter dependencies are importable.

    Hermes calls the platform registry ``check_fn`` while loading status and
    profile-scoped configuration.  Credentials belong to ``validate_config``;
    requiring a process-global token here incorrectly disables profiles whose
    token lives only in ``PlatformConfig.extra``.
    """
    try:
        import httpx  # noqa: F401
        import websockets  # noqa: F401
    except ImportError:
        return False
    return True


def validate_config(config) -> bool:
    extra = getattr(config, "extra", {}) or {}
    token = _fluxer_env("FLUXER_BOT_TOKEN") or extra.get("bot_token", "")
    return bool(str(token).strip())


def is_connected(config) -> bool:
    return validate_config(config)


def _coerce_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value if str(item).strip())
    return str(value)


def _set_env_default(name: str, value: Any) -> None:
    if value is None or os.getenv(name):
        return
    if isinstance(value, str) and not value.strip():
        return
    os.environ[name] = _coerce_env_value(value)


def _profile_scoped_config_load() -> bool:
    """Return True while Hermes loads a multiplexed profile's configuration.

    A secondary profile must not copy its Fluxer token or authorization policy
    into process-global environment variables.  The values returned by
    ``_apply_yaml_config`` remain scoped to that profile instead.
    """
    try:
        from agent.secret_scope import is_multiplex_active
    except ImportError:
        return False
    return bool(is_multiplex_active())


def _apply_yaml_config(yaml_cfg: dict, platform_cfg: dict) -> dict | None:
    """Bridge ``fluxer:`` config.yaml keys into env/default extras.

    Hermes calls this during gateway config loading. Environment variables keep
    precedence over YAML, so operators can override container/deployment values
    without editing config.yaml.
    """

    cfg = platform_cfg if isinstance(platform_cfg, dict) else {}
    scoped_load = _profile_scoped_config_load()

    # Return adapter-consumable extras for every load.  Environment variables
    # still win in ``FluxerAdapter.__init__``; these values are the profile-safe
    # fallback and are essential when multiple Hermes profiles share a process.
    seed = dict(cfg)
    home_channel = cfg.get("home_channel")
    if home_channel and not isinstance(home_channel, dict):
        home_name = str(cfg.get("home_channel_name") or home_channel)
        seed["home_channel"] = {"chat_id": str(home_channel), "name": home_name}
    if "home_guilds" in cfg and "home_guild_ids" not in seed:
        seed["home_guild_ids"] = cfg.get("home_guilds")

    # Hermes performs a second authorization gate after the adapter.  In a
    # multiplexed profile, YAML values intentionally never become process-global
    # FLUXER_* variables, so mirror this adapter's allowlist into Hermes' generic
    # config-only policy keys.  This preserves deny-by-default while preventing
    # an authorized profile user from being rejected by the gateway's env-only
    # fallback.
    if _coerce_bool(cfg.get("allow_all_users"), False):
        seed.setdefault("allow_from", ["*"])
        seed.setdefault("group_allow_from", ["*"])
    elif cfg.get("allowed_users"):
        seed.setdefault("allow_from", cfg.get("allowed_users"))
        seed.setdefault("group_allow_from", cfg.get("allowed_users"))

    scalar_env = {
        "bot_token": "FLUXER_BOT_TOKEN",
        "base_url": "FLUXER_BASE_URL",
        "gateway_url": "FLUXER_GATEWAY_URL",
        "allowed_users": "FLUXER_ALLOWED_USERS",
        "allow_all_users": "FLUXER_ALLOW_ALL_USERS",
        "home_channel": "FLUXER_HOME_CHANNEL",
        "home_channel_name": "FLUXER_HOME_CHANNEL_NAME",
        "backlog_recovery": "FLUXER_BACKLOG_RECOVERY",
        "backlog_limit": "FLUXER_BACKLOG_LIMIT",
        "backlog_bootstrap_seconds": "FLUXER_BACKLOG_BOOTSTRAP_SECONDS",
        "require_mention": "FLUXER_REQUIRE_MENTION",
        "strict_mention": "FLUXER_STRICT_MENTION",
        "free_response_channels": "FLUXER_FREE_RESPONSE_CHANNELS",
        "home_guild_id": "FLUXER_HOME_GUILD_ID",
        "home_guilds": "FLUXER_HOME_GUILDS",
        "auto_free_response_home_guild": "FLUXER_AUTO_FREE_RESPONSE_HOME_GUILD",
        "mention_gated_channels": "FLUXER_MENTION_GATED_CHANNELS",
        "allowed_channels": "FLUXER_ALLOWED_CHANNELS",
        "mention_patterns": "FLUXER_MENTION_PATTERNS",
        "allow_mention_everyone": "FLUXER_ALLOW_MENTION_EVERYONE",
        "allow_mention_roles": "FLUXER_ALLOW_MENTION_ROLES",
        "allow_mention_users": "FLUXER_ALLOW_MENTION_USERS",
        "allow_mention_replied_user": "FLUXER_ALLOW_MENTION_REPLIED_USER",
        "delivery_verification": "FLUXER_DELIVERY_VERIFICATION",
        "register_native_commands": "FLUXER_REGISTER_NATIVE_COMMANDS",
        "application_id": "FLUXER_APPLICATION_ID",
        "native_command_guilds": "FLUXER_NATIVE_COMMAND_GUILDS",
    }
    for key, env_name in scalar_env.items():
        if not scoped_load and key in cfg:
            _set_env_default(env_name, cfg.get(key))

    voice_raw = cfg.get("voice")
    voice: dict = voice_raw if isinstance(voice_raw, dict) else {}
    voice_scalar_env = {
        "enabled": "FLUXER_VOICE_ENABLED",
        "auto_join": "FLUXER_VOICE_AUTO_JOIN",
        "target_user_ids": "FLUXER_VOICE_TARGET_USER_IDS",
        "channel_ids": "FLUXER_VOICE_CHANNEL_IDS",
        "guild_ids": "FLUXER_VOICE_GUILD_IDS",
        "participant_prefix": "FLUXER_VOICE_PARTICIPANT_PREFIX",
        "brain_provider": "FLUXER_VOICE_BRAIN_PROVIDER",
        "stt_provider": "FLUXER_VOICE_STT_PROVIDER",
        "stt_model": "FLUXER_VOICE_STT_MODEL",
        "elevenlabs_language_code": "FLUXER_VOICE_ELEVENLABS_LANGUAGE_CODE",
        "tts_voice": "FLUXER_VOICE_TTS_VOICE",
        "context_file": "FLUXER_VOICE_CONTEXT_FILE",
        "session_db": "FLUXER_VOICE_SESSION_DB",
        "turn_log_jsonl": "FLUXER_VOICE_TURN_LOG_JSONL",
        "python": "FLUXER_VOICE_PYTHON",
        "hermes_url": "FLUXER_VOICE_HERMES_URL",
        "hermes_model": "FLUXER_VOICE_HERMES_MODEL",
        "hermes_timeout_seconds": "FLUXER_VOICE_HERMES_TIMEOUT_SECONDS",
        "hermes_max_tokens": "FLUXER_VOICE_HERMES_MAX_TOKENS",
        "hermes_temperature": "FLUXER_VOICE_HERMES_TEMPERATURE",
        "hermes_session_id": "FLUXER_VOICE_HERMES_SESSION_ID",
        "hermes_session_key": "FLUXER_VOICE_HERMES_SESSION_KEY",
        "max_turns": "FLUXER_VOICE_MAX_TURNS",
        "initial_settle_seconds": "FLUXER_VOICE_INITIAL_SETTLE_SECONDS",
        "sample_rate": "FLUXER_VOICE_SAMPLE_RATE",
    }
    for key, env_name in voice_scalar_env.items():
        if not scoped_load and key in voice:
            _set_env_default(env_name, voice.get(key))

    vad_raw = voice.get("vad")
    vad: dict = vad_raw if isinstance(vad_raw, dict) else {}
    for key, env_name in {
        "silence_ms": "FLUXER_VOICE_SILENCE_MS",
        "end_padding_ms": "FLUXER_VOICE_END_PADDING_MS",
        "min_segment_ms": "FLUXER_VOICE_MIN_SEGMENT_MS",
        "max_segment_seconds": "FLUXER_VOICE_MAX_SEGMENT_SECONDS",
        "energy_threshold": "FLUXER_VOICE_ENERGY_THRESHOLD",
        "frame_ms": "FLUXER_VOICE_FRAME_MS",
    }.items():
        if not scoped_load and key in vad:
            _set_env_default(env_name, vad.get(key))

    timeouts_raw = voice.get("timeouts")
    timeouts: dict = timeouts_raw if isinstance(timeouts_raw, dict) else {}
    for key, env_name in {
        "capture_seconds": "FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS",
        "connect_seconds": "FLUXER_VOICE_CONNECT_TIMEOUT_SECONDS",
        "xai_seconds": "FLUXER_VOICE_XAI_TIMEOUT_SECONDS",
        "xai_first_audio_seconds": "FLUXER_VOICE_XAI_FIRST_AUDIO_TIMEOUT_SECONDS",
        "max_runtime_seconds": "FLUXER_VOICE_MAX_RUNTIME_SECONDS",
        "start_cooldown_seconds": "FLUXER_VOICE_START_COOLDOWN_SECONDS",
        "stop_timeout_seconds": "FLUXER_VOICE_STOP_TIMEOUT_SECONDS",
    }.items():
        if not scoped_load and key in timeouts:
            _set_env_default(env_name, timeouts.get(key))

    barge_in_raw = voice.get("barge_in")
    barge_in: dict = barge_in_raw if isinstance(barge_in_raw, dict) else {}
    for key, env_name in {
        "disable": "FLUXER_VOICE_DISABLE_BARGE_IN",
        "energy_threshold": "FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD",
        "min_ms": "FLUXER_VOICE_BARGE_IN_MIN_MS",
        "window_ms": "FLUXER_VOICE_BARGE_IN_WINDOW_MS",
        "capture_timeout_seconds": "FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS",
        "stop_phrase_energy_threshold": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD",
        "stop_phrase_min_ms": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS",
        "stop_phrase_silence_ms": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS",
        "stop_phrase_max_seconds": "FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS",
        "after_first_audio_only": "FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY",
    }.items():
        if not scoped_load and key in barge_in:
            _set_env_default(env_name, barge_in.get(key))

    return seed or None


def _env_enablement() -> dict | None:
    base_url = _fluxer_env("FLUXER_BASE_URL", "").strip()
    token = _fluxer_env("FLUXER_BOT_TOKEN", "").strip()
    if not token:
        return None
    seed: dict = {"bot_token": token}
    if base_url:
        seed["base_url"] = base_url
    gateway_url = _fluxer_env("FLUXER_GATEWAY_URL", "").strip()
    if gateway_url:
        seed["gateway_url"] = gateway_url
    home = _fluxer_env("FLUXER_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": _fluxer_env("FLUXER_HOME_CHANNEL_NAME", "").strip() or home,
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[Any]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    adapter = FluxerAdapter(pconfig)
    metadata = {"thread_id": thread_id} if thread_id else None
    try:
        last: Optional[SendResult] = None
        sent_message_ids: List[str] = []

        async def _rollback_partial_delivery() -> List[str]:
            rollback_failures: List[str] = []
            for message_id in reversed(sent_message_ids):
                if not await adapter.delete_message(chat_id, message_id):
                    rollback_failures.append(message_id)
            return rollback_failures

        media_items = media_files or []
        native_caption: Optional[str] = None
        if message and len(media_items) == 1:
            media_item = media_items[0]
            if isinstance(media_item, (tuple, list)):
                media_path = str(media_item[0])
                is_voice_directive = bool(media_item[1]) if len(media_item) > 1 else False
            else:
                media_path = str(media_item)
                is_voice_directive = False
            if force_document or not is_voice_directive:
                candidate = message.strip()
                # Mention neutralization adds one UTF-16 unit per recognized
                # mention. Count the worst case so the final wire caption can
                # never cross Fluxer's 4,000-unit limit.
                mention_expansion = sum(
                    len(pattern.findall(candidate))
                    for pattern in (
                        _MENTION_EVERYONE_RE,
                        _MENTION_ROLE_RE,
                        _MENTION_USER_RE,
                    )
                )
                if (
                    candidate
                    and utf16_len(candidate) + mention_expansion
                    <= MAX_MESSAGE_LENGTH
                ):
                    native_caption = candidate
        if message and native_caption is None:
            last = await adapter.send(chat_id, message, metadata=metadata)
            if not last.success:
                return {"error": last.error or "Fluxer send failed"}
            if last.message_id:
                sent_message_ids.append(last.message_id)
        for media_item in media_items:
            if isinstance(media_item, (tuple, list)):
                media_path = str(media_item[0])
                is_voice_directive = bool(media_item[1]) if len(media_item) > 1 else False
            else:
                media_path = str(media_item)
                is_voice_directive = False
            ext = Path(media_path).suffix.lower()
            if not force_document and ext in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
                last = await adapter.send_image_file(
                    chat_id, media_path, caption=native_caption, metadata=metadata
                )
            elif not force_document and ext in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                last = await adapter.send_video(
                    chat_id, media_path, caption=native_caption, metadata=metadata
                )
            elif not force_document and is_voice_directive:
                last = await adapter.send_voice(chat_id, media_path, metadata=metadata)
            else:
                last = await adapter.send_document(
                    chat_id, media_path, caption=native_caption, metadata=metadata
                )
            if not last.success:
                if not sent_message_ids:
                    return {"error": "Fluxer media send failed"}
                rollback_failures = await _rollback_partial_delivery()
                if rollback_failures:
                    return {
                        "success": True,
                        "platform": "fluxer",
                        "chat_id": chat_id,
                        "message_id": rollback_failures[-1],
                        "partial_delivery": True,
                        "warnings": [
                            "Fluxer media delivery was incomplete; existing "
                            "messages were kept to prevent duplicate retries"
                        ],
                    }
                return {
                    "error": (
                        "Fluxer media send failed; partial delivery rolled back"
                    ),
                    "rollback_complete": True,
                }
            if last.message_id:
                sent_message_ids.append(last.message_id)
        if last and last.success:
            return {"success": True, "platform": "fluxer", "chat_id": chat_id, "message_id": last.message_id}
        return {"error": "Fluxer send failed: empty message and no media"}
    except Exception as exc:
        return {"error": str(exc)}


async def _send_message_handler(
    args: Dict[str, Any],
    chat_id: str,
    _platform_name: str,
    pconfig: PlatformConfig,
    *,
    normalized: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Preserve Fluxer MEDIA directives on every host send path."""
    if normalized is not None:
        cleaned_message = str(normalized.get("message") or "")
        media_files = list(normalized.get("media_files") or [])
        force_document = bool(normalized.get("force_document"))
        thread_id = normalized.get("thread_id")
    else:
        raw_message = str(args.get("message") or "")
        force_document = "[[as_document]]" in raw_message
        media_files, cleaned_message = BasePlatformAdapter.extract_media(raw_message)
        media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
        thread_id = args.get("thread_id")
    return await _standalone_send(
        pconfig,
        chat_id,
        cleaned_message,
        thread_id=str(thread_id) if thread_id else None,
        media_files=media_files,
        force_document=force_document,
    )


def _supports_normalized_send_handler_context() -> bool:
    try:
        from tools.send_message_tool import (
            PLUGIN_SEND_HANDLER_NORMALIZED_CONTEXT,
        )

        return PLUGIN_SEND_HANDLER_NORMALIZED_CONTEXT >= 1
    except (ImportError, TypeError):
        return False


def interactive_setup() -> None:
    print("Fluxer platform setup")
    print("Set FLUXER_BOT_TOKEN in ~/.hermes/.env, then restart the gateway.")
    print("Optional: set FLUXER_BASE_URL for self-hosted Fluxer; official hosted defaults to https://api.fluxer.app/v1.")


def on_pre_command(
    *, surface: str = "", command: str = "", args_raw: str = "",
    session_key: Optional[str] = None, platform: Optional[str] = None, **_ignored: Any,
) -> None:
    """Mirror ``/title`` onto the thread, matching the gateway's own topic renames.

    A host that has no platform-generic thread-rename lane (Hermes core) still lets
    the plugin observe the command: ``pre_command`` is fired from the gateway before
    dispatch, on the gateway loop, and carries the session key. Observer-only, so
    the session record is still titled by the host — this only names the thread.
    """
    if str(platform or "").lower() != "fluxer":
        return
    if str(command or "").strip().lower() != "title":
        return
    title = str(args_raw or "").strip()
    if not title or not session_key:
        return
    for adapter in list(_LIVE_ADAPTERS):
        try:
            adapter.rename_thread_from_session_title(str(session_key), title)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Fluxer /title hook failed: %s", exc)


def register(ctx) -> None:
    # ``/title`` sets the session title through the host's own command handler; the
    # observer below mirrors it onto the thread, so no host-side rename lane is needed.
    try:
        ctx.register_hook("pre_command", on_pre_command)
    except Exception as exc:  # pragma: no cover - older hosts
        logger.debug("Fluxer plugin could not register the pre_command hook: %s", exc)
    ctx.register_platform(
        name="fluxer",
        label="Fluxer",
        adapter_factory=lambda cfg: FluxerAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["FLUXER_BOT_TOKEN"],
        install_hint="pip install httpx websockets   # Fluxer adapter dependencies",
        setup_fn=interactive_setup,
        env_enablement_fn=_env_enablement,
        apply_yaml_config_fn=_apply_yaml_config,
        cron_deliver_env_var="FLUXER_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        send_message_handler=(
            _send_message_handler
            if _supports_normalized_send_handler_context()
            else None
        ),
        allowed_users_env="FLUXER_ALLOWED_USERS",
        allow_all_env="FLUXER_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_LENGTH,
        emoji="⚡",
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via Fluxer, a Discord-like open-source chat "
            "platform. Fluxer supports rich Markdown, channels, DMs, files, "
            "and voice/video. Prefer normal Markdown for structured replies."
        ),
    )
