"""Hermes gateway test doubles for standalone plugin CI.

The live plugin imports Hermes gateway classes at runtime. The standalone plugin
repository should still be testable without installing the whole Hermes tree, so
CI provides minimal gateway modules only when the real package is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import atexit
import os
import re
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace
from typing import Any

import pytest


# Tests instantiate real Hermes BasePlatformAdapter when the Hermes checkout is
# importable. Its _mark_connected/_mark_disconnected helpers write
# gateway_state.json under HERMES_HOME, so never let standalone plugin tests
# mutate the live user's dashboard/gateway runtime files.
_hermes_test_home = tempfile.mkdtemp(prefix="fluxer-platform-test-home-")
atexit.register(shutil.rmtree, _hermes_test_home, ignore_errors=True)
os.environ["HERMES_HOME"] = _hermes_test_home

# The plugin's adapters and voice scripts read `FLUXER_*` / STT env vars BEFORE
# any config file or `extra` mapping, so a shell that sourced the operator's
# `~/.hermes/.env` (any live probe does) silently re-points the suite at the
# operator's deployment: the home-guild, free-response and mention gates then
# assert against the wrong ids and four `test_plugin_package.py` tests fail as a
# phantom regression. The tests never need real credentials — they set what they
# need with `monkeypatch.setenv` — so the whole `FLUXER_*` / STT surface is
# cleared at import time and the operator's real secrets are never read.
_ENV_LEAK_PREFIXES = ("FLUXER_",)
_ENV_LEAK_EXACT = (
    "HERMES_LOCAL_STT_COMMAND",
    "ELEVENLABS_API_KEY",
    "ELEVENLABS_STT_BASE_URL",
    "GROQ_API_KEY",
    "GROQ_BASE_URL",
    "XAI_API_KEY",
    "XAI_STT_BASE_URL",
)

# The voice scripts resolve `--env-file` at PARSE time from `HERMES_ENV_FILE`,
# else `~/.hermes/.env`. Left alone, a test that calls `parse_args` without
# `--env-file` inherits the operator's real env file and `load_env_file` then
# mutates `os.environ` with their live secrets. Pin the default at an empty file
# inside the throwaway test home so the resolution is deterministic and harmless.
ISOLATED_ENV_FILE = Path(_hermes_test_home) / "isolated.env"

for _key in list(os.environ):
    if _key.startswith(_ENV_LEAK_PREFIXES) or _key in _ENV_LEAK_EXACT:
        os.environ.pop(_key, None)


@pytest.fixture(autouse=True)
def _isolate_fluxer_env(monkeypatch):
    """Keep the leak surface clean for every test, and prove no real secret is read.

    A test may set `FLUXER_*` / STT vars with `monkeypatch.setenv`; those are
    reverted per test. Anything the suite needs is synthetic, so no operator
    credential in `~/.hermes/.env` or the ambient shell is ever consumed.
    """

    for key in list(os.environ):
        if key.startswith(_ENV_LEAK_PREFIXES) or key in _ENV_LEAK_EXACT:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HERMES_ENV_FILE", str(ISOLATED_ENV_FILE))
    yield


try:  # Prefer the real Hermes gateway package when the test runner has it.
    # The voice scripts put the Hermes root on `sys.path` themselves, which makes
    # `hermes_state` / `hermes_constants` importable even when the runner did not.
    # Resolve that same root here BEFORE the gateway probe, so `gateway` and
    # `hermes_state` always agree: both real, or both stubbed. A mixed pair
    # (stubbed `gateway` + real `hermes_state`) silently fails the host-integration
    # tests with `AttributeError: set_owner_profile`.
    _hermes_root = Path(
        os.getenv("HERMES_AGENT_ROOT", str(Path.home() / ".hermes" / "hermes-agent"))
    )
    if _hermes_root.is_dir() and str(_hermes_root) not in sys.path:
        sys.path.append(str(_hermes_root))
    import gateway.config  # type: ignore[import-not-found]  # noqa: F401
    import gateway.platforms.base  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    gateway_pkg = types.ModuleType("gateway")
    gateway_pkg.__path__ = []  # mark as package
    config_mod = types.ModuleType("gateway.config")
    platforms_pkg = types.ModuleType("gateway.platforms")
    platforms_pkg.__path__ = []
    base_mod = types.ModuleType("gateway.platforms.base")

    class Platform(str, Enum):
        FLUXER = "fluxer"

    @dataclass
    class PlatformConfig:
        enabled: bool = True
        extra: dict[str, Any] = field(default_factory=dict)
        home_channel: str | None = None
        home_channel_name: str | None = None

    class MessageType(Enum):
        TEXT = "text"
        AUDIO = "audio"
        VOICE = "voice"
        IMAGE = "image"
        DOCUMENT = "document"

    @dataclass
    class MessageEvent:
        text: str = ""
        message_type: MessageType = MessageType.TEXT
        source: Any = None
        raw_message: dict[str, Any] = field(default_factory=dict)
        message_id: str | None = None
        media_urls: list[str] = field(default_factory=list)
        media_types: list[str] = field(default_factory=list)
        reply_to_message_id: str | None = None
        reply_to_text: str | None = None
        timestamp: Any = None

    @dataclass
    class SendResult:
        success: bool
        message_id: str | None = None
        error: str | None = None
        retryable: bool = False
        raw_response: dict[str, Any] | None = None
        retry_after: float | None = None
        continuation_message_ids: tuple = ()
        error_kind: str | None = None

    class BasePlatformAdapter:
        def __init__(self, config: PlatformConfig, platform: Platform | str | None = None):
            self.config = config
            self.platform = platform

        def format_message(self, content: str) -> str:
            return content

        # Host contract surface the adapter overrides / advertises. The real base
        # defines these; the stub carries the same names so a capability test can
        # assert the override against a base that actually has the hook.
        async def _send_exec_approval_prompt(self, prompt: Any) -> Any:
            return SendResult(success=False, error="Not supported")

        @classmethod
        def supports_exec_approval_buttons(cls) -> bool:
            return cls._send_exec_approval_prompt is not BasePlatformAdapter._send_exec_approval_prompt

        def build_source(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        def set_session_store(self, session_store: Any) -> None:
            self._session_store = session_store

        def validate_media_delivery_path(self, file_path: str) -> Path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(file_path)
            return path

        @staticmethod
        def extract_media(content: str) -> tuple[list[tuple[str, bool]], str]:
            """Standalone-test subset of Hermes' MEDIA directive contract."""
            has_voice_tag = "[[audio_as_voice]]" in content
            cleaned = content.replace("[[audio_as_voice]]", "").replace(
                "[[as_document]]", ""
            )
            media: list[tuple[str, bool]] = []
            matched = False
            for match in re.finditer(r"MEDIA:(?P<path>\S+)", cleaned):
                matched = True
                media_path = os.path.expanduser(match.group("path"))
                is_voice = has_voice_tag and Path(media_path).suffix.lower() in {
                    ".mp3", ".m4a", ".ogg", ".opus", ".wav", ".flac", ".aac"
                }
                media.append((media_path, is_voice))
            cleaned = re.sub(r"MEDIA:\S+", "", cleaned)
            if matched or cleaned != content:
                cleaned = "\n".join(
                    line.strip() for line in cleaned.splitlines() if line.strip()
                )
            return media, cleaned

        @staticmethod
        def filter_media_delivery_paths(
            media_files: list[tuple[str, bool]] | None,
            session_key: str = "",
        ) -> list[tuple[str, bool]]:
            del session_key
            safe_media: list[tuple[str, bool]] = []
            for media_path, is_voice in media_files or []:
                path = Path(media_path)
                if path.exists() and path.is_file():
                    safe_media.append((str(path.resolve()), bool(is_voice)))
            return safe_media

        @staticmethod
        def truncate_message(
            content: str,
            max_length: int = 4096,
            len_fn: Any = None,
        ) -> list[str]:
            """Small standalone-test equivalent of Hermes' unit-aware splitter."""
            measure = len_fn or len
            chunks: list[str] = []
            remaining = content
            while remaining:
                if measure(remaining) <= max_length:
                    chunks.append(remaining)
                    break
                low, high = 1, len(remaining)
                while low < high:
                    mid = (low + high + 1) // 2
                    if measure(remaining[:mid]) <= max_length:
                        low = mid
                    else:
                        high = mid - 1
                split_at = max(1, low)
                chunks.append(remaining[:split_at])
                remaining = remaining[split_at:]
            return chunks

    async def _cache_from_url(url: str, *_args: Any, **_kwargs: Any) -> str:
        return url

    async def cache_document_from_bytes(data: bytes, filename: str, *_args: Any, **_kwargs: Any) -> str:
        path = Path("/tmp") / filename
        path.write_bytes(data)
        return str(path)

    def safe_url_for_log(url: str) -> str:
        return url

    def utf16_len(value: str) -> int:
        return len(value.encode("utf-16-le")) // 2

    config_mod.Platform = Platform
    config_mod.PlatformConfig = PlatformConfig
    base_mod.BasePlatformAdapter = BasePlatformAdapter
    base_mod.MessageEvent = MessageEvent
    base_mod.MessageType = MessageType
    base_mod.SendResult = SendResult
    base_mod.SUPPORTED_IMAGE_DOCUMENT_TYPES = {"image/png", "image/jpeg", "image/webp"}
    base_mod.cache_audio_from_url = _cache_from_url
    base_mod.cache_document_from_bytes = cache_document_from_bytes
    base_mod.cache_image_from_url = _cache_from_url
    base_mod.safe_url_for_log = safe_url_for_log
    base_mod.utf16_len = utf16_len

    def get_inbound_media_max_bytes() -> int:
        return 128 * 1024 * 1024

    base_mod.get_inbound_media_max_bytes = get_inbound_media_max_bytes

    sys.modules.setdefault("gateway", gateway_pkg)
    sys.modules.setdefault("gateway.config", config_mod)
    sys.modules.setdefault("gateway.platforms", platforms_pkg)
    sys.modules.setdefault("gateway.platforms.base", base_mod)


try:  # Prefer the real Hermes tools package when the runner has it.
    import tools.url_safety  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:
    # The plugin's SSRF/size/approval hardening reaches host helpers under
    # ``tools.*``. Standalone plugin CI has no Hermes tree, so provide minimal
    # stand-ins with the same names — tests monkeypatch them, and the adapter's
    # own fallbacks are exercised against them. A test needing the REAL contract
    # uses ``pytest.importorskip("tools.url_safety")`` and skips here.
    import ipaddress
    from urllib.parse import urlparse as _urlparse

    tools_pkg = sys.modules.get("tools") or types.ModuleType("tools")
    tools_pkg.__path__ = getattr(tools_pkg, "__path__", [])
    url_safety_mod = types.ModuleType("tools.url_safety")
    approval_mod = types.ModuleType("tools.approval")
    approval_context_mod = types.ModuleType("tools.approval_context")

    def is_safe_url(url: str) -> bool:
        parsed = _urlparse(url)
        if (parsed.scheme or "").lower() not in {"http", "https"}:
            return False
        host = (parsed.hostname or "").lower()
        if not host:
            return False
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True  # a hostname is assumed public in the stub
        return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved)

    def _no_guard(**kwargs):
        raise ModuleNotFoundError("SSRF-safe client unavailable in standalone CI")

    def redirect_target_from_response(response):
        """Same shape as the host's: resolve from Location, then next_request."""
        from urllib.parse import urljoin as _urljoin

        if not getattr(response, "is_redirect", False):
            return None
        location = (getattr(response, "headers", {}) or {}).get("location")
        if location:
            return _urljoin(str(getattr(response, "url", "")), str(location))
        next_request = getattr(response, "next_request", None)
        return str(next_request.url) if next_request else None

    url_safety_mod.is_safe_url = is_safe_url
    url_safety_mod.create_ssrf_safe_async_client = _no_guard
    url_safety_mod.redirect_target_from_response = redirect_target_from_response

    def resolve_gateway_approval(session_key, choice, resolve_all=False, reason=None, request_id=None):
        return 0

    approval_mod.resolve_gateway_approval = resolve_gateway_approval
    approval_mod.list_gateway_approvals = lambda session_key: []

    approval_context_mod._get_approval_timeout = lambda: 300

    sys.modules.setdefault("tools", tools_pkg)
    sys.modules.setdefault("tools.url_safety", url_safety_mod)
    sys.modules.setdefault("tools.approval", approval_mod)
    sys.modules.setdefault("tools.approval_context", approval_context_mod)
