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
import shutil
import sys
import tempfile
import types
from types import SimpleNamespace
from typing import Any


# Tests instantiate real Hermes BasePlatformAdapter when the Hermes checkout is
# importable. Its _mark_connected/_mark_disconnected helpers write
# gateway_state.json under HERMES_HOME, so never let standalone plugin tests
# mutate the live user's dashboard/gateway runtime files.
_hermes_test_home = tempfile.mkdtemp(prefix="fluxer-platform-test-home-")
atexit.register(shutil.rmtree, _hermes_test_home, ignore_errors=True)
os.environ["HERMES_HOME"] = _hermes_test_home


try:  # Prefer the real Hermes gateway package when the test runner has it.
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

    class BasePlatformAdapter:
        def __init__(self, config: PlatformConfig, platform: Platform | str | None = None):
            self.config = config
            self.platform = platform

        def format_message(self, content: str) -> str:
            return content

        def build_source(self, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(**kwargs)

        def validate_media_delivery_path(self, file_path: str) -> Path:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(file_path)
            return path

    async def _cache_from_url(url: str, *_args: Any, **_kwargs: Any) -> str:
        return url

    async def cache_document_from_bytes(data: bytes, filename: str, *_args: Any, **_kwargs: Any) -> str:
        path = Path("/tmp") / filename
        path.write_bytes(data)
        return str(path)

    def safe_url_for_log(url: str) -> str:
        return url

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

    sys.modules.setdefault("gateway", gateway_pkg)
    sys.modules.setdefault("gateway.config", config_mod)
    sys.modules.setdefault("gateway.platforms", platforms_pkg)
    sys.modules.setdefault("gateway.platforms.base", base_mod)
