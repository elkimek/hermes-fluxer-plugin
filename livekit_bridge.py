"""Small Fluxer LiveKit smoke bridge.

This module is deliberately transport-only: it can connect to the LiveKit room
that Fluxer returns in VOICE_SERVER_UPDATE, then disconnect cleanly. It does not
publish/listen to audio yet. The goal is to prove the bot token can enter the
room before wiring realtime STT/TTS.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


class LiveKitRoomLike(Protocol):
    async def connect(self, url: str, token: str, options: Any = ...) -> None: ...

    async def disconnect(self, **kwargs: Any) -> None: ...


RoomFactory = Callable[[], LiveKitRoomLike]


@dataclass(frozen=True)
class FluxerLiveKitConnectionInfo:
    """Non-secret connection metadata captured after a successful smoke join."""

    endpoint: str
    guild_id: Optional[str]
    channel_id: Optional[str]
    connection_id: Optional[str]
    room_name: Optional[str] = None
    participant_identity: Optional[str] = None


def _load_livekit_room_factory() -> tuple[RoomFactory, Callable[..., Any]]:
    """Load LiveKit lazily so normal text/voice-message plugin use stays light."""

    try:
        from livekit import rtc  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message text tests.
        raise RuntimeError(
            "Fluxer realtime voice requires the optional dependency: "
            "pip install 'hermes-fluxer-plugin[realtime]'"
        ) from exc
    return rtc.Room, rtc.RoomOptions


def _string_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


class FluxerLiveKitSmokeBridge:
    """Connect/disconnect proof for Fluxer's LiveKit voice-room handoff."""

    def __init__(self, *, room_factory: Optional[RoomFactory] = None, auto_subscribe: bool = False) -> None:
        self._room_factory = room_factory
        self._auto_subscribe = auto_subscribe
        self._room: Optional[LiveKitRoomLike] = None
        self.last_connection: Optional[FluxerLiveKitConnectionInfo] = None

    @property
    def connected(self) -> bool:
        return self._room is not None

    async def connect_from_voice_server_update(self, update: dict[str, Any]) -> FluxerLiveKitConnectionInfo:
        """Connect using a raw Fluxer VOICE_SERVER_UPDATE payload.

        The input may contain the ephemeral LiveKit token. The token is used only
        as a local argument to `Room.connect(...)`; it is never stored on `self`,
        returned, or logged.
        """

        endpoint = _string_or_none(update.get("endpoint"))
        token = _string_or_none(update.get("token"))
        if not endpoint:
            raise ValueError("Fluxer VOICE_SERVER_UPDATE did not include a LiveKit endpoint")
        if not token:
            raise ValueError("Fluxer VOICE_SERVER_UPDATE did not include a LiveKit token")

        await self.disconnect()

        room_factory = self._room_factory
        options: Any = None
        if room_factory is None:
            room_factory, room_options_factory = _load_livekit_room_factory()
            options = room_options_factory(auto_subscribe=self._auto_subscribe)

        room = room_factory()
        if options is None:
            await room.connect(endpoint, token)
        else:
            await room.connect(endpoint, token, options)

        info = FluxerLiveKitConnectionInfo(
            endpoint=endpoint,
            guild_id=_string_or_none(update.get("guild_id")),
            channel_id=_string_or_none(update.get("channel_id")),
            connection_id=_string_or_none(update.get("connection_id")),
            room_name=_string_or_none(getattr(room, "name", None)),
            participant_identity=_string_or_none(getattr(getattr(room, "local_participant", None), "identity", None)),
        )
        self._room = room
        self.last_connection = info
        logger.info(
            "Fluxer LiveKit smoke bridge connected endpoint=%s channel=%s guild=%s connection=%s",
            info.endpoint,
            info.channel_id or "<none>",
            info.guild_id or "<dm>",
            info.connection_id or "<none>",
        )
        return info

    async def disconnect(self) -> None:
        room = self._room
        self._room = None
        if room is not None:
            result = room.disconnect()
            if asyncio.iscoroutine(result):
                await result
