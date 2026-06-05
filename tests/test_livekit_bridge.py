import pytest

import livekit_bridge


class FakeParticipant:
    identity = "fluxer-bot"


class FakeRoom:
    def __init__(self):
        self.connected = []
        self.disconnected = False
        self.name = "Fluxer voice room"
        self.local_participant = FakeParticipant()

    async def connect(self, url, token, options=None):
        self.connected.append((url, token, options))

    async def disconnect(self, **kwargs):
        self.disconnected = True


@pytest.mark.asyncio
async def test_smoke_bridge_connects_with_raw_token_but_only_returns_safe_metadata():
    rooms = []

    def room_factory():
        room = FakeRoom()
        rooms.append(room)
        return room

    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=room_factory)

    info = await bridge.connect_from_voice_server_update(
        {
            "guild_id": "guild-1",
            "channel_id": "voice-chan",
            "connection_id": "conn-1",
            "endpoint": "wss://livekit.fluxer.example",
            "token": "ephemeral-livekit-token",
        }
    )

    assert bridge.connected is True
    assert rooms[0].connected == [("wss://livekit.fluxer.example", "ephemeral-livekit-token", None)]
    assert info.endpoint == "wss://livekit.fluxer.example"
    assert info.guild_id == "guild-1"
    assert info.channel_id == "voice-chan"
    assert info.connection_id == "conn-1"
    assert info.room_name == "Fluxer voice room"
    assert info.participant_identity == "fluxer-bot"
    assert "ephemeral-livekit-token" not in repr(info)
    assert "ephemeral-livekit-token" not in repr(bridge.__dict__)


@pytest.mark.asyncio
async def test_smoke_bridge_disconnects_existing_room_before_reconnect():
    rooms = []

    def room_factory():
        room = FakeRoom()
        rooms.append(room)
        return room

    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=room_factory)
    await bridge.connect_from_voice_server_update(
        {"endpoint": "wss://one.example", "token": "first", "channel_id": "one"}
    )
    await bridge.connect_from_voice_server_update(
        {"endpoint": "wss://two.example", "token": "second", "channel_id": "two"}
    )

    assert len(rooms) == 2
    assert rooms[0].disconnected is True
    assert rooms[1].disconnected is False
    assert bridge.last_connection is not None
    assert bridge.last_connection.channel_id == "two"


@pytest.mark.asyncio
async def test_smoke_bridge_requires_endpoint_and_token_without_leaking_token():
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=FakeRoom)

    with pytest.raises(ValueError, match="endpoint"):
        await bridge.connect_from_voice_server_update({"token": "secret-token"})

    with pytest.raises(ValueError, match="token") as exc:
        await bridge.connect_from_voice_server_update(
            {"endpoint": "wss://livekit.fluxer.example", "token": ""}
        )

    assert "secret-token" not in str(exc.value)
