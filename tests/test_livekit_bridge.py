import pytest

import livekit_bridge


class FakeParticipant:
    identity = "fluxer-bot"

    def __init__(self):
        self.published = []

    async def publish_track(self, track, options):
        self.published.append((track, options))
        return type("Publication", (), {"sid": "track-sid"})()


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

class FakeAudioSource:
    instances = []

    def __init__(self, sample_rate, num_channels):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.frames = []
        self.waited = False
        self.closed = False
        FakeAudioSource.instances.append(self)

    async def capture_frame(self, frame):
        self.frames.append(frame)

    async def wait_for_playout(self):
        self.waited = True

    async def aclose(self):
        self.closed = True


class FakeAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class FakeLocalAudioTrack:
    @staticmethod
    def create_audio_track(name, source):
        return {"name": name, "source": source}


class FakeTrackPublishOptions:
    def __init__(self):
        self.source = None


class FakeTrackSource:
    SOURCE_MICROPHONE = 2


class FakeRtc:
    AudioSource = FakeAudioSource
    AudioFrame = FakeAudioFrame
    LocalAudioTrack = FakeLocalAudioTrack
    TrackPublishOptions = FakeTrackPublishOptions
    TrackSource = FakeTrackSource


@pytest.mark.asyncio
async def test_publish_test_tone_publishes_pcm_audio_frames(monkeypatch):
    FakeAudioSource.instances = []
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})

    await bridge.publish_test_tone(duration_seconds=0.04, sample_rate=1000, frequency_hz=100, frame_ms=20)

    source = FakeAudioSource.instances[0]
    assert room.local_participant.published[0][0] == {"name": "zofka-test-tone", "source": source}
    assert room.local_participant.published[0][1].source == FakeTrackSource.SOURCE_MICROPHONE
    assert [frame.samples_per_channel for frame in source.frames] == [20, 20]
    assert all(frame.sample_rate == 1000 and frame.num_channels == 1 for frame in source.frames)
    assert source.waited is True
    assert source.closed is True


@pytest.mark.asyncio
async def test_publish_test_tone_requires_connected_room():
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=FakeRoom)

    with pytest.raises(RuntimeError, match="not connected"):
        await bridge.publish_test_tone()
