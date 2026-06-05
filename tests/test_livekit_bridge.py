import pytest

import livekit_bridge


class FakeParticipant:
    identity = "fluxer-bot"

    def __init__(self):
        self.published = []
        self.unpublished = []

    async def publish_track(self, track, options):
        self.published.append((track, options))
        return type("Publication", (), {"sid": "track-sid", "track_sid": "track-sid"})()

    async def unpublish_track(self, track_sid):
        self.unpublished.append(track_sid)


class FakeRoom:
    def __init__(self):
        self.connected = []
        self.disconnected = False
        self.name = "Fluxer voice room"
        self.local_participant = FakeParticipant()
        self.remote_participants = {}
        self.handlers = {}

    def on(self, event, callback=None):
        self.handlers[event] = callback
        return callback

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

    def __init__(self, sample_rate, num_channels, queue_size_ms=1000):
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.queue_size_ms = queue_size_ms
        self.frames = []
        self.waited = False
        self.cleared = False
        self.closed = False
        self.queued_duration = 0.42
        FakeAudioSource.instances.append(self)

    async def capture_frame(self, frame):
        self.frames.append(frame)

    async def wait_for_playout(self):
        self.waited = True

    def clear_queue(self):
        self.cleared = True
        self.queued_duration = 0.0

    async def aclose(self):
        self.closed = True


class FakeAudioFrame:
    def __init__(self, data, sample_rate, num_channels, samples_per_channel):
        self.data = data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
        self.samples_per_channel = samples_per_channel


class FakeLocalAudioTrack:
    instances = []

    def __init__(self, name, source):
        self.name = name
        self.source = source
        self.stopped = False
        FakeLocalAudioTrack.instances.append(self)

    @staticmethod
    def create_audio_track(name, source):
        return FakeLocalAudioTrack(name, source)

    def stop(self):
        self.stopped = True


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
    assert room.local_participant.published[0][0].name == "zofka-test-tone"
    assert room.local_participant.published[0][0].source is source
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

@pytest.mark.asyncio
async def test_publish_wav_file_publishes_pcm_audio_frames(monkeypatch, tmp_path):
    import wave

    FakeAudioSource.instances = []
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})
    wav_path = tmp_path / "zofka.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"\x00\x00" * 40)

    await bridge.publish_wav_file(wav_path, frame_ms=20)

    source = FakeAudioSource.instances[0]
    assert room.local_participant.published[0][0].name == "zofka-tts-smoke"
    assert room.local_participant.published[0][0].source is source
    assert [frame.samples_per_channel for frame in source.frames] == [20, 20]
    assert source.waited is True
    assert source.closed is True


@pytest.mark.asyncio
async def test_publish_wav_file_rejects_non_mono_pcm(monkeypatch, tmp_path):
    import wave

    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})
    wav_path = tmp_path / "stereo.wav"
    with wave.open(str(wav_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(1000)
        wav.writeframes(b"\x00\x00" * 80)

    with pytest.raises(ValueError, match="mono 16-bit PCM"):
        await bridge.publish_wav_file(wav_path)

@pytest.mark.asyncio
async def test_pcm16_publisher_streams_chunks_and_flushes_remainder(monkeypatch):
    FakeAudioSource.instances = []
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})

    async with bridge.pcm16_publisher(sample_rate=1000, frame_ms=20, track_name="zofka-stream") as publisher:
        await publisher.write(b"\x01\x00" * 15)
        assert publisher.frames_published == 0
        await publisher.write(b"\x02\x00" * 30)
        assert publisher.frames_published == 2

    source = FakeAudioSource.instances[0]
    assert room.local_participant.published[0][0].name == "zofka-stream"
    assert room.local_participant.published[0][0].source is source
    assert room.local_participant.published[0][1].source == FakeTrackSource.SOURCE_MICROPHONE
    assert source.queue_size_ms == 120
    assert [frame.samples_per_channel for frame in source.frames] == [20, 20, 5]
    assert publisher.bytes_published == 90
    assert source.waited is True
    assert source.closed is True


@pytest.mark.asyncio
async def test_pcm16_publisher_interrupt_clears_queue_without_playout(monkeypatch):
    FakeAudioSource.instances = []
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})

    publisher = bridge.pcm16_publisher(sample_rate=1000, frame_ms=20, track_name="zofka-stream")
    await publisher.__aenter__()
    await publisher.write(b"\x01\x00" * 40)
    await publisher.write(b"\x02\x00" * 5)
    await publisher.interrupt()

    source = FakeAudioSource.instances[0]
    track = room.local_participant.published[0][0]
    assert publisher.interrupted is True
    assert source.cleared is True
    assert publisher.last_queue_duration_before_interrupt == pytest.approx(0.42)
    assert publisher.last_queue_duration_after_clear == pytest.approx(0.0)
    assert source.waited is False
    assert source.closed is True
    assert track.stopped is True
    assert room.local_participant.unpublished == ["track-sid"]
    assert [frame.samples_per_channel for frame in source.frames] == [20, 20]
    with pytest.raises(RuntimeError, match="not open"):
        await publisher.write(b"\x03\x00" * 20)


@pytest.mark.asyncio
async def test_pcm16_publisher_interruptible_write_stops_mid_chunk(monkeypatch):
    FakeAudioSource.instances = []
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: FakeRtc)
    room = FakeRoom()
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    await bridge.connect_from_voice_server_update({"endpoint": "wss://livekit.fluxer.example", "token": "secret"})

    publisher = bridge.pcm16_publisher(sample_rate=1000, frame_ms=20, track_name="zofka-stream")
    await publisher.__aenter__()
    calls = 0

    async def should_interrupt():
        nonlocal calls
        calls += 1
        return calls >= 3

    interrupted = await publisher.write_interruptible(b"\x01\x00" * 100, should_interrupt)

    source = FakeAudioSource.instances[0]
    assert interrupted is True
    assert publisher.interrupted is True
    assert source.cleared is True
    assert publisher.last_queue_duration_before_interrupt == pytest.approx(0.42)
    assert publisher.last_queue_duration_after_clear == pytest.approx(0.0)
    assert source.closed is True
    assert room.local_participant.unpublished == ["track-sid"]
    assert [frame.samples_per_channel for frame in source.frames] == [20, 20]


class FakeAudioFrameEvent:
    def __init__(self, data):
        self.frame = type("Frame", (), {"data": data})()


class FakeAudioStream:
    tracks = []

    def __init__(self, events):
        self.events = list(events)
        self.closed = False

    @classmethod
    def from_track(cls, *, track, sample_rate, num_channels, frame_size_ms):
        cls.tracks.append((track, sample_rate, num_channels, frame_size_ms))
        return cls([FakeAudioFrameEvent(b"\x01\x00" * 200), FakeAudioFrameEvent(b"\x02\x00" * 200)])

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
async def test_collect_remote_audio_pcm16_from_existing_track(monkeypatch):
    FakeAudioStream.tracks = []
    room = FakeRoom()
    participant = type("RemoteParticipant", (), {"identity": "user-a", "track_publications": {}})()
    track = type("RemoteAudioTrack", (), {"kind": "audio"})()
    participant.track_publications["pub"] = type("Publication", (), {"track": track})()
    room.remote_participants["user-a"] = participant

    fake_rtc = type("FakeRtc", (), {"AudioStream": FakeAudioStream})()
    monkeypatch.setattr(livekit_bridge, "_load_livekit_audio_helpers", lambda: fake_rtc)
    bridge = livekit_bridge.FluxerLiveKitSmokeBridge(room_factory=lambda: room)
    bridge._room = room

    pcm = await bridge.collect_remote_audio_pcm16(duration_seconds=0.01, sample_rate=24000, participant_identity="user-a")

    assert len(pcm) == 480
    assert FakeAudioStream.tracks == [(track, 24000, 1, 20)]
    assert room.handlers["track_subscribed"] is not None
