import argparse

import pytest

from scripts import fluxer_xai_room_loop as room_loop


def pcm16(value: int, samples: int) -> bytes:
    return b"".join(value.to_bytes(2, byteorder="little", signed=True) for _ in range(samples))


async def chunks(*items: bytes):
    for item in items:
        yield item


class FakeChunkIterator:
    def __init__(self, items):
        self.items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)

    async def aclose(self):
        self.closed = True


class FakeBargeBridge:
    def __init__(self, items):
        self.iterator = FakeChunkIterator(items)

    def iter_remote_audio_pcm16(self, **kwargs):
        self.kwargs = kwargs
        return self.iterator


@pytest.mark.asyncio
async def test_speech_segments_trim_final_silence_tail_but_keep_padding():
    # sample_rate=1000 means each 20-sample frame is 20ms and easy to reason about.
    frames = [pcm16(1200, 20) for _ in range(10)]  # 200ms speech
    frames += [pcm16(0, 20) for _ in range(25)]  # 500ms silence; trips 450ms threshold

    segments = room_loop._speech_segments(
        chunks(*frames),
        sample_rate=1000,
        energy_threshold=350,
        silence_ms=450,
        end_padding_ms=100,
        min_segment_ms=100,
        max_segment_seconds=5.0,
    )

    segment = await anext(segments)

    assert len(segment) == (200 + 100) * 2
    assert segment.startswith(pcm16(1200, 20))
    assert segment.endswith(pcm16(0, 100))


@pytest.mark.asyncio
async def test_speech_segments_preserve_short_internal_pauses():
    frames = [pcm16(1200, 20) for _ in range(5)]  # 100ms speech
    frames += [pcm16(0, 20) for _ in range(10)]  # 200ms pause, below end threshold
    frames += [pcm16(1200, 20) for _ in range(5)]  # 100ms speech
    frames += [pcm16(0, 20) for _ in range(25)]  # final 500ms silence

    segments = room_loop._speech_segments(
        chunks(*frames),
        sample_rate=1000,
        energy_threshold=350,
        silence_ms=450,
        end_padding_ms=100,
        min_segment_ms=100,
        max_segment_seconds=5.0,
    )

    segment = await anext(segments)

    # 100ms speech + 200ms internal pause + 100ms speech + 100ms final padding.
    assert len(segment) == 500 * 2
    assert pcm16(0, 200) in segment


@pytest.mark.asyncio
async def test_speech_segments_reject_invalid_end_padding():
    with pytest.raises(ValueError, match="end_padding_ms"):
        segments = room_loop._speech_segments(
            chunks(pcm16(0, 20)),
            sample_rate=1000,
            energy_threshold=350,
            silence_ms=450,
            end_padding_ms=500,
            min_segment_ms=100,
            max_segment_seconds=5.0,
        )
        await anext(segments)


@pytest.mark.asyncio
async def test_barge_in_requires_sustained_voice_and_closes_listener():
    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity="user-a",
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
    )
    bridge = FakeBargeBridge(
        [
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(900, 20),
            pcm16(900, 20),
        ]
    )
    stop_event = room_loop.asyncio.Event()

    await room_loop._wait_for_barge_in(args, bridge, stop_event)

    assert stop_event.is_set()
    assert bridge.iterator.closed is True
    assert bridge.kwargs["participant_identity"] == "user-a"


@pytest.mark.asyncio
async def test_barge_in_resets_on_short_noise_gap():
    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
    )
    bridge = FakeBargeBridge(
        [
            pcm16(900, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(900, 20),
        ]
    )
    stop_event = room_loop.asyncio.Event()

    await room_loop._wait_for_barge_in(args, bridge, stop_event)

    assert not stop_event.is_set()
    assert bridge.iterator.closed is True
