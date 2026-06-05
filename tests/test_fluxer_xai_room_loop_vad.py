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


@pytest.mark.asyncio
async def test_barge_in_capture_sets_interrupt_early_and_retains_utterance_pcm():
    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity="user-a",
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
        silence_ms=80,
        end_padding_ms=20,
        min_segment_ms=60,
        max_segment_seconds=2.0,
    )
    bridge = FakeBargeBridge(
        [
            pcm16(900, 20),
            pcm16(900, 20),
            pcm16(900, 20),  # interrupt should fire here at 60ms
            pcm16(900, 20),  # but capture should keep the rest of the utterance
            pcm16(0, 20),
            pcm16(0, 20),
            pcm16(0, 20),
            pcm16(0, 20),
        ]
    )
    capture = room_loop.BargeInCapture()
    task = room_loop.asyncio.create_task(room_loop._wait_for_barge_in(args, bridge, capture))

    for _ in range(10):
        if capture.event.is_set():
            break
        await room_loop.asyncio.sleep(0)

    assert capture.event.is_set()

    await task

    # 80ms speech plus only 20ms of retained final silence.
    assert capture.pcm == pcm16(900, 80) + pcm16(0, 20)
    assert capture.captured_audio_seconds == pytest.approx(0.1)
    assert bridge.iterator.closed is True


@pytest.mark.asyncio
async def test_conversation_loop_reuses_barge_in_pcm_as_next_turn(monkeypatch):
    args = argparse.Namespace(
        max_runtime_seconds=10.0,
        max_turns=1,
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        energy_threshold=350,
        silence_ms=80,
        end_padding_ms=20,
        min_segment_ms=60,
        max_segment_seconds=2.0,
        disable_wake_gate=True,
        xai_model="grok-voice-latest",
        xai_voice="eve",
        xai_instructions="test",
        wake_gate_instructions="gate",
        xai_timeout=5.0,
        xai_first_audio_timeout=5.0,
        disable_barge_in=False,
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
        barge_in_capture_timeout=2.0,
    )
    first_prompt = pcm16(1200, 80) + pcm16(0, 20)
    interrupt_prompt = pcm16(900, 80) + pcm16(0, 20)

    class FakePublisher:
        def __init__(self):
            self.interrupted = False
            self.bytes_published = 0

        async def __aenter__(self):
            return self

        async def write(self, chunk):
            self.bytes_published += len(chunk)

        async def interrupt(self):
            self.interrupted = True

        async def close(self, **kwargs):
            pass

    class FakeBridge:
        def __init__(self):
            self.capture_calls = 0
            self.barge_calls = 0
            self.publishers = []

        def iter_remote_audio_pcm16(self, **kwargs):
            if self.capture_calls == 0:
                self.capture_calls += 1
                return FakeChunkIterator(
                    [
                        pcm16(1200, 20),
                        pcm16(1200, 20),
                        pcm16(1200, 20),
                        pcm16(1200, 20),
                        pcm16(0, 20),
                        pcm16(0, 20),
                        pcm16(0, 20),
                        pcm16(0, 20),
                    ]
                )
            self.barge_calls += 1
            if self.barge_calls > 1:
                return FakeChunkIterator([])
            return FakeChunkIterator(
                [
                    pcm16(900, 20),
                    pcm16(900, 20),
                    pcm16(900, 20),
                    pcm16(900, 20),
                    pcm16(0, 20),
                    pcm16(0, 20),
                    pcm16(0, 20),
                    pcm16(0, 20),
                ]
            )

        def pcm16_publisher(self, **kwargs):
            publisher = FakePublisher()
            self.publishers.append(publisher)
            return publisher

    class FakeXAIResult:
        def __init__(self, payload):
            self.bytes_written = len(payload)
            self.transcript = "ok"
            self.events_seen = ["response.done"]

    class FakeXAI:
        prompts = []

        def __init__(self, **kwargs):
            pass

        async def audio_response_from_pcm16_to_sink(self, pcm, sink, **kwargs):
            self.prompts.append(pcm)
            await sink(pcm16(300, 20))
            await room_loop.asyncio.sleep(0)
            await sink(pcm16(300, 20))
            return FakeXAIResult(pcm)

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)
    FakeXAI.prompts = []

    result = await room_loop._conversation_loop(args, FakeBridge())

    assert [turn.get("interrupted") for turn in result["turns"]] == [True, None]
    assert FakeXAI.prompts == [first_prompt, interrupt_prompt]
    assert result["turns"][0]["barge_in_carryover_pcm_bytes"] == len(interrupt_prompt)
    assert result["turns"][1]["from_barge_in_carryover"] is True
