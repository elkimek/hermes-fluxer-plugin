import argparse
import asyncio
import inspect

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
async def test_close_async_generator_safely_suppresses_already_running_cleanup_race():
    class AlreadyRunningGenerator:
        async def aclose(self):
            raise RuntimeError("aclose(): asynchronous generator is already running")

    await room_loop._close_async_generator_safely(AlreadyRunningGenerator())


@pytest.mark.asyncio
async def test_close_async_generator_safely_reraises_other_runtime_errors():
    class BrokenGenerator:
        async def aclose(self):
            raise RuntimeError("different cleanup failure")

    with pytest.raises(RuntimeError, match="different cleanup failure"):
        await room_loop._close_async_generator_safely(BrokenGenerator())


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
async def test_speech_segments_yields_final_partial_segment_when_stream_ends():
    frames = [pcm16(1200, 20) for _ in range(5)]  # 100ms speech, then stream ends.

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

    assert segment == pcm16(1200, 100)


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
async def test_barge_in_accumulates_bursty_voice_inside_configured_window():
    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        barge_in_energy_threshold=700,
        barge_in_min_ms=200,
        barge_in_window_ms=1200,
    )
    # Browser/Fluxer voice activity can blink on/off while the user is really
    # speaking. Ten 20ms voiced chunks separated by short gaps should count as
    # one barge-in attempt even though no chunk run is 200ms continuous.
    bridge = FakeBargeBridge(
        [
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
            pcm16(0, 20),
            pcm16(900, 20),
        ]
    )
    capture = room_loop.BargeInCapture()

    await room_loop._wait_for_barge_in(args, bridge, capture)

    assert capture.event.is_set()
    assert capture.detected_voiced_ms == 200
    assert capture.voiced_ms == 200
    assert bridge.iterator.closed is True


@pytest.mark.asyncio
async def test_barge_in_ignores_short_echo_bursts_below_windowed_minimum():
    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        barge_in_energy_threshold=180,
        barge_in_min_ms=200,
        barge_in_window_ms=1200,
    )
    # Soundbar/speaker echo can create loud but short bursts while the assistant
    # is speaking. Four 20ms bursts should not interrupt even though each is
    # above the energy threshold.
    bridge = FakeBargeBridge(
        [
            pcm16(500, 20),
            pcm16(0, 120),
            pcm16(500, 20),
            pcm16(0, 120),
            pcm16(500, 20),
            pcm16(0, 120),
            pcm16(500, 20),
        ]
    )
    capture = room_loop.BargeInCapture()

    await room_loop._wait_for_barge_in(args, bridge, capture)

    assert not capture.event.is_set()
    assert capture.voiced_ms == 80
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
        max_turns=2,
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

        async def __aexit__(self, exc_type, exc, tb):
            await self.close()

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
            for _ in range(6):
                await sink(pcm16(300, 20))
                await room_loop.asyncio.sleep(0)
            return FakeXAIResult(pcm)

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)
    FakeXAI.prompts = []

    result = await room_loop._conversation_loop(args, FakeBridge())

    assert [turn.get("interrupted") for turn in result["turns"]] == [True, None]
    assert FakeXAI.prompts == [first_prompt, interrupt_prompt]
    assert result["turns"][0]["barge_in_carryover_pcm_bytes"] == len(interrupt_prompt)
    assert result["turns"][0]["barge_in_carryover_discarded"] is False
    assert result["turns"][1]["from_barge_in_carryover"] is True


@pytest.mark.asyncio
async def test_conversation_loop_max_turns_counts_interrupted_turns(monkeypatch):
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

        async def __aexit__(self, exc_type, exc, tb):
            await self.close()

        async def write(self, chunk):
            self.bytes_published += len(chunk)

        async def interrupt(self):
            self.interrupted = True

        async def close(self, **kwargs):
            pass

    class FakeBridge:
        def __init__(self):
            self.capture_calls = 0

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
            return FakePublisher()

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
            for _ in range(6):
                await sink(pcm16(300, 20))
                await room_loop.asyncio.sleep(0)
            return FakeXAIResult(pcm)

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)
    FakeXAI.prompts = []

    result = await room_loop._conversation_loop(args, FakeBridge())

    assert len(result["turns"]) == 1
    assert result["turn_count"] == 1
    assert result["turns"][0]["interrupted"] is True
    assert result["turns"][0]["barge_in_carryover_pcm_bytes"] == len(interrupt_prompt)
    assert FakeXAI.prompts == [first_prompt]


def test_barge_in_carryover_decision_discards_tiny_interrupt_fragment():
    args = argparse.Namespace(min_segment_ms=750)
    tiny_stop_fragment = pcm16(900, 260)
    usable, discarded, duration = room_loop._barge_in_carryover_decision(
        args,
        tiny_stop_fragment,
        sample_rate=1000,
    )

    assert usable is None
    assert discarded is True
    assert duration == pytest.approx(0.26)


def test_barge_in_carryover_decision_reuses_full_interrupt_utterance():
    args = argparse.Namespace(min_segment_ms=750)
    full_utterance = pcm16(900, 900)
    usable, discarded, duration = room_loop._barge_in_carryover_decision(
        args,
        full_utterance,
        sample_rate=1000,
    )

    assert usable == full_utterance
    assert discarded is False
    assert duration == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_conversation_loop_exits_cleanly_when_remote_stream_ends(monkeypatch):
    args = argparse.Namespace(
        max_runtime_seconds=10.0,
        max_turns=2,
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
        disable_barge_in=True,
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
        barge_in_capture_timeout=2.0,
    )

    class FakeBridge:
        def iter_remote_audio_pcm16(self, **kwargs):
            return FakeChunkIterator([])

    class FakeXAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)

    result = await room_loop._conversation_loop(args, FakeBridge())

    assert result["turns"] == []
    assert result["turn_count"] == 0


@pytest.mark.asyncio
async def test_conversation_loop_exits_cleanly_on_asyncio_capture_timeout(monkeypatch):
    args = argparse.Namespace(
        max_runtime_seconds=10.0,
        max_turns=2,
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
        disable_barge_in=True,
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
        barge_in_capture_timeout=2.0,
    )

    class FakeBridge:
        pass

    async def timeout_capture(*args, **kwargs):
        raise asyncio.TimeoutError

    class FakeXAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)
    monkeypatch.setattr(room_loop, "_capture_one_speech_segment", timeout_capture)

    result = await room_loop._conversation_loop(args, FakeBridge())  # type: ignore[arg-type]

    assert result["turns"] == []
    assert result["turn_count"] == 0


@pytest.mark.asyncio
async def test_cancel_previous_task_safely_propagates_own_cancellation():
    previous_cleanup_started = asyncio.Event()

    async def slow_previous_task():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            previous_cleanup_started.set()
            await asyncio.sleep(10)
            raise

    previous_task = asyncio.create_task(slow_previous_task())
    helper_task = asyncio.create_task(room_loop._cancel_previous_task_safely(previous_task))
    await asyncio.wait_for(previous_cleanup_started.wait(), timeout=0.2)

    helper_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await helper_task

    previous_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await previous_task


@pytest.mark.asyncio
async def test_conversation_loop_closes_publisher_when_enter_fails(monkeypatch):
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
        disable_barge_in=True,
        barge_in_energy_threshold=700,
        barge_in_min_ms=60,
        barge_in_capture_timeout=2.0,
    )

    class FailingPublisher:
        interrupted = False

        def __init__(self):
            self.closed = False

        async def __aenter__(self):
            await self.close()
            raise RuntimeError("publish failed")

        async def __aexit__(self, exc_type, exc, tb):
            await self.close()

        async def close(self, **kwargs):
            self.closed = True

    class FakeBridge:
        def __init__(self):
            self.publisher = FailingPublisher()

        def iter_remote_audio_pcm16(self, **kwargs):
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

        def pcm16_publisher(self, **kwargs):
            return self.publisher

    class FakeXAI:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(room_loop, "XAIRealtimeVoiceClient", FakeXAI)
    bridge = FakeBridge()

    result = await room_loop._conversation_loop(args, bridge)

    assert bridge.publisher.closed is True
    assert result["turns"][0]["published"] is False
    assert result["turns"][0]["error"] == "RuntimeError: publish failed"


@pytest.mark.asyncio
async def test_run_reports_voice_server_update_timeout_without_traceback(monkeypatch, capsys):
    monkeypatch.setenv("FLUXER_BOT_TOKEN", "test-token")

    class FakeAdapter:
        def __init__(self, config):
            self.handler = None
            self.left = False
            self.disconnected = False

        def set_voice_server_update_handler(self, handler):
            self.handler = handler

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            return True

        async def send_voice_state_update(self, channel_id, **kwargs):
            if channel_id is None:
                self.left = True
            return True

        async def disconnect(self):
            self.disconnected = True

    class FakeBridge:
        def __init__(self, *args, **kwargs):
            self.disconnected = False

        async def disconnect(self):
            self.disconnected = True

    monkeypatch.setattr(room_loop, "FluxerAdapter", FakeAdapter)
    monkeypatch.setattr(room_loop, "FluxerLiveKitSmokeBridge", FakeBridge)
    args = argparse.Namespace(
        verbose=False,
        channel_id="voice-1",
        guild_id="guild-1",
        unmute=False,
        connect_timeout=0.001,
        diagnose_barge_in=False,
        diagnose_seconds=1.0,
        max_runtime_seconds=1.0,
    )

    exit_code = await room_loop.run(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "No VOICE_SERVER_UPDATE received within 0.001s" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("diagnose_barge_in", "target_name", "expected"),
    [
        (True, "_diagnose_barge_in", "Barge-in diagnostic exceeded safety timeout"),
        (False, "_conversation_loop", "Conversation loop exceeded safety timeout"),
    ],
)
async def test_run_reports_outer_safety_timeouts_without_traceback(monkeypatch, capsys, diagnose_barge_in, target_name, expected):
    from types import SimpleNamespace

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "test-token")

    class FakeAdapter:
        def __init__(self, config):
            self.handler = None

        def set_voice_server_update_handler(self, handler):
            self.handler = handler

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            return True

        async def send_voice_state_update(self, channel_id, **kwargs):
            if channel_id is not None and self.handler is not None:
                await self.handler(
                    {"endpoint": "wss://livekit.example", "token": "ephemeral"},
                    {"endpoint": "wss://livekit.example", "has_token": True},
                )
            return True

        async def disconnect(self):
            pass

    class FakeBridge:
        def __init__(self, *args, **kwargs):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            return SimpleNamespace(
                endpoint="wss://livekit.example",
                guild_id="guild-1",
                channel_id="voice-1",
                connection_id="conn-1",
                room_name="room",
                participant_identity="bot",
            )

        async def disconnect(self):
            pass

    async def timeout_target(*args, **kwargs):
        raise asyncio.TimeoutError

    monkeypatch.setattr(room_loop, "FluxerAdapter", FakeAdapter)
    monkeypatch.setattr(room_loop, "FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr(room_loop, target_name, timeout_target)
    args = argparse.Namespace(
        verbose=False,
        channel_id="voice-1",
        guild_id="guild-1",
        unmute=False,
        connect_timeout=1.0,
        diagnose_barge_in=diagnose_barge_in,
        diagnose_seconds=1.0,
        max_runtime_seconds=1.0,
    )

    exit_code = await room_loop.run(args)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert expected in captured.err
    assert "Traceback" not in captured.err


def test_redact_exception_message_removes_livekit_tokens():
    exc = RuntimeError('connect failed Bearer abc.def token=secret123 {"token":"json-secret"}')

    message = room_loop._redact_exception_message(exc, "abc.def", "secret123", "json-secret")

    assert "abc.def" not in message
    assert "secret123" not in message
    assert "json-secret" not in message
    assert "[redacted-token]" in message


def test_xai_room_loop_redacts_turn_level_error_result():
    source = inspect.getsource(room_loop._conversation_loop)

    assert "error_text = str(exc)" not in source
    assert "error_text = _redact_exception_message(exc)" in source


def test_diagnose_barge_in_publish_task_cleanup_suppresses_closed_publisher_race():
    source = (room_loop.ROOT / "scripts" / "fluxer_xai_room_loop.py").read_text(encoding="utf-8")

    assert "contextlib.suppress(asyncio.CancelledError, RuntimeError, Exception)" in source


def test_xai_room_loop_voice_server_handler_schedules_livekit_connect_task():
    source = inspect.getsource(room_loop.run)

    handler_start = source.index("async def on_voice_server_update")
    handler_source = source[handler_start : source.index("adapter.set_voice_server_update_handler", handler_start)]

    assert "asyncio.create_task(" in handler_source
    assert "run_voice_update_after_previous(previous_task, raw_update, safe_update)" in handler_source
    assert "voice_update_lock = asyncio.Lock()" in source
    assert "_acquire_lock_with_timeout(voice_update_lock, timeout=args.connect_timeout)" in source
    assert "voice_update_lock.release()" in source
    assert "await bridge.connect_from_voice_server_update" not in handler_source
    assert "await voice_update_task" not in handler_source


def test_xai_room_loop_diagnostic_suppresses_publish_task_exceptions():
    source = inspect.getsource(room_loop._diagnose_barge_in)

    assert "contextlib.suppress(asyncio.CancelledError, RuntimeError, Exception)" in source


def test_xai_room_loop_diagnostic_publish_tone_uses_context_manager():
    source = inspect.getsource(room_loop._diagnose_barge_in)

    publish_start = source.index("async def publish_tone")
    publish_source = source[publish_start : source.index("publish_task =", publish_start)]

    assert "async with publisher:" in publish_source
    assert "await publisher.__aenter__()" not in publish_source


def test_xai_room_loop_conversation_publisher_uses_context_manager():
    source = inspect.getsource(room_loop._conversation_loop)

    assert "async with publisher:" in source
    assert "await publisher.__aenter__()" not in source


def test_xai_room_loop_cancels_xai_task_before_publisher_close():
    source = inspect.getsource(room_loop._conversation_loop)

    cancel_xai = source.index("await _cancel_task_safely(xai_task)")
    close_publisher = source.index("await publisher.close", cancel_xai)

    assert cancel_xai < close_publisher

@pytest.mark.asyncio
async def test_barge_in_stop_phrase_fast_path_interrupts_below_generic_threshold():
    async def transcribe_stop(pcm: bytes) -> str:
        assert pcm
        return "please stop counting"

    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        barge_in_energy_threshold=700,
        barge_in_min_ms=300,
        barge_in_window_ms=1200,
        barge_in_stop_phrase_energy_threshold=300,
        barge_in_stop_phrase_min_ms=120,
        barge_in_stop_phrase_silence_ms=60,
        barge_in_stop_phrase_transcriber=transcribe_stop,
    )
    bridge = FakeBargeBridge(
        [
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(0, 20),
            pcm16(0, 20),
            pcm16(0, 20),
        ]
    )
    capture = room_loop.BargeInCapture()

    await room_loop._wait_for_barge_in(args, bridge, capture)

    assert capture.event.is_set()
    assert capture.semantic_stop_detected is True
    assert capture.semantic_stop_transcript == "please stop counting"
    assert capture.detected_voiced_ms == 120
    assert bridge.iterator.closed is True


@pytest.mark.asyncio
async def test_barge_in_stop_phrase_fast_path_ignores_non_stop_echo():
    async def transcribe_echo(pcm: bytes) -> str:
        assert pcm
        return "one two three four"

    args = argparse.Namespace(
        sample_rate=1000,
        frame_ms=20,
        participant_identity=None,
        barge_in_energy_threshold=700,
        barge_in_min_ms=300,
        barge_in_window_ms=1200,
        barge_in_stop_phrase_energy_threshold=300,
        barge_in_stop_phrase_min_ms=120,
        barge_in_stop_phrase_silence_ms=60,
        barge_in_stop_phrase_transcriber=transcribe_echo,
    )
    bridge = FakeBargeBridge(
        [
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(420, 20),
            pcm16(0, 20),
            pcm16(0, 20),
            pcm16(0, 20),
        ]
    )
    capture = room_loop.BargeInCapture()

    await room_loop._wait_for_barge_in(args, bridge, capture)

    assert not capture.event.is_set()
    assert capture.semantic_stop_detected is False
    assert capture.semantic_stop_transcript == "one two three four"
    assert bridge.iterator.closed is True
