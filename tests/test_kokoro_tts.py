"""Offline tests for the Kokoro speech client (no network access)."""

from __future__ import annotations

import asyncio
import io
import json

from kokoro_tts import (
    KokoroSpeechClient,
    build_kokoro_speech_request,
)


def _fake_response(data: bytes, status: int = 200):
    class _Resp:
        def __init__(self) -> None:
            self.status = status
            self._buf = io.BytesIO(data)

        def read(self, size: int = -1) -> bytes:
            return self._buf.read(size)

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    return _Resp()


def test_build_kokoro_speech_request_uses_openai_speech_shape():
    request = build_kokoro_speech_request(
        "hello", voice="af_bella", model="kokoro", base_url="http://kokoro.test/v1"
    )

    assert request.full_url == "http://kokoro.test/v1/audio/speech"
    assert request.get_header("Content-type") == "application/json"
    body = request.data
    assert isinstance(body, bytes)
    assert json.loads(body.decode("utf-8")) == {
        "model": "kokoro",
        "input": "hello",
        "voice": "af_bella",
        "response_format": "pcm",
        "stream": True,
    }


def test_streams_pcm_chunks_to_sink(monkeypatch):
    pcm = bytes(range(256)) * 64
    monkeypatch.setattr(
        "kokoro_tts.urllib.request.urlopen",
        lambda request, timeout=None: _fake_response(pcm),
    )

    async def _run():
        chunks: list[bytes] = []

        async def sink(data: bytes) -> None:
            chunks.append(data)

        client = KokoroSpeechClient(base_url="http://kokoro.test/v1")
        result = await client.text_response_to_sink("hello", sink, timeout=5.0)
        return result, chunks

    result, chunks = asyncio.run(_run())

    assert result.bytes_written == len(pcm)
    assert b"".join(chunks) == pcm
    assert result.events_seen[0] == "kokoro_speech_request"
    assert "http_200" in result.events_seen
    assert "first_audio_chunk" in result.events_seen
    assert result.events_seen[-1] == "kokoro_speech_complete"
    assert result.transcript == "hello"


def test_force_message_to_sink_shares_streaming_path(monkeypatch):
    pcm = b"\x00\x01" * 2048
    monkeypatch.setattr(
        "kokoro_tts.urllib.request.urlopen",
        lambda request, timeout=None: _fake_response(pcm),
    )

    async def _run():
        chunks: list[bytes] = []

        async def sink(data: bytes) -> None:
            chunks.append(data)

        client = KokoroSpeechClient(base_url="http://kokoro.test/v1")
        result = await client.force_message_to_sink("hello", sink, timeout=5.0)
        return result, chunks

    result, chunks = asyncio.run(_run())

    assert result.bytes_written == len(pcm)
    assert b"".join(chunks) == pcm


def test_empty_text_is_rejected():
    client = KokoroSpeechClient(base_url="http://kokoro.test/v1")

    async def _run():
        async def sink(data: bytes) -> None:  # pragma: no cover - never called
            raise AssertionError("sink must not be called for empty text")

        await client.text_response_to_sink("   ", sink)

    try:
        asyncio.run(_run())
    except ValueError as exc:
        assert "empty" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected ValueError for empty text")


def test_rejects_a_sample_rate_kokoro_cannot_serve():
    import pytest

    with pytest.raises(ValueError, match="cannot\\s+resample"):
        KokoroSpeechClient(sample_rate=16_000)
    # 24 kHz is what the service emits, so the default and an explicit value pass.
    assert KokoroSpeechClient(sample_rate=24_000).sample_rate == 24_000
    assert KokoroSpeechClient().sample_rate == 24_000


def test_a_stream_that_stalls_after_the_first_chunk_fails_the_reply(monkeypatch):
    import pytest
    import time

    class _Stalling:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"\x00\x01" * 32
            time.sleep(0.6)
            return b""

        def __enter__(self) -> "_Stalling":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    monkeypatch.setattr(
        "kokoro_tts.urllib.request.urlopen",
        lambda request, timeout=None: _Stalling(),
    )

    async def _run():
        client = KokoroSpeechClient(base_url="http://kokoro.test/v1")
        written: list[bytes] = []

        async def _sink(chunk: bytes) -> None:
            written.append(chunk)

        with pytest.raises(RuntimeError, match="did not finish within"):
            await client.text_response_to_sink("hello", _sink, timeout=0.15)
        return written

    # The first chunk was delivered, so this is a partial reply, not silence.
    assert asyncio.run(_run())


def test_cancelling_stops_the_producer_at_the_next_chunk(monkeypatch):
    import time

    class _Counting:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, size: int = -1) -> bytes:
            self.calls += 1
            if self.calls == 1:
                return b"\x00\x01" * 32
            time.sleep(0.25)
            return b"\x00\x01" * 32

        def __enter__(self) -> "_Counting":
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    response = _Counting()
    monkeypatch.setattr(
        "kokoro_tts.urllib.request.urlopen",
        lambda request, timeout=None: response,
    )

    async def _run():
        client = KokoroSpeechClient(base_url="http://kokoro.test/v1")

        async def _sink(chunk: bytes) -> None:
            await asyncio.sleep(0.5)  # a sink that never keeps up

        task = asyncio.create_task(client.text_response_to_sink("hello", _sink, timeout=30.0))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        calls_after_cancel = response.calls
        await asyncio.sleep(0.4)
        return calls_after_cancel, response.calls

    before, after = asyncio.run(_run())
    # The stop flag ends the request at the next chunk boundary instead of the
    # producer reading to EOF into a queue nobody drains.
    assert after <= before
