"""Offline tests for the Kokoro speech client (no network access)."""

from __future__ import annotations

import asyncio
import json

from kokoro_tts import (
    KokoroSpeechClient,
    build_kokoro_speech_request,
)


def install_stream(monkeypatch, *, chunks=(), delay=0, status=200):
    """Exercise the real async HTTP client with a controlled byte stream."""
    import httpx

    class Stream(httpx.AsyncByteStream):
        closed = False
        calls = 0

        async def __aiter__(self):
            for chunk in chunks:
                self.calls += 1
                if delay:
                    await asyncio.sleep(delay)
                yield chunk

        async def aclose(self):
            self.closed = True

    stream = Stream()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(status, stream=stream)
    )
    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        "kokoro_tts.httpx.AsyncClient",
        lambda **kwargs: client_factory(transport=transport, **kwargs),
    )
    return stream


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
    install_stream(
        monkeypatch, chunks=[pcm[i : i + 4096] for i in range(0, len(pcm), 4096)]
    )

    async def _run():
        chunks: list[bytes] = []

        async def sink(data: bytes) -> None:
            chunks.append(data)

        client = KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        )
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
    install_stream(
        monkeypatch, chunks=[pcm[i : i + 4096] for i in range(0, len(pcm), 4096)]
    )

    async def _run():
        chunks: list[bytes] = []

        async def sink(data: bytes) -> None:
            chunks.append(data)

        client = KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        )
        result = await client.force_message_to_sink("hello", sink, timeout=5.0)
        return result, chunks

    result, chunks = asyncio.run(_run())

    assert result.bytes_written == len(pcm)
    assert b"".join(chunks) == pcm


def test_empty_text_is_rejected():
    client = KokoroSpeechClient(base_url="http://kokoro.test/v1", voice="test-voice")

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
        KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice", sample_rate=16_000
        )
    # 24 kHz is what the service emits, so the default and an explicit value pass.
    assert (
        KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice", sample_rate=24_000
        ).sample_rate
        == 24_000
    )
    assert (
        KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        ).sample_rate
        == 24_000
    )


def test_a_stream_that_stalls_after_the_first_chunk_fails_the_reply(monkeypatch):
    import pytest
    import httpx

    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            yield b"\x00\x01" * 32
            await asyncio.sleep(0.6)

        async def aclose(self):
            self.closed = True

    stream = Stream()
    factory = httpx.AsyncClient
    monkeypatch.setattr(
        "kokoro_tts.httpx.AsyncClient",
        lambda **kw: factory(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(200, stream=stream)
            ),
            **kw,
        ),
    )

    async def run():
        written = []

        async def sink(chunk):
            written.append(chunk)

        client = KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        )
        with pytest.raises(RuntimeError, match="did not finish within"):
            await client.text_response_to_sink("hello", sink, timeout=0.15)
        assert written
        assert stream.closed

    asyncio.run(run())


def test_cancelling_stops_the_transport_without_reading_ahead(monkeypatch):
    import pytest

    stream = install_stream(monkeypatch, chunks=[b"ab"] * 1000)

    async def run():
        entered = asyncio.Event()

        async def sink(chunk):
            entered.set()
            await asyncio.sleep(10)

        client = KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        )
        task = asyncio.create_task(client.text_response_to_sink("hello", sink))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.calls == 1
        assert stream.closed

    asyncio.run(run())


def test_requires_a_url_and_a_voice(monkeypatch):
    import pytest

    for name in ("FLUXER_VOICE_KOKORO_URL", "FLUXER_VOICE_KOKORO_VOICE"):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValueError, match="FLUXER_VOICE_KOKORO_URL"):
        KokoroSpeechClient()
    with pytest.raises(ValueError, match="FLUXER_VOICE_KOKORO_VOICE"):
        KokoroSpeechClient(base_url="http://kokoro.test/v1")

    # Nothing deployment-specific is compiled in: the env values are what runs.
    monkeypatch.setenv("FLUXER_VOICE_KOKORO_URL", "http://tts.example/v1/")
    monkeypatch.setenv("FLUXER_VOICE_KOKORO_VOICE", "some-voice")
    client = KokoroSpeechClient()
    assert client.base_url == "http://tts.example/v1"
    assert client.voice == "some-voice"


def test_a_slow_sink_loses_no_audio_when_the_queue_fills(monkeypatch):
    chunks = [bytes([index]) * 4096 for index in range(8)]

    install_stream(monkeypatch, chunks=chunks)

    async def _run():
        client = KokoroSpeechClient(
            base_url="http://kokoro.test/v1", voice="test-voice"
        )
        received: list[bytes] = []

        async def _sink(chunk: bytes) -> None:
            await asyncio.sleep(0.005)
            received.append(chunk)

        return await client.text_response_to_sink("hello", _sink, timeout=5.0), received

    result, received = asyncio.run(_run())
    assert received == chunks
    assert "kokoro_speech_complete" in result.events_seen
    assert result.bytes_written == sum(len(chunk) for chunk in chunks)
