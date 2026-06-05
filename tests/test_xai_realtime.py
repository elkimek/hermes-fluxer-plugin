import base64
import json
import wave

import pytest

import xai_realtime


class FakeRealtimeWebSocket:
    def __init__(self, events):
        self.events = list(events)
        self.sent = []
        self.closed = False

    async def send(self, message):
        self.sent.append(json.loads(message))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        return json.dumps(self.events.pop(0))

    async def close(self):
        self.closed = True


def pcm_delta(data: bytes):
    return {"type": "response.output_audio.delta", "delta": base64.b64encode(data).decode("ascii")}


@pytest.mark.asyncio
async def test_xai_realtime_text_response_writes_pcm_wav(tmp_path):
    ws = FakeRealtimeWebSocket(
        [
            {"type": "session.updated"},
            {"type": "response.created"},
            pcm_delta(b"\x00\x00\x01\x00"),
            {"type": "response.output_audio_transcript.delta", "delta": "hi"},
            {"type": "response.done"},
        ]
    )
    client = xai_realtime.XAIRealtimeVoiceClient(api_key="secret", sample_rate=16000)

    result = await client._text_response_to_wav_on_ws(ws, "say hi", tmp_path / "out.wav")

    assert ws.sent[0]["type"] == "session.update"
    assert ws.sent[0]["session"]["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 16000}
    assert ws.sent[1]["type"] == "conversation.item.create"
    assert ws.sent[1]["item"]["content"][0] == {"type": "input_text", "text": "say hi"}
    assert ws.sent[2] == {"type": "response.create"}
    assert result.bytes_written == 4
    assert result.transcript == "hi"
    with wave.open(str(result.wav_path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.readframes(2) == b"\x00\x00\x01\x00"


@pytest.mark.asyncio
async def test_xai_realtime_force_message_does_not_send_response_create(tmp_path):
    ws = FakeRealtimeWebSocket([pcm_delta(b"\x00\x00"), {"type": "response.done"}])
    client = xai_realtime.XAIRealtimeVoiceClient(api_key="secret")

    await client._force_message_to_wav_on_ws(ws, "exact words", tmp_path / "force.wav", interruptible=False)

    assert ws.sent[1]["type"] == "conversation.item.create"
    assert ws.sent[1]["item"]["type"] == "force_message"
    assert ws.sent[1]["item"]["content"][0] == {"type": "output_text", "text": "exact words"}
    assert all(message.get("type") != "response.create" for message in ws.sent)


@pytest.mark.asyncio
async def test_xai_realtime_audio_response_appends_commits_and_creates_response(tmp_path):
    ws = FakeRealtimeWebSocket([pcm_delta(b"\x02\x00"), {"type": "response.done"}])
    client = xai_realtime.XAIRealtimeVoiceClient(api_key="secret", sample_rate=24000)

    result = await client._audio_response_from_pcm16_on_ws(ws, b"\x10\x00\x20\x00", tmp_path / "audio.wav")

    assert ws.sent[1]["type"] == "input_audio_buffer.append"
    assert base64.b64decode(ws.sent[1]["audio"]) == b"\x10\x00\x20\x00"
    assert ws.sent[2] == {"type": "input_audio_buffer.commit"}
    assert ws.sent[3] == {"type": "response.create"}
    assert result.bytes_written == 2


@pytest.mark.asyncio
async def test_xai_realtime_raises_on_error_event(tmp_path):
    ws = FakeRealtimeWebSocket([{"type": "error", "error": {"message": "bad request"}}])
    client = xai_realtime.XAIRealtimeVoiceClient(api_key="secret")

    with pytest.raises(RuntimeError, match="bad request"):
        await client._text_response_to_wav_on_ws(ws, "hello", tmp_path / "out.wav")


def test_xai_realtime_requires_text_and_key(tmp_path, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    client = xai_realtime.XAIRealtimeVoiceClient(api_key="")

    with pytest.raises(RuntimeError, match="XAI_API_KEY"):
        import asyncio

        asyncio.run(client.text_response_to_wav("hello", tmp_path / "out.wav"))

    client = xai_realtime.XAIRealtimeVoiceClient(api_key="secret")
    with pytest.raises(ValueError, match="text"):
        import asyncio

        asyncio.run(client.text_response_to_wav("", tmp_path / "out.wav"))
