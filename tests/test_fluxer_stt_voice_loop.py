from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import sqlite3
import wave
from datetime import datetime
from types import SimpleNamespace

import pytest

from scripts.fluxer_stt_voice_loop import (
    append_jsonl,
    build_answer_prompt,
    build_hermes_messages,
    _redact_exception_message,
    collect_voice_session_recall,
    compose_system_prompt,
    hermes_voice_session_identity,
    hermes_chat_completion,
    is_voice_stop_request,
    load_env_file,
    load_voice_context_cache,
    looks_like_clipped_non_english_noise,
    normalize_voice_transcript,
    parse_args,
    pcm16_rms,
    requested_brain_mode_switch,
    resolve_voice_brain_provider,
    run_stt_voice_loop,
    safe_stt_summary,
    transcribe_with_provider,
    voice_mode_ack,
    voice_recall_time_window,
    write_pcm16_wav,
)


def test_build_answer_prompt_grounds_latest_transcript_and_history():
    prompt = build_answer_prompt(
        "Shevka, what is two past two?",
        history=[{"user": "hello", "assistant": "Hi."}],
    )

    assert "Latest STT transcript from the user: 'Shevka, what is two past two?'" in prompt
    assert "the user: hello" in prompt
    assert "the assistant: Hi." in prompt
    assert "past" in prompt and "plus" in prompt
    assert "plast" in prompt and "plus" in prompt
    assert "participant-targeted capture" in prompt
    assert "realtime TTS" in prompt
    assert "configured Hermes assistant" in prompt
    assert "Speak English by default" not in prompt
    assert "explicitly asks for another language" in prompt
    assert "deep, personal" in prompt
    assert "2-4 substantive spoken sentences" in prompt
    assert "do not end with a generic follow-up question" in prompt
    assert "do not guess from cached context" in prompt


def test_build_hermes_messages_preserves_history_and_latest_transcript():
    messages = build_hermes_messages(
        "what are we building?",
        history=[{"user": "hello", "assistant": "Hi."}],
        system="system context",
    )

    assert messages == [
        {"role": "system", "content": "system context"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi."},
        {"role": "user", "content": "what are we building?"},
    ]


def test_compose_system_prompt_appends_cached_context():
    prompt = compose_system_prompt("base", voice_context_cache="the user likes direct answers")

    assert "base" in prompt
    assert "Cached deployment-local context" in prompt
    assert "the user likes direct answers" in prompt


def test_hermes_voice_session_identity_derives_room_scoped_headers():
    args = argparse.Namespace(
        guild_id="guild/one",
        channel_id="voice channel",
        participant_identity_prefix="user_123_",
        hermes_session_id="",
        hermes_session_key="",
    )

    session_id, session_key = hermes_voice_session_identity(args)

    assert session_id == "fluxer-voice-guild-one-voice-channel-user_123"
    assert session_key == "fluxer:voice:guild:guild-one:channel:voice-channel:participant:user_123"
    assert len(session_id) <= 64
    assert len(session_key) <= 256


def test_hermes_voice_session_identity_bounds_long_room_ids_for_provider_cache_key():
    args = argparse.Namespace(
        guild_id="1234567890123456789",
        channel_id="2234567890123456789",
        participant_identity_prefix="user_3234567890123456789_",
        hermes_session_id="",
        hermes_session_key="",
    )

    session_id, session_key = hermes_voice_session_identity(args)

    assert session_id.startswith("fluxer-voice-")
    assert len(session_id) <= 64
    assert session_id != "fluxer-voice-1234567890123456789-2234567890123456789-user_3234567890123456789"
    assert session_key == (
        "fluxer:voice:guild:1234567890123456789:channel:2234567890123456789:"
        "participant:user_3234567890123456789"
    )
    assert len(session_key) <= 256


@pytest.mark.asyncio
async def test_hermes_chat_completion_sends_session_headers(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"persisted answer"}}]}'

    def fake_urlopen(req, timeout):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setenv("API_SERVER_KEY", "test-key")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.urllib.request.urlopen", fake_urlopen)
    args = argparse.Namespace(
        hermes_url="http://127.0.0.1:8642",
        hermes_model="Hermes",
        hermes_timeout=12,
        hermes_max_tokens=80,
        hermes_temperature=0.4,
        voice_session_db="/missing/state.db",
        voice_system_prompt="system",
        guild_id="guild1",
        channel_id="channel1",
        participant_identity_prefix="user_42_",
        hermes_session_id="",
        hermes_session_key="",
    )

    reply = await hermes_chat_completion("What mode is this?", history=[], args=args)

    assert reply == "persisted answer"
    assert captured["url"] == "http://127.0.0.1:8642/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["headers"]["X-hermes-session-id"] == "fluxer-voice-guild1-channel1-user_42"
    assert captured["headers"]["X-hermes-session-key"] == "fluxer:voice:guild:guild1:channel:channel1:participant:user_42"


def test_load_voice_context_cache_reads_file_once(tmp_path):
    path = tmp_path / "voice-context.md"
    path.write_text("cached facts", encoding="utf-8")

    assert load_voice_context_cache(str(path)) == "cached facts"
    assert load_voice_context_cache(str(tmp_path / "missing.md")) == ""


def test_normalize_voice_transcript_strips_recalled_memory_context():
    transcript = "Yeah, I just want you to improve\n\n<memory-context>\n[System note: recalled memory, not speech]\nPrivate facts and long context.\n</memory-context>"

    assert normalize_voice_transcript(transcript) == "Yeah, I just want you to improve"


def test_build_hermes_messages_does_not_treat_memory_context_as_user_speech():
    messages = build_hermes_messages(
        "Improve it <memory-context>not spoken</memory-context>",
        history=[],
        system="system context",
    )

    assert messages[-1] == {"role": "user", "content": "Improve it"}


def test_is_voice_stop_request_detects_clear_stop_phrases():
    assert is_voice_stop_request("Okay, we can stop here")
    assert is_voice_stop_request("let's stop the voice chat")
    assert is_voice_stop_request("that's enough")
    assert not is_voice_stop_request("stop asking generic questions")


def test_voice_brain_router_auto_escalates_and_supports_spoken_switches():
    assert requested_brain_mode_switch("Okay, switch to full Hermes mode") == "hermes"
    assert requested_brain_mode_switch("Go back to fast mode now") == "xai-fast"
    assert requested_brain_mode_switch("Okay, go back to the fast mode") == "xai-fast"
    assert requested_brain_mode_switch("Switch back to XAI Fest") == "xai-fast"

    provider, sticky, reason = resolve_voice_brain_provider(
        "auto",
        "xai-fast",
        "What were we doing last Monday?",
    )
    assert (provider, sticky, reason) == ("hermes", "xai-fast", "auto_escalate_memory_context")

    provider, sticky, reason = resolve_voice_brain_provider(
        "auto",
        "xai-fast",
        "What were we doing last Friday?",
    )
    assert (provider, sticky, reason) == ("hermes", "xai-fast", "auto_escalate_memory_context")

    provider, sticky, reason = resolve_voice_brain_provider("auto", "xai-fast", "Switch to full brain")
    assert (provider, sticky, reason) == ("hermes", "hermes", "voice_switch_hermes")

    provider, sticky, reason = resolve_voice_brain_provider("auto", sticky, "What is two plus two?")
    assert (provider, sticky, reason) == ("hermes", "hermes", "sticky_hermes")

    provider, sticky, reason = resolve_voice_brain_provider("auto", sticky, "Back to fast mode")
    assert (provider, sticky, reason) == ("xai-fast", "xai-fast", "voice_switch_xai-fast")

    provider, sticky, reason = resolve_voice_brain_provider("auto", sticky, "What is two plus two?")
    assert (provider, sticky, reason) == ("xai-fast", "xai-fast", "auto_fast")

    provider, sticky, reason = resolve_voice_brain_provider("xai-fast", "xai-fast", "Switch to full brain")
    assert (provider, sticky, reason) == ("xai-fast", "xai-fast", "configured_provider_ignores_voice_switch")
    assert "full Hermes brain" in voice_mode_ack("hermes")
    assert "fast voice mode" in voice_mode_ack("xai-fast")


def test_voice_recall_time_window_for_last_weekday():
    now = datetime(2026, 6, 6, 12, 0, 0)
    window = voice_recall_time_window("What were we doing last Monday?", now=now)
    assert window is not None
    start, end, label = window
    assert label == "last Monday"
    assert start.isoformat() == "2026-06-01T00:00:00"
    assert end.isoformat() == "2026-06-02T00:00:00"

    window = voice_recall_time_window("What were we doing last Friday?", now=now)
    assert window is not None
    start, end, label = window
    assert label == "last Friday"
    assert start.isoformat() == "2026-06-05T00:00:00"
    assert end.isoformat() == "2026-06-06T00:00:00"


def test_collect_voice_session_recall_reads_local_state_db(tmp_path):
    db = tmp_path / "state.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        create table sessions (id text primary key, title text, source text, started_at real, ended_at real);
        create table messages (id integer primary key, session_id text, role text, content text, timestamp real);
        """
    )
    con.execute(
        "insert into sessions values (?, ?, ?, ?, ?)",
        (
            "s1",
            "Fluxer work",
            "fluxer",
            datetime(2026, 6, 1, 9, 0, 0).timestamp(),
            datetime(2026, 6, 1, 10, 0, 0).timestamp(),
        ),
    )
    con.execute(
        "insert into messages(session_id, role, content, timestamp) values (?, ?, ?, ?)",
        ("s1", "user", "We built the hybrid voice router", datetime(2026, 6, 1, 9, 30, 0).timestamp()),
    )
    con.commit()
    con.close()

    recall = collect_voice_session_recall(
        "What did we do last Monday?",
        db_path=str(db),
        now=datetime(2026, 6, 6, 12, 0, 0),
    )

    assert "Local Hermes session excerpts for last Monday" in recall
    assert "We built the hybrid voice router" in recall
    assert "Fluxer work" in recall


def test_looks_like_clipped_non_english_noise_rejects_short_vad_hallucinations():
    assert looks_like_clipped_non_english_noise("Je to pračka nebo úplně?")
    assert looks_like_clipped_non_english_noise("E aí")
    assert not looks_like_clipped_non_english_noise("Yeah, I want to get Fluxer talking to you")


def test_append_jsonl_writes_one_turn_per_line(tmp_path):
    path = tmp_path / "turns.jsonl"

    append_jsonl(str(path), {"turn": 1, "transcript": "hello"})
    append_jsonl(str(path), {"turn": 2, "transcript": "world"})

    assert path.read_text(encoding="utf-8").splitlines() == [
        '{"turn": 1, "transcript": "hello"}',
        '{"turn": 2, "transcript": "world"}',
    ]


def test_safe_stt_summary_drops_extra_provider_payload():
    summary = safe_stt_summary(
        {
            "success": True,
            "transcript": "hello <memory-context>not spoken</memory-context>",
            "provider": "local",
            "model": "medium.en",
            "error": None,
            "raw": {"large": "payload"},
        }
    )

    assert summary == {
        "success": True,
        "transcript": "hello",
        "provider": "local",
        "model": "medium.en",
        "error": None,
    }


def test_pcm16_rms_is_pure_python_audioop_replacement():
    assert pcm16_rms(b"") == 0
    assert pcm16_rms((3).to_bytes(2, "little", signed=True) + (4).to_bytes(2, "little", signed=True)) == 3
    assert pcm16_rms((-300).to_bytes(2, "little", signed=True) + (300).to_bytes(2, "little", signed=True)) == 300


def test_parse_args_defaults_to_realtime_voice_stack(monkeypatch):
    for key in (
        "FLUXER_VOICE_SILENCE_MS",
        "FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS",
        "FLUXER_VOICE_INITIAL_SETTLE_SECONDS",
        "FLUXER_VOICE_SAMPLE_RATE",
        "FLUXER_VOICE_FRAME_MS",
        "FLUXER_VOICE_ENERGY_THRESHOLD",
    ):
        monkeypatch.delenv(key, raising=False)
    args = parse_args(["--channel-id", "voice-room"])

    assert args.stt_provider == "elevenlabs"
    assert args.stt_model == "medium.en"
    assert args.elevenlabs_language_code == ""
    assert args.capture_mode == "vad"
    assert args.capture_window_seconds == 3.0
    assert args.brain_provider == "hermes"
    assert args.hermes_url == "http://127.0.0.1:8642"
    assert args.voice_context_file == ""
    assert args.energy_threshold == 300
    assert args.silence_ms == 1500
    assert args.min_segment_ms == 1600
    assert args.max_segment_seconds == 12.0
    assert args.disable_barge_in is False
    assert args.barge_in_energy_threshold == 700
    assert args.barge_in_min_ms == 180
    assert args.barge_in_after_first_audio_only is True


def test_transcribe_with_provider_uses_groq_default_for_local_model_name(monkeypatch, tmp_path):
    seen = {}

    def fake_groq(file_path, model):
        seen["file_path"] = file_path
        seen["model"] = model
        return {"success": True, "transcript": "ok", "provider": "groq"}

    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._transcribe_groq", fake_groq)
    result = transcribe_with_provider(str(tmp_path / "voice.wav"), provider="groq", model="tiny.en")

    assert result["provider"] == "groq"
    assert seen["model"] == "whisper-large-v3-turbo"


def test_transcribe_with_provider_uses_elevenlabs_default_for_local_model_name(monkeypatch, tmp_path):
    seen = {}

    def fake_elevenlabs(file_path, model):
        seen["file_path"] = file_path
        seen["model"] = model
        return {"success": True, "transcript": "ok", "provider": "elevenlabs"}

    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._transcribe_elevenlabs", fake_elevenlabs)
    result = transcribe_with_provider(str(tmp_path / "voice.wav"), provider="elevenlabs", model="medium.en")

    assert result["provider"] == "elevenlabs"
    assert seen["model"] == "scribe_v2"


def test_transcribe_with_provider_overrides_elevenlabs_language_without_global_config(monkeypatch, tmp_path):
    seen = {}

    def fake_language_call(file_path, model, language_code):
        seen["file_path"] = file_path
        seen["model"] = model
        seen["language_code"] = language_code
        return {"success": True, "transcript": "ok", "provider": "elevenlabs"}

    original_loader = object()
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._load_stt_config", original_loader)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.transcribe_elevenlabs_with_language", fake_language_call)

    result = transcribe_with_provider(
        str(tmp_path / "voice.wav"),
        provider="elevenlabs",
        model="scribe_v2",
        elevenlabs_language_code="eng",
    )

    assert result["provider"] == "elevenlabs"
    assert seen["model"] == "scribe_v2"
    assert seen["language_code"] == "eng"
    from scripts import fluxer_stt_voice_loop

    assert fluxer_stt_voice_loop._load_stt_config is original_loader


def test_write_pcm16_wav_roundtrip(tmp_path):
    path = tmp_path / "voice.wav"
    write_pcm16_wav(path, b"\x01\x00\x02\x00", sample_rate=24_000)

    with wave.open(str(path), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 24_000
        assert wav.readframes(2) == b"\x01\x00\x02\x00"


def test_load_env_file_does_not_override_existing_or_shell_source(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP=from_file\nNEW_VALUE='ok'\nBAD LINE WITHOUT EQUALS\n", encoding="utf-8")
    monkeypatch.setenv("KEEP", "existing")
    monkeypatch.delenv("NEW_VALUE", raising=False)

    load_env_file(env_file)

    assert os.environ["KEEP"] == "existing"
    assert os.environ["NEW_VALUE"] == "ok"


@pytest.mark.asyncio
async def test_stt_voice_loop_leaves_with_fluxer_connection_id(monkeypatch, tmp_path):
    sends = []

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
            sends.append((channel_id, kwargs))
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-123",
                    },
                    {"connection_id": "conn-123", "has_token": True},
                )
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity="bot_conn-123",
            )

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-turns",
            "0",
            "--initial-settle-seconds",
            "0",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await run_stt_voice_loop(args)

    assert result["connection"]["connection_id"] == "conn-123"
    assert sends[0] == ("voice-1", {"guild_id": "guild-1", "self_mute": True, "self_deaf": False})
    assert (None, {"guild_id": "guild-1", "connection_id": "conn-123"}) in sends


@pytest.mark.asyncio
async def test_stt_voice_loop_surfaces_livekit_connect_failure_without_connect_timeout(monkeypatch, tmp_path):
    sends = []

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
            sends.append((channel_id, kwargs))
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-fail",
                    },
                    {"connection_id": "conn-fail", "has_token": True},
                )
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            raise RuntimeError("livekit unavailable")

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-turns",
            "0",
            "--connect-timeout",
            "30",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=1.0)

    assert result["error"] == "RuntimeError"
    assert result["message"] == "livekit unavailable"
    assert sends[0] == ("voice-1", {"guild_id": "guild-1", "self_mute": True, "self_deaf": False})
    assert (None, {"guild_id": "guild-1", "connection_id": None}) in sends


@pytest.mark.asyncio
async def test_stt_voice_loop_returns_json_when_voice_server_update_times_out(monkeypatch, tmp_path):
    sends = []

    class FakeAdapter:
        def __init__(self, config):
            pass

        def set_voice_server_update_handler(self, handler):
            pass

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            return True

        async def send_voice_state_update(self, channel_id, **kwargs):
            sends.append((channel_id, kwargs))
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--connect-timeout",
            "0.01",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await run_stt_voice_loop(args)

    assert result["error"] == "TimeoutError"
    assert "voice server update" in result["message"]
    assert sends[0] == ("voice-1", {"guild_id": "guild-1", "self_mute": True, "self_deaf": False})
    assert (None, {"guild_id": "guild-1", "connection_id": None}) in sends


@pytest.mark.asyncio
async def test_stt_voice_loop_returns_json_when_max_runtime_times_out(monkeypatch, tmp_path):
    sends = []
    release_capture = asyncio.Event()

    class FakeAdapter:
        def __init__(self, config):
            self.handler = None
            self.task = None

        def set_voice_server_update_handler(self, handler):
            self.handler = handler

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            return True

        async def send_voice_state_update(self, channel_id, **kwargs):
            sends.append((channel_id, kwargs))
            if channel_id is not None:
                assert self.handler is not None
                self.task = asyncio.create_task(
                    self.handler(
                        {
                            "endpoint": "wss://voice.example",
                            "token": "secret-token",
                            "guild_id": kwargs.get("guild_id"),
                            "channel_id": channel_id,
                            "connection_id": "conn-slow",
                        },
                        {"connection_id": "conn-slow", "has_token": True},
                    )
                )
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity="bot_conn-slow",
            )

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))
            release_capture.set()

    async def slow_capture(*args, **kwargs):
        await release_capture.wait()
        raise TimeoutError

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._capture_one_speech_segment", slow_capture)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-turns",
            "1",
            "--initial-settle-seconds",
            "0",
            "--max-runtime-seconds",
            "0.01",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await run_stt_voice_loop(args)
    await asyncio.sleep(0)

    assert result["error"] == "TimeoutError"
    assert "max runtime" in result["message"]
    assert result["connection"]["connection_id"] == "conn-slow"
    assert (None, {"guild_id": "guild-1", "connection_id": "conn-slow"}) in sends


@pytest.mark.asyncio
async def test_stt_voice_loop_fails_before_voice_state_when_gateway_ready_times_out(monkeypatch, tmp_path):
    sends = []

    class FakeAdapter:
        def __init__(self, config):
            pass

        def set_voice_server_update_handler(self, handler):
            pass

        async def connect(self):
            return True

        async def wait_until_gateway_ready(self, timeout):
            sends.append(("ready", timeout))
            return False

        async def send_voice_state_update(self, channel_id, **kwargs):
            sends.append((channel_id, kwargs))
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--connect-timeout",
            "0.01",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    with pytest.raises(RuntimeError, match="READY"):
        await run_stt_voice_loop(args)

    assert sends == [("ready", 0.01), (None, {"guild_id": "guild-1", "connection_id": None}), ("bridge_disconnect", {}), ("adapter_disconnect", {})]


@pytest.mark.asyncio
async def test_stt_voice_loop_barge_in_watcher_cancels_after_first_audio_when_xai_stalls(monkeypatch, tmp_path):
    sends = []
    first_audio_written = asyncio.Event()

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
            sends.append((channel_id, kwargs))
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-barge",
                    },
                    {"connection_id": "conn-barge", "has_token": True},
                )
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakePublisher:
        def __init__(self):
            self.bytes_published = 0
            self.interrupted = False
            self.last_queue_duration_before_interrupt = 0.25
            self.last_queue_duration_after_clear = 0.0
            self.closed = False

        async def __aenter__(self):
            return self

        async def write_interruptible(self, chunk, should_interrupt):
            if await should_interrupt():
                await self.interrupt()
                return True
            self.bytes_published += len(chunk)
            first_audio_written.set()
            return False

        async def interrupt(self):
            self.interrupted = True
            self.last_queue_duration_after_clear = 0.0

        async def close(self, **kwargs):
            self.closed = True

    publishers = []

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity="bot_conn-barge",
            )

        def pcm16_publisher(self, **kwargs):
            publisher = FakePublisher()
            publishers.append(publisher)
            return publisher

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    class FakeVoiceClient:
        def __init__(self, **kwargs):
            pass

        async def force_message_to_sink(self, prompt, sink, **kwargs):
            await sink(b"\x01\x00" * 240)
            await asyncio.Event().wait()

    async def fake_barge_in(args, bridge, capture):
        await first_audio_written.wait()
        capture.detected_seconds = 0.05
        capture.max_rms = 1234
        capture.voiced_ms = args.barge_in_min_ms
        capture.chunks_seen = 2
        capture.captured_audio_seconds = 0.2
        capture.event.set()
        capture.ready.set()
        await asyncio.Event().wait()

    async def fake_capture(*args, **kwargs):
        return b"\x01\x00" * 24_000

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.XAIRealtimeVoiceClient", FakeVoiceClient)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._wait_for_barge_in", fake_barge_in)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._capture_one_speech_segment", fake_capture)
    monkeypatch.setattr(
        "scripts.fluxer_stt_voice_loop.transcribe_with_provider",
        lambda *args, **kwargs: {"success": True, "transcript": "give me a long answer", "provider": "fake", "model": "fake"},
    )
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.hermes_chat_completion", lambda *args, **kwargs: asyncio.sleep(0, result="long reply"))

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-turns",
            "1",
            "--initial-settle-seconds",
            "0",
            "--brain-provider",
            "hermes",
            "--barge-in-after-first-audio-only",
            "--turn-log-jsonl",
            str(tmp_path / "turns.jsonl"),
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=2.0)

    assert result["turns"][0]["interrupted"] is True
    assert result["turns"][0]["published"] is False
    assert result["turns"][0]["publisher_queue_after_clear_seconds"] == 0.0
    assert publishers[0].interrupted is True


@pytest.mark.asyncio
async def test_stt_voice_loop_voice_server_handler_returns_before_session_finishes(monkeypatch, tmp_path):
    sends = []
    handler_returned = asyncio.Event()
    capture_started = asyncio.Event()

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
            sends.append((channel_id, kwargs))
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-nonblocking",
                    },
                    {"connection_id": "conn-nonblocking", "has_token": True},
                )
                handler_returned.set()
            return True

        async def disconnect(self):
            sends.append(("adapter_disconnect", {}))

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity="bot_conn-nonblocking",
            )

        async def disconnect(self):
            sends.append(("bridge_disconnect", {}))

    async def blocking_capture(*args, **kwargs):
        capture_started.set()
        await asyncio.Event().wait()

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._capture_one_speech_segment", blocking_capture)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-turns",
            "1",
            "--initial-settle-seconds",
            "0",
            "--max-runtime-seconds",
            "0.01",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=1.0)

    assert handler_returned.is_set()
    assert capture_started.is_set()
    assert result["error"] == "TimeoutError"
    assert "max runtime" in result["message"]
    assert (None, {"guild_id": "guild-1", "connection_id": "conn-nonblocking"}) in sends


@pytest.mark.asyncio
async def test_stt_voice_loop_voice_server_handler_returns_before_livekit_connect_finishes(monkeypatch, tmp_path):
    handler_returned = asyncio.Event()
    connect_started = asyncio.Event()
    release_connect = asyncio.Event()

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
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-slow-connect",
                    },
                    {"connection_id": "conn-slow-connect", "has_token": True},
                )
                handler_returned.set()
            return True

        async def disconnect(self):
            pass

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            connect_started.set()
            await release_connect.wait()
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity="bot_conn-slow-connect",
            )

        async def disconnect(self):
            pass

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--connect-timeout",
            "0.01",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=1.0)

    assert handler_returned.is_set()
    assert connect_started.is_set()
    assert result["error"] == "TimeoutError"
    assert "voice server update" in result["message"]


@pytest.mark.asyncio
async def test_stt_voice_loop_awaits_cancelled_voice_server_connect_before_reconnect(monkeypatch, tmp_path):
    first_connect_started = asyncio.Event()
    first_connect_cancelled = asyncio.Event()
    second_update_sent = asyncio.Event()
    max_concurrent_connects = 0
    active_connects = 0

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
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token-1",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-1",
                    },
                    {"connection_id": "conn-1", "has_token": True},
                )

                async def send_second():
                    await first_connect_started.wait()
                    assert self.handler is not None
                    await self.handler(
                        {
                            "endpoint": "wss://voice.example",
                            "token": "secret-token-2",
                            "guild_id": kwargs.get("guild_id"),
                            "channel_id": channel_id,
                            "connection_id": "conn-2",
                        },
                        {"connection_id": "conn-2", "has_token": True},
                    )
                    second_update_sent.set()

                asyncio.create_task(send_second())
            return True

        async def disconnect(self):
            pass

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            nonlocal active_connects, max_concurrent_connects
            active_connects += 1
            max_concurrent_connects = max(max_concurrent_connects, active_connects)
            try:
                if raw_update["connection_id"] == "conn-1":
                    first_connect_started.set()
                    await asyncio.Event().wait()
                return SimpleNamespace(
                    endpoint=raw_update["endpoint"],
                    guild_id=raw_update["guild_id"],
                    channel_id=raw_update["channel_id"],
                    connection_id=raw_update["connection_id"],
                    room_name="room",
                    participant_identity=f"bot_{raw_update['connection_id']}",
                )
            except asyncio.CancelledError:
                if raw_update["connection_id"] == "conn-1":
                    first_connect_cancelled.set()
                raise
            finally:
                active_connects -= 1

        async def disconnect(self):
            pass

    async def blocking_capture(*args, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._capture_one_speech_segment", blocking_capture)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-runtime-seconds",
            "0.05",
            "--initial-settle-seconds",
            "0",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=1.0)

    assert await asyncio.wait_for(second_update_sent.wait(), timeout=0.1) is True
    assert first_connect_cancelled.is_set()
    assert max_concurrent_connects == 1
    assert result["connection"]["connection_id"] == "conn-2"
    assert result["error"] == "TimeoutError"


def test_stt_voice_loop_serializes_voice_server_update_connects():
    source = inspect.getsource(run_stt_voice_loop)

    assert "voice_update_lock = asyncio.Lock()" in source
    assert "_acquire_lock_with_timeout(voice_update_lock, timeout=args.connect_timeout)" in source
    assert "voice_update_lock.release()" in source


@pytest.mark.asyncio
async def test_stt_voice_loop_cancels_active_session_before_second_voice_server_update(monkeypatch, tmp_path):
    first_capture_started = asyncio.Event()
    first_capture_cancelled = asyncio.Event()
    second_update_sent = asyncio.Event()
    connect_ids = []

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
            if channel_id is not None:
                assert self.handler is not None
                await self.handler(
                    {
                        "endpoint": "wss://voice.example",
                        "token": "secret-token-1",
                        "guild_id": kwargs.get("guild_id"),
                        "channel_id": channel_id,
                        "connection_id": "conn-1",
                    },
                    {"connection_id": "conn-1", "has_token": True},
                )

                async def send_second():
                    await first_capture_started.wait()
                    assert self.handler is not None
                    await self.handler(
                        {
                            "endpoint": "wss://voice.example",
                            "token": "secret-token-2",
                            "guild_id": kwargs.get("guild_id"),
                            "channel_id": channel_id,
                            "connection_id": "conn-2",
                        },
                        {"connection_id": "conn-2", "has_token": True},
                    )
                    second_update_sent.set()

                asyncio.create_task(send_second())
            return True

        async def disconnect(self):
            pass

    class FakeBridge:
        def __init__(self, auto_subscribe=True):
            pass

        async def connect_from_voice_server_update(self, raw_update):
            connect_ids.append(raw_update["connection_id"])
            return SimpleNamespace(
                endpoint=raw_update["endpoint"],
                guild_id=raw_update["guild_id"],
                channel_id=raw_update["channel_id"],
                connection_id=raw_update["connection_id"],
                room_name="room",
                participant_identity=f"bot_{raw_update['connection_id']}",
            )

        async def disconnect(self):
            pass

    async def blocking_capture(*args, **kwargs):
        if not first_capture_started.is_set():
            first_capture_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                first_capture_cancelled.set()
                raise
        await asyncio.Event().wait()

    monkeypatch.setenv("FLUXER_BOT_TOKEN", "token")
    monkeypatch.setenv("XAI_API_KEY", "xai")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerAdapter", FakeAdapter)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop.FluxerLiveKitSmokeBridge", FakeBridge)
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._capture_one_speech_segment", blocking_capture)

    args = parse_args(
        [
            "--channel-id",
            "voice-1",
            "--guild-id",
            "guild-1",
            "--max-runtime-seconds",
            "0.05",
            "--initial-settle-seconds",
            "0",
            "--voice-context-file",
            str(tmp_path / "missing-context.md"),
        ]
    )

    result = await asyncio.wait_for(run_stt_voice_loop(args), timeout=1.0)

    assert await asyncio.wait_for(second_update_sent.wait(), timeout=0.1) is True
    assert first_capture_cancelled.is_set()
    assert connect_ids[:2] == ["conn-1", "conn-2"]
    assert result["connection"]["connection_id"] == "conn-2"
    assert result["error"] == "TimeoutError"


def test_stt_voice_loop_enters_publisher_before_starting_barge_in_tasks():
    source = inspect.getsource(run_stt_voice_loop)

    publisher_enter = source.index("await publisher.__aenter__()")
    first_barge_in_task = source.index("barge_in_task = asyncio.create_task")
    first_interrupt_watcher = source.index("interrupt_watcher_task = asyncio.create_task")

    assert publisher_enter < first_barge_in_task
    assert publisher_enter < first_interrupt_watcher


def test_stt_voice_loop_cancels_xai_task_before_publisher_close():
    source = inspect.getsource(run_stt_voice_loop)

    cancel_xai = source.index("if xai_task is not None and not xai_task.done():")
    close_publisher = source.index("await publisher.close", cancel_xai)

    assert cancel_xai < close_publisher


def test_stt_voice_loop_barge_in_path_cancels_xai_before_closing_publisher():
    source = inspect.getsource(run_stt_voice_loop)

    barge_in_handler = source.index("except BargeInInterrupt:")
    first_barge_close = source.index("await publisher.close", barge_in_handler)
    final_cleanup = source.index("finally:", barge_in_handler)
    cancel_xai = source.index("_cancel_task_safely(xai_task", barge_in_handler)

    assert cancel_xai < first_barge_close
    assert cancel_xai < final_cleanup


def test_stt_voice_loop_redacts_livekit_token_from_join_errors():
    exc = RuntimeError('connect failed Bearer abc.def token=secret123 {"token":"json-secret"}')

    message = _redact_exception_message(exc, "abc.def", "secret123", "json-secret")

    assert "abc.def" not in message
    assert "secret123" not in message
    assert "json-secret" not in message
    assert "[redacted-token]" in message


def test_stt_voice_loop_redacts_session_level_error_result():
    source = inspect.getsource(run_stt_voice_loop)

    assert 'result["message"] = str(exc)' not in source
    assert 'result["message"] = _redact_exception_message(exc)' in source


def test_stt_voice_loop_interrupts_publisher_during_cancellation_cleanup():
    source = inspect.getsource(run_stt_voice_loop)

    cancellation_handler = source.index("except asyncio.CancelledError:\n                    logger.info(\"Cancelling STT-backed voice turn")
    cancellation_block = source[cancellation_handler : source.index("except BargeInInterrupt:", cancellation_handler)]

    assert "await publisher.interrupt()" in cancellation_block
    assert 'result["turns"].append(turn)' in cancellation_block
    assert "raise" in cancellation_block
    assert "await publisher.close(wait_for_playout=False, flush_remainder=False)" not in cancellation_block
    assert "shutdown_requested.set()\n        finished.set()" not in source
