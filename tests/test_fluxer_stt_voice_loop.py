from __future__ import annotations

import os
import wave

from scripts.fluxer_stt_voice_loop import (
    append_jsonl,
    build_answer_prompt,
    build_hermes_messages,
    load_env_file,
    normalize_voice_transcript,
    parse_args,
    safe_stt_summary,
    transcribe_with_provider,
    write_pcm16_wav,
)


def test_build_answer_prompt_grounds_latest_transcript_and_history():
    prompt = build_answer_prompt(
        "Shevka, what is two past two?",
        history=[{"user": "hello", "assistant": "Hi."}],
    )

    assert "Latest STT transcript from Elkim: 'Shevka, what is two past two?'" in prompt
    assert "Elkim: hello" in prompt
    assert "Žofka: Hi." in prompt
    assert "past" in prompt and "plus" in prompt
    assert "plast" in prompt and "plus" in prompt
    assert "participant-targeted capture" in prompt
    assert "ElevenLabs Scribe STT" in prompt
    assert "xAI Eve TTS" in prompt
    assert "Fluxer implementation" in prompt
    assert "Speak English by default" in prompt
    assert "Do not switch to Czech" in prompt
    assert "No filler greetings" in prompt


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


def test_parse_args_defaults_to_realtime_voice_stack():
    args = parse_args(["--channel-id", "voice-room"])

    assert args.stt_provider == "elevenlabs"
    assert args.stt_model == "medium.en"
    assert args.capture_mode == "vad"
    assert args.capture_window_seconds == 3.0
    assert args.brain_provider == "hermes"
    assert args.hermes_url == "http://127.0.0.1:8642"
    assert args.silence_ms == 500


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
