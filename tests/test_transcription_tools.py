from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "transcription_tools.py"
SPEC = importlib.util.spec_from_file_location("fluxer_standalone_transcription_tools", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
transcription_tools = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transcription_tools)


def test_elevenlabs_helper_reuses_shared_endpoint(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_STT_BASE_URL", "https://shared.example.test/v1")
    common_path = MODULE_PATH.with_name("transcription_common.py")
    common_spec = importlib.util.spec_from_file_location("tools.transcription_common", common_path)
    assert common_spec is not None and common_spec.loader is not None
    common = importlib.util.module_from_spec(common_spec)
    common_spec.loader.exec_module(common)
    monkeypatch.setitem(sys.modules, "tools.transcription_common", common)

    # A second getenv/default definition would pick a different endpoint.
    monkeypatch.setenv("ELEVENLABS_STT_BASE_URL", "https://independent.example.test/v1")
    assert SPEC is not None and SPEC.loader is not None
    helper = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(helper)

    assert helper.ELEVENLABS_STT_BASE_URL == common.ELEVENLABS_STT_BASE_URL


def test_local_stt_command_quotes_model_placeholder(monkeypatch, tmp_path):
    marker = tmp_path / "injected"
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setenv("HERMES_LOCAL_STT_COMMAND", "printf transcript-{model}")

    result = transcription_tools.transcribe_audio(str(audio_path), model=f"base; touch {marker}")

    assert result["success"] is True
    assert "base; touch" in result["transcript"]
    assert not marker.exists()


def test_local_stt_command_quotes_input_and_output_placeholders(monkeypatch, tmp_path):
    audio_path = tmp_path / "audio with spaces.wav"
    audio_path.write_bytes(b"RIFF")
    monkeypatch.setenv(
        "HERMES_LOCAL_STT_COMMAND",
        "python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).name)' {input_path}",
    )

    result = transcription_tools.transcribe_audio(str(audio_path), model="base")

    assert result["success"] is True
    assert result["transcript"] == "audio with spaces.wav"
