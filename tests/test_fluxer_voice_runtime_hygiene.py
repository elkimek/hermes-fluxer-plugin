"""Runtime-hygiene regressions for the Fluxer voice scripts.

Covers the private-audio tempfile path, the read-only SQLite URI, the startup
VAD-timing validation and the ElevenLabs language path's dependency surface.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime

import pytest

from scripts import fluxer_stt_voice_loop as stt_loop


def _seed_state_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
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


def test_collect_voice_session_recall_handles_special_chars_in_db_path(tmp_path):
    """A raw ``file:{path}?mode=ro`` URI truncates at the first ``?`` in the path,
    so the handle silently opens (and creates) the wrong database. The recall
    lookup must percent-encode the path the way ``Path.as_uri()`` does."""

    db = tmp_path / "home?profile" / "state.db"
    _seed_state_db(db)

    recall = stt_loop.collect_voice_session_recall(
        "What did we do last Monday?",
        db_path=str(db),
        now=datetime(2026, 6, 6, 12, 0, 0),
    )

    assert "We built the hybrid voice router" in recall
    assert "Session recall lookup failed" not in recall
    # The truncated sibling must never have been created by the read.
    assert not (tmp_path / "home").exists()


def test_read_only_uri_percent_encodes_reserved_characters(tmp_path):
    path = tmp_path / "weird?name#frag" / "state.db"
    uri = stt_loop._read_only_sqlite_uri(path)

    assert uri.startswith("file://")
    assert uri.endswith("?mode=ro")
    assert "%3F" in uri  # the '?' survives as an escape, not as the query separator


def test_voice_loop_writes_private_audio_to_secure_tempfile(tmp_path):
    """Raw speech must not land in a predictable world-readable /tmp path."""

    with stt_loop.secure_temp_audio_path() as path:
        assert path.exists()
        assert (path.stat().st_mode & 0o777) == 0o600
        assert "fluxer_stt_loop_input" not in path.name
        path.write_bytes(b"private speech")
        held = path

    assert not held.exists()


def test_voice_loop_source_has_no_predictable_tmp_audio_paths():
    from pathlib import Path

    source = Path(stt_loop.__file__).read_text(encoding="utf-8")

    assert "fluxer_stt_loop_input_" not in source
    assert "fluxer_stt_loop_barge_stop_" not in source


def test_validate_vad_timing_accepts_valid_pair_and_rejects_invalid():
    valid = stt_loop.parse_args(["--channel-id", "v", "--silence-ms", "1500", "--end-padding-ms", "300"])
    stt_loop._validate_vad_timing(valid)  # must not raise

    invalid = stt_loop.parse_args(["--channel-id", "v", "--silence-ms", "100", "--end-padding-ms", "500"])
    with pytest.raises(ValueError, match="end_padding_ms"):
        stt_loop._validate_vad_timing(invalid)


@pytest.mark.asyncio
async def test_run_loop_rejects_invalid_vad_timing_before_any_connect():
    args = stt_loop.parse_args(["--channel-id", "v", "--silence-ms", "100", "--end-padding-ms", "500"])

    with pytest.raises(ValueError, match="end_padding_ms"):
        await stt_loop.run_stt_voice_loop(args)


def test_elevenlabs_language_path_uses_stdlib_multipart(monkeypatch, tmp_path):
    seen: dict = {}

    def fake_post(url, *, headers, fields, file_field, file_path):
        seen.update(url=url, headers=headers, fields=fields, file_field=file_field, file_path=file_path)
        return {"text": "ok"}

    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr("scripts.fluxer_stt_voice_loop._post_json_multipart", fake_post)
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"RIFF")

    result = stt_loop.transcribe_elevenlabs_with_language(str(audio), "scribe_v2", "eng")

    assert result["success"] is True
    assert result["transcript"] == "ok"
    assert seen["fields"]["language_code"] == "eng"
    assert seen["headers"]["xi-api-key"] == "k"
    assert seen["file_field"] == "file"
    assert seen["url"].endswith("/speech-to-text")


def test_voice_loop_does_not_import_undeclared_requests():
    from pathlib import Path

    source = Path(stt_loop.__file__).read_text(encoding="utf-8")

    assert "import requests" not in source
    assert "requests." not in source


def test_no_operator_secret_is_read_and_env_is_isolated():
    """The suite must never consult the operator's real deployment.

    A shell that sourced `~/.hermes/.env` used to re-point the gate tests at the
    operator's guild/channel ids (four phantom failures). The conftest clears the
    whole leak surface, so nothing here sees the ambient `FLUXER_*` values, and no
    real credential is required to run the suite.
    """

    import conftest

    assert conftest._ENV_LEAK_PREFIXES == ("FLUXER_",)
    assert not any(key.startswith("FLUXER_") for key in os.environ)
    assert os.environ["HERMES_ENV_FILE"] == str(conftest.ISOLATED_ENV_FILE)
    assert not conftest.ISOLATED_ENV_FILE.exists()  # nothing to load, no secrets


def test_voice_scripts_default_env_file_resolves_to_the_isolated_path():
    """`--env-file` defaults to `HERMES_ENV_FILE`, so a test that omits the flag
    must resolve the throwaway file rather than the operator's real `.env`."""

    import conftest

    args = stt_loop.parse_args(["--channel-id", "voice-1"])
    assert args.env_file == str(conftest.ISOLATED_ENV_FILE)


def test_load_env_file_only_reads_the_isolated_file(tmp_path, monkeypatch):
    """`load_env_file` mutates `os.environ`, so it must be pointed at a synthetic
    file, never the operator's real one."""

    env_file = tmp_path / "synthetic.env"
    env_file.write_text("FLUXER_SYNTHETIC_ONLY=1\n", encoding="utf-8")

    stt_loop.load_env_file(env_file)

    assert os.environ.get("FLUXER_SYNTHETIC_ONLY") == "1"
    assert "FLUXER_BOT_TOKEN" not in os.environ


def test_secure_temp_audio_path_cleans_up_when_the_body_raises():
    held = None
    with pytest.raises(RuntimeError):
        with stt_loop.secure_temp_audio_path() as path:
            held = path
            raise RuntimeError("boom")
    assert held is not None and not held.exists()


def test_room_loop_gate_audio_uses_a_secure_tempfile():
    from pathlib import Path

    from scripts import fluxer_xai_room_loop as room_loop

    source = Path(room_loop.__file__).read_text(encoding="utf-8")

    assert "fluxer_xai_room_loop_gate_" not in source
    assert "secure_temp_audio_path" in source


def test_scripts_never_shadow_the_host_tools_package():
    """Every voice script that needs the Hermes root must put it on `sys.path`
    without ever shipping a plugin-local `tools` package."""

    from pathlib import Path

    root = Path(stt_loop.__file__).resolve().parents[1]
    assert not (root / "tools").exists()
    assert (root / "fluxer_plugin_tools").is_dir()

    for script in sorted((root / "scripts").glob("*.py")):
        text = script.read_text(encoding="utf-8")
        assert "from tools." not in text, f"{script.name} imports host tools by a shadowable name"
        assert "import tools." not in text, f"{script.name} imports host tools by a shadowable name"

