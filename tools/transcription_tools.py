"""Standalone speech-to-text helpers used by the Fluxer voice plugin.

The plugin normally runs inside Hermes, where ``tools.transcription_tools`` is
available from the main agent tree.  The public plugin repository and CI run on
their own, so this module provides the small compatible surface the Fluxer voice
loop needs without depending on a checkout of Hermes Agent.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

ELEVENLABS_STT_BASE_URL = os.getenv("ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
XAI_STT_BASE_URL = os.getenv("XAI_STT_BASE_URL", "https://api.x.ai/v1")


def get_env_value(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def is_truthy_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _load_stt_config() -> dict[str, Any]:
    return {}


def _extract_transcript_text(result: Any) -> str:
    if isinstance(result, dict):
        for key in ("text", "transcript"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        words = result.get("words")
        if isinstance(words, list):
            joined = " ".join(str(item.get("text") or item.get("word") or "").strip() for item in words if isinstance(item, dict))
            if joined.strip():
                return joined.strip()
        segments = result.get("segments")
        if isinstance(segments, list):
            joined = " ".join(str(item.get("text") or "").strip() for item in segments if isinstance(item, dict))
            if joined.strip():
                return joined.strip()
    return ""


def _multipart_form(fields: dict[str, str], file_field: str, file_path: str) -> tuple[bytes, str]:
    boundary = "----fluxer-stt-boundary"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    path = Path(file_path)
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{path.name}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    chunks.append(path.read_bytes())
    chunks.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _post_json_multipart(url: str, *, headers: dict[str, str], fields: dict[str, str], file_field: str, file_path: str) -> dict[str, Any]:
    body, boundary = _multipart_form(fields, file_field, file_path)
    req = urllib_request.Request(
        url,
        data=body,
        headers={**headers, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _transcribe_elevenlabs(file_path: str, model: str = "scribe_v2") -> dict[str, Any]:
    api_key = get_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "ELEVENLABS_API_KEY not set"}
    try:
        result = _post_json_multipart(
            f"{ELEVENLABS_STT_BASE_URL.rstrip('/')}/speech-to-text",
            headers={"xi-api-key": api_key},
            fields={"model_id": model, "tag_audio_events": "false", "diarize": "false"},
            file_field="file",
            file_path=file_path,
        )
        transcript = _extract_transcript_text(result)
        return {"success": bool(transcript), "transcript": transcript, "provider": "elevenlabs", "model": model}
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        return {"success": False, "transcript": "", "error": f"ElevenLabs STT transcription failed: {exc}"}


def _transcribe_groq(file_path: str, model: str = "whisper-large-v3-turbo") -> dict[str, Any]:
    api_key = get_env_value("GROQ_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "GROQ_API_KEY not set"}
    try:
        result = _post_json_multipart(
            f"{GROQ_BASE_URL.rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            fields={"model": model},
            file_field="file",
            file_path=file_path,
        )
        transcript = _extract_transcript_text(result)
        return {"success": bool(transcript), "transcript": transcript, "provider": "groq", "model": model}
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        return {"success": False, "transcript": "", "error": f"Groq STT transcription failed: {exc}"}


def _transcribe_xai(file_path: str, model: str = "grok-stt") -> dict[str, Any]:
    api_key = get_env_value("XAI_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "XAI_API_KEY not set"}
    try:
        result = _post_json_multipart(
            f"{XAI_STT_BASE_URL.rstrip('/')}/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            fields={"model": model},
            file_field="file",
            file_path=file_path,
        )
        transcript = _extract_transcript_text(result)
        return {"success": bool(transcript), "transcript": transcript, "provider": "xai", "model": model}
    except (HTTPError, URLError, OSError, json.JSONDecodeError) as exc:
        return {"success": False, "transcript": "", "error": f"xAI STT transcription failed: {exc}"}


def transcribe_audio(file_path: str, model: str | None = None) -> dict[str, Any]:
    """Best-effort local transcription fallback.

    If ``HERMES_LOCAL_STT_COMMAND`` is configured, it may contain
    ``{input_path}``, ``{model}``, and ``{output_dir}`` placeholders and should
    emit transcript text to stdout or a txt file in the output dir. Without that
    command, standalone CI/runtime reports a clear unavailable error instead of
    failing import-time collection.
    """

    command_template = os.getenv("HERMES_LOCAL_STT_COMMAND", "").strip()
    if not command_template:
        return {"success": False, "transcript": "", "provider": "local", "error": "local STT command not configured"}
    with tempfile.TemporaryDirectory() as tmp:
        command = command_template.format(
            input_path=shlex.quote(str(Path(file_path))),
            model=shlex.quote(model or "base"),
            output_dir=shlex.quote(tmp),
        )
        proc = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=300)
        if proc.returncode != 0:
            return {"success": False, "transcript": "", "provider": "local", "error": proc.stderr.strip() or proc.stdout.strip()}
        transcript = proc.stdout.strip()
        if not transcript:
            txt_files = sorted(Path(tmp).glob("*.txt"))
            if txt_files:
                transcript = txt_files[0].read_text(errors="replace").strip()
        return {"success": bool(transcript), "transcript": transcript, "provider": "local", "model": model or "base"}
