#!/usr/bin/env python3
"""STT-backed Fluxer voice room loop.

This is the "actually listen" prototype:
1. join a Fluxer voice channel,
2. capture audio only from a targeted LiveKit participant/user prefix,
3. transcribe that PCM with Hermes STT,
4. ask xAI Realtime to speak a text-grounded answer,
5. publish the resulting WAV back into Fluxer LiveKit.

It intentionally avoids xAI's direct audio-understanding path because live tests
showed it produced generic filler ("hey, ready to chat?") even when transport
was working.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import tempfile
import time
import urllib.request
import wave
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERMES_ROOT = Path(os.getenv("HERMES_AGENT_ROOT", str(Path.home() / ".hermes" / "hermes-agent")))
for candidate in (ROOT, HERMES_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from adapter import FluxerAdapter  # noqa: E402
from gateway.config import PlatformConfig  # noqa: E402
from livekit_bridge import FluxerLiveKitSmokeBridge  # noqa: E402
from scripts.fluxer_xai_room_loop import (  # noqa: E402
    BargeInCapture,
    BargeInInterrupt,
    _acquire_lock_with_timeout,
    _capture_one_speech_segment,
    _cancel_previous_task_safely,
    _cancel_task_safely,
    _redact_exception_message,
    _wait_for_barge_in,
)
from tools.transcription_tools import (  # noqa: E402
    ELEVENLABS_STT_BASE_URL,
    _extract_transcript_text,
    _load_stt_config,
    _transcribe_elevenlabs,
    _transcribe_groq,
    _transcribe_xai,
    get_env_value,
    is_truthy_value,
    transcribe_audio,
)
from xai_realtime import XAIRealtimeVoiceClient  # noqa: E402

logger = logging.getLogger("fluxer_stt_voice_loop")

DEFAULT_TEXT_SYSTEM = """You are the configured Hermes assistant in a live Fluxer voice chat.
Answer naturally for realtime speech: brief for operational turns, but not evasive when the user asks for substance. Use the latest transcript and recent voice-chat history. Do not claim platform history, private memory, files, or previous-session details unless they were provided to this turn as explicit recall context.

Current implementation context you know:
- The room path is Fluxer LiveKit capture → STT → text-grounded answer → realtime TTS → LiveKit publish.
- In room mode, participant-targeted capture can make a wake name unnecessary; if capture is explicitly targeted, speech from the configured participant counts as addressed to the assistant.
- If asked about a specific past day, date, previous session, transcript, or memory and this is not full Hermes brain with retrieved evidence, do not guess from cached context. Say you need full brain/session recall for that.

Conversation rules:
- Answer the transcript directly. Be brief for operational chatter; be meaningfully longer when the user asks for depth, introspection, personal observations, or self-reflection.
- Do not end with generic follow-up questions unless a clarification is genuinely needed.
- If the user asks to stop or leave, acknowledge briefly and stop; do not ask another question.
- For pleasant small talk, respond warmly and stop.
- No filler greetings unless the user greeted you.
- Correct obvious ASR homophones when context is clear, e.g. "past", "plast", or "plastic" can mean "plus" in arithmetic.
- Default to English unless the user explicitly asks for another language or clearly speaks another language.
""".strip()


def env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def pcm16_rms(pcm: bytes) -> int:
    if not pcm:
        return 0
    width = 2
    sample_count = len(pcm) // width
    if sample_count <= 0:
        return 0
    total = 0
    for i in range(0, sample_count * width, width):
        sample = int.from_bytes(pcm[i : i + width], byteorder="little", signed=True)
        total += sample * sample
    return int((total / sample_count) ** 0.5)


def load_env_file(path: Path) -> None:
    """Load simple KEY=value lines without shell-sourcing secrets."""

    if not path.exists():
        return
    for line in path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key and not os.environ.get(key):
            os.environ[key] = value.strip().strip('"').strip("'")


def write_pcm16_wav(path: Path, pcm: bytes, *, sample_rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return path


_MEMORY_CONTEXT_RE = re.compile(r"<memory-context>.*?</memory-context>", re.IGNORECASE | re.DOTALL)
_SYSTEM_NOTE_RE = re.compile(r"\[System note:.*?\]", re.IGNORECASE | re.DOTALL)
_NON_ENGLISH_DIACRITIC_RE = re.compile(r"[ÁČĎÉĚÍŇÓŘŠŤÚŮÝŽáčďéěíňóřšťúůýžÃÁÀÂÉÊÍÓÔÕÚÇãáàâéêíóôõúç]")


def looks_like_clipped_non_english_noise(transcript: str) -> bool:
    """Reject tiny VAD fragments that STT hallucinates as non-English speech."""

    text = " ".join((transcript or "").split()).strip()
    if not text:
        return False
    words = text.split()
    if len(words) <= 6 and _NON_ENGLISH_DIACRITIC_RE.search(text):
        return True
    lower = text.lower().strip(" .!?…")
    return lower in {"e aí", "je to", "je to pračka", "ahoj", "dobře"}


def is_voice_stop_request(transcript: str) -> bool:
    """Return true when the user clearly asks the live voice loop to stop."""

    lower = " ".join((transcript or "").lower().split()).strip(" .!?…")
    if not lower:
        return False
    stop_phrases = (
        "we can stop here",
        "let's stop here",
        "stop here",
        "stop the voice chat",
        "stop voice chat",
        "end the voice chat",
        "end voice chat",
        "that's enough",
        "we are done",
        "i'm done talking",
    )
    return any(phrase in lower for phrase in stop_phrases)


def requested_brain_mode_switch(transcript: str) -> str | None:
    """Detect spoken requests to stick future turns to fast or full Hermes brain."""

    lower = " ".join((transcript or "").lower().split()).strip(" .!?…")
    if not lower:
        return None
    fast_phrases = (
        "switch back to fast",
        "go back to fast",
        "go back to the fast",
        "use fast mode",
        "use the fast mode",
        "use the fast brain",
        "switch to fast brain",
        "switch to the fast brain",
        "back to xai fast",
        "back to xai fest",
        "back to fast mode",
        "back to the fast mode",
        "switch back to xai fast",
        "switch back to xai fest",
        "casual mode",
    )
    hermes_phrases = (
        "switch to full brain",
        "switch to full hermes",
        "switch to hermes",
        "use full brain",
        "use the full brain",
        "use hermes mode",
        "use full hermes",
        "full hermes mode",
        "full brain hermes mode",
        "turn on full brain",
        "deep mode",
    )
    if any(phrase in lower for phrase in fast_phrases):
        return "xai-fast"
    if any(phrase in lower for phrase in hermes_phrases):
        return "hermes"
    return None


def transcript_needs_full_brain(transcript: str) -> bool:
    """Heuristic for one-turn escalation from fast voice brain to Hermes brain."""

    lower = " ".join((transcript or "").lower().split()).strip(" .!?…")
    if not lower:
        return False
    if re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower):
        return True
    temporal_memory_terms = (
        "last monday",
        "yesterday",
        "last week",
        "earlier today",
        "what were we doing",
        "what did we do",
        "what did we talk about",
        "where did we leave",
        "remember when",
        "do you remember",
        "session history",
        "conversation history",
        "past conversation",
        "previous conversation",
        "look at the transcript",
        "check the transcript",
        "check memory",
        "search memory",
        "use honcho",
        "use gbrain",
        "use tools",
        "full brain",
        "full hermes",
        "hermes mode",
    )
    return any(term in lower for term in temporal_memory_terms)


def resolve_voice_brain_provider(configured_provider: str, sticky_provider: str, transcript: str) -> tuple[str, str, str]:
    """Return provider for this turn, new sticky provider, and routing reason."""

    requested = requested_brain_mode_switch(transcript)
    if configured_provider != "auto":
        if requested and requested != configured_provider:
            return configured_provider, configured_provider, "configured_provider_ignores_voice_switch"
        return configured_provider, configured_provider, "configured_provider"
    if requested:
        return requested, requested, f"voice_switch_{requested}"
    if sticky_provider == "hermes":
        return "hermes", sticky_provider, "sticky_hermes"
    if transcript_needs_full_brain(transcript):
        return "hermes", sticky_provider, "auto_escalate_memory_context"
    return "xai-fast", sticky_provider, "auto_fast"


def voice_mode_ack(mode: str) -> str:
    if mode == "hermes":
        return "Switched to full Hermes brain. I’ll be slower, but deeper."
    return "Back to fast voice mode."


def normalize_voice_transcript(transcript: str) -> str:
    """Drop non-spoken context wrappers before routing STT text to Hermes.

    The live API may append recalled memory/context blocks beside a voice
    transcript. Those are useful to the agent, but they are not something the
    user said out loud; feeding them back as user speech makes the assistant answer the
    wrapper instead of the spoken turn.
    """

    cleaned = _MEMORY_CONTEXT_RE.sub(" ", transcript or "")
    cleaned = _SYSTEM_NOTE_RE.sub(" ", cleaned)
    return " ".join(cleaned.split()).strip()


def load_voice_context_cache(path: str | None) -> str:
    if not path:
        return ""
    target = Path(path).expanduser()
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8", errors="ignore").strip()


def compose_system_prompt(base: str = DEFAULT_TEXT_SYSTEM, *, voice_context_cache: str = "") -> str:
    voice_context_cache = voice_context_cache.strip()
    if not voice_context_cache:
        return base
    return (
        f"{base}\n\n"
        "Cached deployment-local context loaded once at voice-loop startup. "
        "Use it as background, but answer the latest spoken transcript, not the cache itself.\n"
        f"{voice_context_cache}"
    )


def build_answer_prompt(transcript: str, *, history: list[dict[str, str]], system: str = DEFAULT_TEXT_SYSTEM) -> str:
    """Build a compact text prompt for a voice answer grounded in STT text."""

    transcript = normalize_voice_transcript(transcript)
    lower_transcript = " ".join(transcript.lower().split())
    continuation_instruction = ""
    if "count" in lower_transcript or "counting" in lower_transcript:
        continuation_instruction = (
            " If the latest transcript asks you to count or keep counting, count slowly from one upward as a continuous spoken stream; "
            "do not summarize the task and do not say 'let me know when to stop'."
        )
    history_lines: list[str] = []
    for item in history[-6:]:
        user = (item.get("user") or "").strip()
        assistant = (item.get("assistant") or "").strip()
        if user:
            history_lines.append(f"the user: {user}")
        if assistant:
            history_lines.append(f"the assistant: {assistant}")
    history_text = "\n".join(history_lines) or "(none)"
    return (
        f"{system}\n\n"
        f"Recent voice-chat history:\n{history_text}\n\n"
        f"Latest STT transcript from the user: {transcript!r}\n\n"
        "Speak the assistant's next reply now. Use the latest transcript, relevant voice history, and the implementation context above. "
        "For casual or operational turns, keep it short. If the latest transcript asks for deep, personal, introspective, reflective, or self-knowledge content, answer with 2-4 substantive spoken sentences and do not end with a generic follow-up question. "
        "If the latest transcript is a stop/leave request, acknowledge briefly and stop without asking another question."
        f"{continuation_instruction}"
    )


def build_hermes_messages(
    transcript: str,
    *,
    history: list[dict[str, str]],
    system: str = DEFAULT_TEXT_SYSTEM,
    recall_context: str = "",
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}]
    recall_context = recall_context.strip()
    if recall_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Relevant read-only session/memory context retrieved for this voice turn:\n"
                    + recall_context
                    + "\n\nUse these excerpts as evidence. If the recall says no local messages were found, say that directly and do not infer an answer from unrelated cached context."
                ),
            }
        )
    for item in history[-8:]:
        user = (item.get("user") or "").strip()
        assistant = (item.get("assistant") or "").strip()
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": normalize_voice_transcript(transcript)})
    return messages


def _last_weekday_window(weekday: int, now: datetime | None = None) -> tuple[datetime, datetime]:
    now = now or datetime.now().astimezone()
    days_since = (now.weekday() - weekday) % 7
    if days_since == 0:
        days_since = 7
    target = now - timedelta(days=days_since)
    start = target.replace(hour=0, minute=0, second=0, microsecond=0)
    return start, start + timedelta(days=1)


def _last_monday_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    return _last_weekday_window(0, now)


def voice_recall_time_window(transcript: str, *, now: datetime | None = None) -> tuple[datetime, datetime, str] | None:
    lower = " ".join((transcript or "").lower().split())
    now = now or datetime.now().astimezone()
    weekday_match = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", lower)
    if weekday_match:
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        day_name = weekday_match.group(1)
        start, end = _last_weekday_window(weekdays[day_name], now)
        return start, end, f"last {day_name.title()}"
    if "yesterday" in lower:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start, start + timedelta(days=1), "yesterday"
    return None


def collect_voice_session_recall(
    transcript: str,
    *,
    db_path: str,
    max_messages: int = 24,
    now: datetime | None = None,
) -> str:
    """Return compact read-only session DB excerpts for temporal voice questions."""

    window = voice_recall_time_window(transcript, now=now)
    if not window:
        return ""
    path = Path(db_path).expanduser()
    if not path.exists():
        return ""
    start, end, label = window
    start_ts = start.timestamp()
    end_ts = end.timestamp()
    uri = f"file:{path}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True, timeout=2)
        sessions = con.execute(
            """
            select id, coalesce(title, ''), source
            from sessions
            where started_at < ? and coalesce(ended_at, started_at) >= ?
            order by started_at asc
            limit 12
            """,
            (end_ts, start_ts),
        ).fetchall()
        rows: list[tuple[float, str, str, str, str]] = []
        per_session_limit = max(2, max_messages // max(1, len(sessions))) if sessions else max_messages
        for session_id, title, source in sessions:
            rows.extend(
                con.execute(
                    """
                    select timestamp, role, content, ?, ?
                    from messages
                    where session_id = ?
                      and timestamp >= ? and timestamp < ?
                      and role in ('user', 'assistant')
                      and content is not null
                      and length(trim(content)) > 0
                    order by timestamp asc
                    limit ?
                    """,
                    (title, source, session_id, start_ts, end_ts, per_session_limit),
                ).fetchall()
            )
        rows.sort(key=lambda row: row[0])
        rows = rows[:max_messages]
    except sqlite3.Error as exc:
        return f"Session recall lookup failed: {type(exc).__name__}: {exc}"
    finally:
        with contextlib.suppress(Exception):
            con.close()  # type: ignore[possibly-undefined]
    if not rows:
        return f"No local Hermes session messages found for {label} ({start.date()})."
    lines = [f"Local Hermes session excerpts for {label} ({start.date()}):"]
    for ts, role, content, title, source in rows:
        when = datetime.fromtimestamp(float(ts)).strftime("%H:%M")
        snippet = " ".join(str(content).split())[:360]
        title_part = f" [{title}]" if title else ""
        lines.append(f"- {when} {role}{title_part} ({source}): {snippet}")
    return "\n".join(lines)


def build_full_brain_transcript(transcript: str, *, args: argparse.Namespace) -> tuple[str, str]:
    recall = collect_voice_session_recall(
        transcript,
        db_path=getattr(args, "voice_session_db", "~/.hermes/state.db"),
    )
    if not recall:
        return transcript, ""
    return transcript, recall


def _safe_session_fragment(value: str | None, *, fallback: str) -> str:
    """Return a header-safe session-id fragment for Hermes API calls."""

    raw = str(value or "").strip()
    if not raw:
        raw = fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", raw).strip("-._:")
    return cleaned[:80] or fallback


def _bounded_session_id(value: str, *, max_len: int = 64) -> str:
    """Keep session ids short enough for provider prompt-cache keys.

    Hermes accepts longer session headers, but some providers derive a
    ``prompt_cache_key`` from the session id and reject values over 64 chars.
    Preserve readability for short ids; hash only when needed.
    """

    cleaned = _safe_session_fragment(value, fallback="fluxer-voice")
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
    prefix = cleaned[: max_len - len(digest) - 1].rstrip("-._:") or "fluxer-voice"
    return f"{prefix}-{digest}"[:max_len]


def hermes_voice_session_identity(args: argparse.Namespace) -> tuple[str, str]:
    """Stable Hermes session id/key for a Fluxer voice room.

    ``/v1/chat/completions`` is stateless unless callers provide
    ``X-Hermes-Session-Id`` and ``X-Hermes-Session-Key``. Voice mode uses
    stable room-scoped values so full-brain turns become normal persisted
    Hermes sessions and Honcho can scope long-term memory to the same room.
    """

    configured_id = str(getattr(args, "hermes_session_id", "") or "").strip()
    configured_key = str(getattr(args, "hermes_session_key", "") or "").strip()
    guild = _safe_session_fragment(getattr(args, "guild_id", None), fallback="noguild")
    channel = _safe_session_fragment(getattr(args, "channel_id", None), fallback="nochannel")
    participant = _safe_session_fragment(
        getattr(args, "participant_identity_prefix", None) or getattr(args, "participant_identity", None),
        fallback="room",
    )
    default_id = f"fluxer-voice-{guild}-{channel}-{participant}"
    default_key = f"fluxer:voice:guild:{guild}:channel:{channel}:participant:{participant}"
    session_id = _bounded_session_id(configured_id or default_id, max_len=64)
    session_key = configured_key or default_key
    # The API server caps session headers at 256 chars, while provider prompt
    # cache keys may reject session ids over 64 chars.
    return session_id, session_key[:256]


async def hermes_chat_completion(transcript: str, *, history: list[dict[str, str]], args: argparse.Namespace) -> str:
    api_key = os.getenv("API_SERVER_KEY", "").strip()
    if not api_key:
        raise RuntimeError("API_SERVER_KEY is required for Hermes brain mode")
    full_transcript, recall_context = build_full_brain_transcript(transcript, args=args)
    payload = json.dumps(
        {
            "model": args.hermes_model,
            "messages": build_hermes_messages(
                full_transcript,
                history=history,
                system=getattr(args, "voice_system_prompt", DEFAULT_TEXT_SYSTEM),
                recall_context=recall_context,
            ),
            "max_tokens": args.hermes_max_tokens,
            "temperature": args.hermes_temperature,
        }
    ).encode("utf-8")
    session_id, session_key = hermes_voice_session_identity(args)
    req = urllib.request.Request(
        args.hermes_url.rstrip("/") + "/v1/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Hermes-Session-Id": session_id,
            "X-Hermes-Session-Key": session_key,
        },
    )
    def _post_completion() -> dict[str, Any]:
        with urllib.request.urlopen(req, timeout=args.hermes_timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    data = await asyncio.to_thread(_post_completion)
    content = data["choices"][0]["message"]["content"]
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Hermes API returned empty assistant content")
    return content.strip()


def transcribe_elevenlabs_with_language(file_path: str, model_name: str, language_code: str) -> dict[str, Any]:
    api_key = get_env_value("ELEVENLABS_API_KEY")
    if not api_key:
        return {"success": False, "transcript": "", "error": "ELEVENLABS_API_KEY not set"}

    stt_config = _load_stt_config()
    elevenlabs_config = dict(stt_config.get("elevenlabs") or {})
    base_url = str(
        elevenlabs_config.get("base_url")
        or get_env_value("ELEVENLABS_STT_BASE_URL")
        or ELEVENLABS_STT_BASE_URL
    ).strip().rstrip("/")
    tag_audio_events = is_truthy_value(elevenlabs_config.get("tag_audio_events", False))
    diarize = is_truthy_value(elevenlabs_config.get("diarize", False))

    try:
        import requests

        data: dict[str, str] = {
            "model_id": model_name,
            "tag_audio_events": "true" if tag_audio_events else "false",
            "diarize": "true" if diarize else "false",
            "language_code": language_code,
        }
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                f"{base_url}/speech-to-text",
                headers={"xi-api-key": api_key},
                files={"file": (Path(file_path).name, audio_file)},
                data=data,
                timeout=120,
            )
        if response.status_code != 200:
            detail = ""
            try:
                err_body = response.json()
                error_value = err_body.get("detail") or err_body.get("error")
                if isinstance(error_value, dict):
                    detail = str(error_value.get("message") or error_value)
                elif error_value:
                    detail = str(error_value)
                else:
                    detail = response.text[:300]
            except Exception:
                detail = response.text[:300]
            return {
                "success": False,
                "transcript": "",
                "error": f"ElevenLabs STT API error (HTTP {response.status_code}): {detail}",
            }
        result = response.json()
        transcript_text = _extract_transcript_text(result)
        if not transcript_text:
            return {"success": False, "transcript": "", "error": "ElevenLabs STT returned empty transcript"}
        return {"success": True, "transcript": transcript_text, "provider": "elevenlabs"}
    except PermissionError:
        return {"success": False, "transcript": "", "error": f"Permission denied: {file_path}"}
    except Exception as exc:
        logger.error("ElevenLabs STT transcription failed: %s", exc, exc_info=True)
        return {"success": False, "transcript": "", "error": f"ElevenLabs STT transcription failed: {exc}"}


def transcribe_with_provider(
    file_path: str,
    *,
    provider: str,
    model: str | None,
    elevenlabs_language_code: str | None = None,
) -> dict[str, Any]:
    """Transcribe with an explicit provider for this spike loop."""

    if provider == "auto":
        return transcribe_audio(file_path, model=model)
    if provider == "local":
        return transcribe_audio(file_path, model=model)
    if provider == "groq":
        groq_model = model if model and model not in {"tiny.en", "base.en", "small.en", "medium.en"} else "whisper-large-v3-turbo"
        return _transcribe_groq(file_path, groq_model)
    if provider == "xai":
        return _transcribe_xai(file_path, model or "grok-stt")
    if provider == "elevenlabs":
        elevenlabs_model = model if model and model not in {"tiny.en", "base.en", "small.en", "medium.en"} else "scribe_v2"
        language_code = (elevenlabs_language_code or "").strip()
        if not language_code:
            return _transcribe_elevenlabs(file_path, elevenlabs_model)
        return transcribe_elevenlabs_with_language(file_path, elevenlabs_model, language_code)
    raise ValueError(f"Unsupported STT provider: {provider}")


def safe_stt_summary(result: dict[str, Any]) -> dict[str, Any]:
    summary = {key: result.get(key) for key in ("success", "transcript", "provider", "model", "error")}
    transcript = summary.get("transcript")
    if isinstance(transcript, str):
        summary["transcript"] = normalize_voice_transcript(transcript)
    return summary


def append_jsonl(path: str | None, item: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")


async def run_stt_voice_loop(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(Path(args.env_file).expanduser())
    bot_token = os.getenv("FLUXER_BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError("FLUXER_BOT_TOKEN is required")
    if not os.getenv("XAI_API_KEY", "").strip():
        raise RuntimeError("XAI_API_KEY is required")

    voice_context_cache = load_voice_context_cache(args.voice_context_file)
    args.voice_system_prompt = compose_system_prompt(voice_context_cache=voice_context_cache)

    adapter = FluxerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "bot_token": bot_token,
                "allow_all_users": env_truthy("FLUXER_ALLOW_ALL_USERS"),
                "gateway_state_updates": False,
                "voice": {"supervisor_disabled": True},
            },
        )
    )
    bridge = FluxerLiveKitSmokeBridge(auto_subscribe=True)
    connected = asyncio.Event()
    finished = asyncio.Event()
    shutdown_requested = asyncio.Event()
    leave_connection_id: str | None = None
    result: dict[str, Any] = {
        "mode": "stt_backed_voice_loop",
        "turn_count": 0,
        "published_turn_count": 0,
        "ignored_turn_count": 0,
        "turns": [],
        "voice_context_cache_chars": len(voice_context_cache),
    }
    history: list[dict[str, str]] = []
    sticky_brain_provider = "xai-fast"
    session_task: asyncio.Task[Any] | None = None
    voice_update_task: asyncio.Task[Any] | None = None
    voice_update_lock = asyncio.Lock()
    session_generation = 0

    async def run_voice_session(info: Any, safe_update: dict[str, Any], generation: int) -> None:
        nonlocal sticky_brain_provider
        try:
            # Let Fluxer publish/subscription state settle before the first fixed window.
            await asyncio.sleep(args.initial_settle_seconds)

            async def transcribe_barge_in_stop_phrase(pcm: bytes) -> str:
                wav_path = Path(tempfile.gettempdir()) / f"fluxer_stt_loop_barge_stop_{generation}_{int(time.time() * 1000)}.wav"
                write_pcm16_wav(wav_path, pcm, sample_rate=args.sample_rate)
                stt_result = await asyncio.to_thread(
                    transcribe_with_provider,
                    str(wav_path),
                    provider=args.stt_provider,
                    model=args.stt_model,
                    elevenlabs_language_code=args.elevenlabs_language_code,
                )
                return normalize_voice_transcript((stt_result.get("transcript") or "").strip())

            args.barge_in_stop_phrase_transcriber = transcribe_barge_in_stop_phrase

            for turn_no in range(1, args.max_turns + 1):
                if shutdown_requested.is_set():
                    result["stop_requested"] = True
                    break
                turn_started = time.monotonic()
                try:
                    if args.capture_mode == "vad":
                        pcm = await _capture_one_speech_segment(args, bridge, timeout=args.capture_timeout)
                    else:
                        pcm = await bridge.collect_remote_audio_pcm16(
                            duration_seconds=args.capture_window_seconds,
                            sample_rate=args.sample_rate,
                            frame_size_ms=args.frame_ms,
                            participant_identity=args.participant_identity,
                            participant_identity_prefix=args.participant_identity_prefix,
                            timeout=args.capture_timeout,
                        )
                except (TimeoutError, asyncio.TimeoutError):
                    turn = {
                        "turn": turn_no,
                        "published": False,
                        "reason": "capture_timeout",
                        "capture_timeout_seconds": args.capture_timeout,
                    }
                    result["ignored_turn_count"] += 1
                    result["turns"].append(turn)
                    append_jsonl(args.turn_log_jsonl, turn)
                    continue
                wav_path = Path(tempfile.gettempdir()) / f"fluxer_stt_loop_input_{turn_no}.wav"
                write_pcm16_wav(wav_path, pcm, sample_rate=args.sample_rate)
                stt_started = time.monotonic()
                stt_result = await asyncio.to_thread(
                    transcribe_with_provider,
                    str(wav_path),
                    provider=args.stt_provider,
                    model=args.stt_model,
                    elevenlabs_language_code=args.elevenlabs_language_code,
                )
                stt_seconds = time.monotonic() - stt_started
                transcript = normalize_voice_transcript((stt_result.get("transcript") or "").strip())
                turn: dict[str, Any] = {
                    "turn": turn_no,
                    "captured_pcm_bytes": len(pcm),
                    "captured_audio_seconds": round(len(pcm) / 2 / args.sample_rate, 3),
                    "input_rms": pcm16_rms(pcm),
                    "stt": safe_stt_summary(stt_result),
                    "stt_seconds": round(stt_seconds, 3),
                }

                if not transcript:
                    result["ignored_turn_count"] += 1
                    turn["published"] = False
                    turn["reason"] = "empty_stt_transcript"
                    result["turns"].append(turn)
                    append_jsonl(args.turn_log_jsonl, turn)
                    if args.stop_on_empty_stt:
                        break
                    continue

                if looks_like_clipped_non_english_noise(transcript):
                    result["ignored_turn_count"] += 1
                    turn["published"] = False
                    turn["reason"] = "clipped_non_english_noise"
                    result["turns"].append(turn)
                    append_jsonl(args.turn_log_jsonl, turn)
                    continue

                should_stop_after_reply = is_voice_stop_request(transcript)
                selected_brain_provider, sticky_brain_provider, brain_route_reason = resolve_voice_brain_provider(
                    args.brain_provider,
                    sticky_brain_provider,
                    transcript,
                )
                hermes_session_id, hermes_session_key = hermes_voice_session_identity(args)
                brain_started = time.monotonic()
                if should_stop_after_reply:
                    reply_text = "Got it, stopping here."
                    prompt = reply_text
                elif brain_route_reason.startswith("voice_switch_"):
                    reply_text = voice_mode_ack(selected_brain_provider)
                    prompt = reply_text
                elif selected_brain_provider == "hermes":
                    reply_text = await hermes_chat_completion(transcript, history=history, args=args)
                    prompt = reply_text
                else:
                    prompt = build_answer_prompt(transcript, history=history, system=args.voice_system_prompt)
                    reply_text = ""
                brain_seconds = time.monotonic() - brain_started

                voice = XAIRealtimeVoiceClient(
                    sample_rate=args.sample_rate,
                    voice=args.voice,
                    instructions="Speak the assistant's answer naturally and without extra preamble. Keep operational replies short, but allow 2-4 spoken sentences when the user asks for depth, introspection, or something personal. Do not add generic follow-up questions. Use subtle xAI speech effects sparingly, like <soft>, [breath], or [chuckle], only when they improve the feeling.",
                )
                xai_started = time.monotonic()
                first_audio_seconds: float | None = None
                barge_in_seconds: float | None = None
                barge_in_capture = BargeInCapture()
                barge_in_task: asyncio.Task[Any] | None = None
                interrupt_watcher_task: asyncio.Task[Any] | None = None
                xai_task: asyncio.Task[Any] | None = None
                publisher = bridge.pcm16_publisher(
                    sample_rate=args.sample_rate,
                    frame_ms=args.frame_ms,
                    track_name=f"fluxer-stt-loop-reply-{turn_no}",
                )
                await publisher.__aenter__()
                arm_barge_after_first_audio = bool(getattr(args, "barge_in_after_first_audio_only", True))

                async def interrupt_on_barge_in() -> None:
                    nonlocal barge_in_seconds
                    await barge_in_capture.event.wait()
                    barge_in_seconds = time.monotonic() - xai_started
                    await publisher.interrupt()
                    if xai_task is not None and not xai_task.done():
                        xai_task.cancel()

                def ensure_interrupt_watcher() -> None:
                    nonlocal interrupt_watcher_task
                    if interrupt_watcher_task is None and not args.disable_barge_in:
                        interrupt_watcher_task = asyncio.create_task(interrupt_on_barge_in())

                if not args.disable_barge_in and not arm_barge_after_first_audio:
                    barge_in_task = asyncio.create_task(_wait_for_barge_in(args, bridge, barge_in_capture))
                    ensure_interrupt_watcher()

                async def publish_delta(chunk: bytes) -> None:
                    nonlocal first_audio_seconds, barge_in_seconds, barge_in_task

                    async def should_interrupt() -> bool:
                        nonlocal barge_in_seconds
                        if barge_in_capture.event.is_set():
                            barge_in_seconds = time.monotonic() - xai_started
                            return True
                        return False

                    if await should_interrupt():
                        await publisher.interrupt()
                        raise BargeInInterrupt("user interrupted assistant speech")
                    if first_audio_seconds is None:
                        first_audio_seconds = time.monotonic() - xai_started
                        if arm_barge_after_first_audio and not args.disable_barge_in and barge_in_task is None:
                            barge_in_task = asyncio.create_task(_wait_for_barge_in(args, bridge, barge_in_capture))
                            ensure_interrupt_watcher()
                    write_interruptible = getattr(publisher, "write_interruptible", None)
                    if write_interruptible is not None:
                        interrupted = await write_interruptible(chunk, should_interrupt)
                        if interrupted:
                            raise BargeInInterrupt("user interrupted assistant speech")
                    else:
                        await publisher.write(chunk)

                try:
                    xai_task = asyncio.create_task(
                        voice.force_message_to_sink(
                            prompt,
                            publish_delta,
                            timeout=args.xai_timeout,
                            first_audio_timeout=args.xai_first_audio_timeout,
                        )
                        if reply_text
                        else voice.text_response_to_sink(
                            prompt,
                            publish_delta,
                            timeout=args.xai_timeout,
                            first_audio_timeout=args.xai_first_audio_timeout,
                        )
                    )
                    barge_event_task: asyncio.Task[Any] | None = None
                    if not args.disable_barge_in and not arm_barge_after_first_audio:
                        barge_event_task = asyncio.create_task(barge_in_capture.event.wait())
                    try:
                        if barge_event_task is None:
                            try:
                                xai_result = await xai_task
                            except asyncio.CancelledError:
                                if barge_in_capture.event.is_set():
                                    raise BargeInInterrupt("user interrupted assistant speech")
                                raise
                        else:
                            done, _pending = await asyncio.wait({xai_task, barge_event_task}, return_when=asyncio.FIRST_COMPLETED)
                            if barge_event_task in done and barge_in_capture.event.is_set() and not xai_task.done():
                                barge_in_seconds = time.monotonic() - xai_started
                                await publisher.interrupt()
                                await _cancel_task_safely(xai_task)
                                raise BargeInInterrupt("user interrupted assistant speech before xAI audio")
                            try:
                                xai_result = await xai_task
                            except asyncio.CancelledError:
                                if barge_in_capture.event.is_set():
                                    raise BargeInInterrupt("user interrupted assistant speech")
                                raise
                    finally:
                        if barge_event_task is not None:
                            barge_event_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                                await barge_event_task
                    xai_seconds = time.monotonic() - xai_started
                    if barge_in_capture.event.is_set():
                        barge_in_seconds = time.monotonic() - xai_started
                        await publisher.interrupt()
                        raise BargeInInterrupt("user interrupted assistant speech")
                except asyncio.CancelledError:
                    logger.info("Cancelling STT-backed voice turn %s; interrupting publisher", turn_no)
                    await _cancel_task_safely(xai_task)
                    if not getattr(publisher, "interrupted", False):
                        await publisher.interrupt()
                    turn.update(
                        {
                            "published": False,
                            "interrupted": True,
                            "reason": "session_cancelled",
                            "partial_response_bytes": getattr(publisher, "bytes_published", 0),
                            "brain_provider": selected_brain_provider,
                            "brain_provider_config": args.brain_provider,
                            "brain_route_reason": brain_route_reason,
                            "hermes_session_id": hermes_session_id if selected_brain_provider == "hermes" else None,
                            "hermes_session_key": hermes_session_key if selected_brain_provider == "hermes" else None,
                            "reply_transcript": reply_text,
                            "barge_in_diagnostic": {
                                "chunks_seen": barge_in_capture.chunks_seen,
                                "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                                if barge_in_capture.first_chunk_seconds is not None
                                else None,
                                "max_rms": barge_in_capture.max_rms,
                                "voiced_ms": barge_in_capture.voiced_ms,
                                "detected_voiced_ms": barge_in_capture.detected_voiced_ms,
                                "detected_seconds": round(barge_in_capture.detected_seconds, 3)
                                if barge_in_capture.detected_seconds is not None
                                else None,
                                "threshold": args.barge_in_energy_threshold,
                                "min_ms": args.barge_in_min_ms,
                                "window_ms": getattr(args, "barge_in_window_ms", 0),
                                "semantic_stop_detected": barge_in_capture.semantic_stop_detected,
                                "semantic_stop_transcript": barge_in_capture.semantic_stop_transcript,
                                "semantic_stop_error": barge_in_capture.semantic_stop_error,
                            },
                            "publisher_queue_before_interrupt_seconds": round(
                                getattr(publisher, "last_queue_duration_before_interrupt", 0.0) or 0.0,
                                3,
                            ),
                            "publisher_queue_after_clear_seconds": round(
                                getattr(publisher, "last_queue_duration_after_clear", 0.0) or 0.0,
                                3,
                            ),
                            "timing": {
                                "turn_seconds": round(time.monotonic() - turn_started, 3),
                                "stt_seconds": round(stt_seconds, 3),
                                "brain_seconds": round(brain_seconds, 3),
                                "xai_first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                            },
                        }
                    )
                    result["turns"].append(turn)
                    append_jsonl(args.turn_log_jsonl, turn)
                    raise
                except BargeInInterrupt:
                    logger.info("Barge-in interrupted STT-backed voice turn %s", turn_no)
                    await _cancel_task_safely(xai_task)
                    if barge_in_capture.event.is_set():
                        with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
                            await asyncio.wait_for(
                                barge_in_capture.ready.wait(),
                                timeout=getattr(args, "barge_in_capture_timeout", 2.0),
                            )
                    if not getattr(publisher, "interrupted", False):
                        await publisher.interrupt()
                    turn.update(
                        {
                            "published": False,
                            "interrupted": True,
                            "partial_response_bytes": getattr(publisher, "bytes_published", 0),
                            "brain_provider": selected_brain_provider,
                            "brain_provider_config": args.brain_provider,
                            "brain_route_reason": brain_route_reason,
                            "hermes_session_id": hermes_session_id if selected_brain_provider == "hermes" else None,
                            "hermes_session_key": hermes_session_key if selected_brain_provider == "hermes" else None,
                            "reply_transcript": reply_text,
                            "barge_in_diagnostic": {
                                "chunks_seen": barge_in_capture.chunks_seen,
                                "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                                if barge_in_capture.first_chunk_seconds is not None
                                else None,
                                "max_rms": barge_in_capture.max_rms,
                                "voiced_ms": barge_in_capture.voiced_ms,
                                "detected_voiced_ms": barge_in_capture.detected_voiced_ms,
                                "detected_seconds": round(barge_in_capture.detected_seconds, 3)
                                if barge_in_capture.detected_seconds is not None
                                else None,
                                "captured_audio_seconds": round(barge_in_capture.captured_audio_seconds, 3),
                            },
                            "publisher_queue_before_interrupt_seconds": round(
                                getattr(publisher, "last_queue_duration_before_interrupt", 0.0) or 0.0,
                                3,
                            ),
                            "publisher_queue_after_clear_seconds": round(
                                getattr(publisher, "last_queue_duration_after_clear", 0.0) or 0.0,
                                3,
                            ),
                            "timing": {
                                "turn_seconds": round(time.monotonic() - turn_started, 3),
                                "stt_seconds": round(stt_seconds, 3),
                                "brain_seconds": round(brain_seconds, 3),
                                "xai_first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                                "barge_in_seconds": round(barge_in_seconds, 3) if barge_in_seconds is not None else None,
                            },
                        }
                    )
                    result["turns"].append(turn)
                    append_jsonl(args.turn_log_jsonl, turn)
                    result["stop_requested"] = True
                    break
                finally:
                    if barge_in_task is not None:
                        barge_in_capture.stop_event.set()
                        barge_in_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                            await barge_in_task
                    if interrupt_watcher_task is not None:
                        interrupt_watcher_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                            await interrupt_watcher_task
                    await _cancel_task_safely(xai_task)
                    if not getattr(publisher, "interrupted", False):
                        await publisher.close()
                xai_seconds = time.monotonic() - xai_started
                spoken_reply = reply_text or xai_result.transcript
                history.append({"user": transcript, "assistant": spoken_reply})
                turn.update(
                    {
                        "published": True,
                        "brain_provider": selected_brain_provider,
                        "brain_provider_config": args.brain_provider,
                        "brain_route_reason": brain_route_reason,
                        "hermes_session_id": hermes_session_id if selected_brain_provider == "hermes" else None,
                        "hermes_session_key": hermes_session_key if selected_brain_provider == "hermes" else None,
                        "sticky_brain_provider": sticky_brain_provider,
                        "brain_seconds": round(brain_seconds, 3),
                        "reply_transcript": spoken_reply,
                        "reply_bytes": xai_result.bytes_written,
                        "xai_events_tail": list(xai_result.events_seen[-5:]),
                        "barge_in_diagnostic": {
                            "chunks_seen": barge_in_capture.chunks_seen,
                            "first_chunk_seconds": round(barge_in_capture.first_chunk_seconds, 3)
                            if barge_in_capture.first_chunk_seconds is not None
                            else None,
                            "max_rms": barge_in_capture.max_rms,
                            "voiced_ms": barge_in_capture.voiced_ms,
                            "threshold": args.barge_in_energy_threshold,
                            "min_ms": args.barge_in_min_ms,
                            "semantic_stop_detected": barge_in_capture.semantic_stop_detected,
                            "semantic_stop_transcript": barge_in_capture.semantic_stop_transcript,
                            "semantic_stop_error": barge_in_capture.semantic_stop_error,
                        }
                        if not args.disable_barge_in
                        else None,
                        "timing": {
                            "turn_seconds": round(time.monotonic() - turn_started, 3),
                            "stt_seconds": round(stt_seconds, 3),
                            "brain_seconds": round(brain_seconds, 3),
                            "xai_first_audio_seconds": round(first_audio_seconds, 3) if first_audio_seconds is not None else None,
                            "xai_seconds": round(xai_seconds, 3),
                        },
                    }
                )
                result["published_turn_count"] += 1
                result["turns"].append(turn)
                append_jsonl(args.turn_log_jsonl, turn)
                if should_stop_after_reply:
                    result["stop_requested"] = True
                    break

            result["turn_count"] = len(result["turns"])
        except Exception as exc:
            logger.exception("STT-backed Fluxer voice loop failed")
            result["error"] = type(exc).__name__
            result["message"] = _redact_exception_message(exc)
            connected.set()
        finally:
            if generation == session_generation:
                finished.set()

    async def process_voice_server_update(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        nonlocal leave_connection_id, session_task, session_generation
        try:
            if session_task is not None and not session_task.done():
                session_generation += 1
                session_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                    await session_task
            await _acquire_lock_with_timeout(voice_update_lock, timeout=args.connect_timeout)
            try:
                info = await bridge.connect_from_voice_server_update(raw_update)
            finally:
                voice_update_lock.release()
            leave_connection_id = info.connection_id
            result["connection"] = {
                "endpoint": info.endpoint,
                "guild_id": info.guild_id,
                "channel_id": info.channel_id,
                "connection_id": info.connection_id,
                "room_name": info.room_name,
                "participant_identity": info.participant_identity,
            }
            result["safe_update"] = safe_update
            connected.set()
            session_generation += 1
            session_task = asyncio.create_task(run_voice_session(info, safe_update, session_generation), name="fluxer-stt-voice-session")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("STT-backed Fluxer voice loop failed to join LiveKit")
            result["error"] = type(exc).__name__
            result["message"] = _redact_exception_message(exc, str(raw_update.get("token") or ""))
            connected.set()
            finished.set()

    async def run_voice_update_after_previous(
        previous_task: asyncio.Task[Any] | None,
        raw_update: dict[str, Any],
        safe_update: dict[str, Any],
    ) -> None:
        await _cancel_previous_task_safely(previous_task)
        await process_voice_server_update(raw_update, safe_update)

    async def handler(raw_update: dict[str, Any], safe_update: dict[str, Any]) -> None:
        nonlocal voice_update_task
        previous_task = voice_update_task
        voice_update_task = asyncio.create_task(
            run_voice_update_after_previous(previous_task, raw_update, safe_update),
            name="fluxer-stt-voice-server-update",
        )

    adapter.set_voice_server_update_handler(handler)
    loop = asyncio.get_running_loop()
    previous_signal_handlers: dict[signal.Signals, Any] = {}

    def request_shutdown_from_signal() -> None:
        result["signal_stop_requested"] = True
        shutdown_requested.set()
        if session_task is not None and not session_task.done():
            session_task.cancel()
        elif voice_update_task is not None and not voice_update_task.done():
            voice_update_task.cancel()
        else:
            finished.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[sig] = signal.getsignal(sig)
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, request_shutdown_from_signal)

    connected_ok = await adapter.connect()
    if not connected_ok:
        raise RuntimeError("Fluxer adapter did not connect to gateway")
    try:
        if not await adapter.wait_until_gateway_ready(timeout=args.connect_timeout):
            raise RuntimeError("Fluxer gateway did not emit READY before timeout")
        await adapter.send_voice_state_update(args.channel_id, guild_id=args.guild_id, self_mute=True, self_deaf=False)
        try:
            await asyncio.wait_for(connected.wait(), timeout=args.connect_timeout)
        except asyncio.TimeoutError:
            result["error"] = "TimeoutError"
            result["message"] = f"Timed out waiting {args.connect_timeout} seconds for Fluxer voice server update"
            return result
        try:
            await asyncio.wait_for(finished.wait(), timeout=args.max_runtime_seconds)
        except asyncio.TimeoutError:
            result["error"] = "TimeoutError"
            result["message"] = f"Timed out after max runtime of {args.max_runtime_seconds} seconds"
            result["turn_count"] = len(result["turns"])
            return result
    finally:
        shutdown_requested.set()
        if voice_update_task is not None and not voice_update_task.done():
            voice_update_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await voice_update_task
        if session_task is not None and not session_task.done():
            session_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, RuntimeError):
                await session_task
        with contextlib.suppress(Exception):
            await adapter.send_voice_state_update(None, guild_id=args.guild_id, connection_id=leave_connection_id)
        await bridge.disconnect()
        await adapter.disconnect()
        for sig, old_handler in previous_signal_handlers.items():
            with contextlib.suppress(NotImplementedError):
                loop.remove_signal_handler(sig)
            if old_handler not in (None, signal.SIG_DFL):
                signal.signal(sig, old_handler)
    return result


def _preload_env_file(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env-file", default=os.getenv("HERMES_ENV_FILE", str(Path.home() / ".hermes" / ".env")))
    args, _ = parser.parse_known_args(argv)
    load_env_file(Path(args.env_file).expanduser())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT-backed Fluxer voice room loop")
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--guild-id")
    parser.add_argument("--participant-identity", help="Only capture this exact remote LiveKit participant identity")
    parser.add_argument(
        "--participant-identity-prefix",
        help="Only capture remote LiveKit participants whose identity starts with this prefix, e.g. user_<FluxerUserId>_",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--capture-mode", choices=("vad", "fixed"), default="vad")
    parser.add_argument("--capture-window-seconds", type=float, default=3.0)
    parser.add_argument("--capture-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS", "25.0")))
    parser.add_argument("--initial-settle-seconds", type=float, default=float(os.getenv("FLUXER_VOICE_INITIAL_SETTLE_SECONDS", "0.8")))
    parser.add_argument("--sample-rate", type=int, default=int(os.getenv("FLUXER_VOICE_SAMPLE_RATE", "24000")))
    parser.add_argument("--frame-ms", type=int, default=int(os.getenv("FLUXER_VOICE_FRAME_MS", "20")))
    parser.add_argument("--energy-threshold", type=int, default=int(os.getenv("FLUXER_VOICE_ENERGY_THRESHOLD", "300")))
    parser.add_argument("--silence-ms", type=int, default=int(os.getenv("FLUXER_VOICE_SILENCE_MS", "1500")))
    parser.add_argument("--end-padding-ms", type=int, default=int(os.getenv("FLUXER_VOICE_END_PADDING_MS", "300")))
    parser.add_argument("--min-segment-ms", type=int, default=int(os.getenv("FLUXER_VOICE_MIN_SEGMENT_MS", "1600")))
    parser.add_argument("--max-segment-seconds", type=float, default=float(os.getenv("FLUXER_VOICE_MAX_SEGMENT_SECONDS", "12.0")))
    parser.add_argument("--voice", default=os.getenv("FLUXER_VOICE_TTS_VOICE", "eve"))
    parser.add_argument("--brain-provider", choices=("auto", "xai-fast", "xai", "hermes"), default=os.getenv("FLUXER_VOICE_BRAIN_PROVIDER", "hermes"))
    parser.add_argument("--hermes-url", default=os.getenv("FLUXER_VOICE_HERMES_URL", "http://127.0.0.1:8642"))
    parser.add_argument("--hermes-model", default=os.getenv("FLUXER_VOICE_HERMES_MODEL") or os.getenv("API_SERVER_MODEL_NAME") or "Hermes")
    parser.add_argument("--hermes-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_HERMES_TIMEOUT_SECONDS", "90.0")))
    parser.add_argument("--hermes-max-tokens", type=int, default=int(os.getenv("FLUXER_VOICE_HERMES_MAX_TOKENS", "90")))
    parser.add_argument("--hermes-temperature", type=float, default=float(os.getenv("FLUXER_VOICE_HERMES_TEMPERATURE", "0.4")))
    parser.add_argument(
        "--hermes-session-id",
        default=os.getenv("FLUXER_VOICE_HERMES_SESSION_ID", ""),
        help="Optional stable Hermes session id for persisted full-brain voice turns; defaults to the Fluxer voice room identity",
    )
    parser.add_argument(
        "--hermes-session-key",
        default=os.getenv("FLUXER_VOICE_HERMES_SESSION_KEY", ""),
        help="Optional stable Hermes long-term memory key for persisted full-brain voice turns; defaults to the Fluxer voice room identity",
    )
    parser.add_argument("--stt-provider", choices=("auto", "local", "groq", "xai", "elevenlabs"), default=os.getenv("FLUXER_VOICE_STT_PROVIDER", "elevenlabs"))
    parser.add_argument("--stt-model", default=os.getenv("FLUXER_VOICE_STT_MODEL", "medium.en"), help="STT model; ElevenLabs commonly uses scribe_v2, local commonly uses medium.en, Groq commonly uses whisper-large-v3-turbo")
    parser.add_argument(
        "--elevenlabs-language-code",
        default=os.getenv("FLUXER_VOICE_ELEVENLABS_LANGUAGE_CODE", ""),
        help="Per-run ElevenLabs Scribe language_code override; empty string allows autodetect",
    )
    parser.add_argument("--xai-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_XAI_TIMEOUT_SECONDS", "45.0")))
    parser.add_argument("--xai-first-audio-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_XAI_FIRST_AUDIO_TIMEOUT_SECONDS", "12.0")))
    parser.add_argument("--disable-barge-in", action="store_true", default=env_truthy("FLUXER_VOICE_DISABLE_BARGE_IN"))
    parser.add_argument("--barge-in-energy-threshold", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD", "700")))
    parser.add_argument("--barge-in-min-ms", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_MIN_MS", "180")))
    parser.add_argument("--barge-in-window-ms", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_WINDOW_MS", "0")))
    parser.add_argument("--barge-in-capture-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS", "2.0")))
    parser.add_argument("--barge-in-stop-phrase-energy-threshold", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_STOP_PHRASE_ENERGY_THRESHOLD", "450")))
    parser.add_argument("--barge-in-stop-phrase-min-ms", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MIN_MS", "120")))
    parser.add_argument("--barge-in-stop-phrase-silence-ms", type=int, default=int(os.getenv("FLUXER_VOICE_BARGE_IN_STOP_PHRASE_SILENCE_MS", "180")))
    parser.add_argument("--barge-in-stop-phrase-max-seconds", type=float, default=float(os.getenv("FLUXER_VOICE_BARGE_IN_STOP_PHRASE_MAX_SECONDS", "2.0")))
    parser.add_argument(
        "--barge-in-after-first-audio-only",
        action=argparse.BooleanOptionalAction,
        default=env_truthy("FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY") if os.getenv("FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY") is not None else True,
    )
    parser.add_argument("--connect-timeout", type=float, default=float(os.getenv("FLUXER_VOICE_CONNECT_TIMEOUT_SECONDS", "30.0")))
    parser.add_argument("--max-runtime-seconds", type=float, default=float(os.getenv("FLUXER_VOICE_MAX_RUNTIME_SECONDS", "180.0")))
    parser.add_argument("--env-file", default=os.getenv("HERMES_ENV_FILE", str(Path.home() / ".hermes" / ".env")))
    parser.add_argument(
        "--voice-context-file",
        default=os.getenv("FLUXER_VOICE_CONTEXT_FILE", ""),
        help="Optional deployment-local context file loaded once into RAM at startup for fast voice mode",
    )
    parser.add_argument(
        "--voice-session-db",
        default=os.getenv("FLUXER_VOICE_SESSION_DB", str(Path.home() / ".hermes" / "state.db")),
        help="Read-only Hermes session DB used to augment full-brain temporal recall questions",
    )
    parser.add_argument("--stop-on-empty-stt", action="store_true")
    parser.add_argument(
        "--turn-log-jsonl",
        default=os.getenv("FLUXER_VOICE_TURN_LOG_JSONL", "/tmp/hermes_fluxer_voice_loop_turns.jsonl"),
        help="Append each turn as JSONL so long-running sessions keep transcripts/timing even when stopped",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _preload_env_file(argv)
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    # Avoid leaking Authorization/Cookie headers from websocket/http debug logs.
    # Keep our script/adapter diagnostics verbose, but force dependency transport
    # loggers back down to INFO where they do not dump handshake headers.
    for noisy_secret_logger in ("websockets", "websockets.client", "httpcore", "httpx", "urllib3"):
        logging.getLogger(noisy_secret_logger).setLevel(logging.INFO)
    result = asyncio.run(run_stt_voice_loop(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not result.get("error") else 1


if __name__ == "__main__":
    raise SystemExit(main())
