"""Admission of plain-text commands and surviving a refused reply reference.

Two delivery-robustness rules for a platform with no native command surface and a
reference field the API can refuse outright:

- A command typed as plain text is admitted without a mention. Fluxer registers
  no application commands, so ``/new``, ``/help``, ``/stop`` and ``/title`` only
  ever arrive as messages, and a mention gate that drops them makes the whole
  command pipeline unreachable in exactly the rooms the gate protects -- with no
  reply to tell the user why.
- A refused reply reference is retried WITHOUT the reference. Fluxer answers
  ``400 CANNOT_REPLY_TO_SYSTEM_MESSAGE`` for a reference to a message it types as
  a system message (the gateway's own ``type: 4`` progress bubbles), and that
  refusal names the reference, never the content, so re-sending the identical
  payload can only fail again -- which is how a reply gets dropped entirely.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock

import adapter as fluxer_adapter
from gateway.config import PlatformConfig

GUILD = "guild-1"
GATED_CHANNEL = "chan-gated"
COMMAND_TEXTS = ["/new", "/help", "/stop", "/title some words", "/skills"]
PLAIN_TEXTS = ["hello there", "what is /usr/bin for", "just chatting"]


def _adapter(**extra) -> fluxer_adapter.FluxerAdapter:
    options = {
        "bot_token": "app.secret",
        "allow_all_users": True,
        "require_mention": True,
        "delivery_verification": False,
    }
    options.update(extra)
    return fluxer_adapter.FluxerAdapter(PlatformConfig(enabled=True, extra=options))


def _gate(adapter, channel_id=GATED_CHANNEL, text="hello", *, guild=GUILD, chat_type="text"):
    return adapter._should_process_message(
        channel_id=channel_id,
        chat_type=chat_type,
        text=text,
        data={"guild_id": guild} if guild else {},
        reply_to_message_id=None,
    )[0]


def _rejected(exc: Exception) -> bool:
    return fluxer_adapter._is_reply_reference_rejected_error(exc)


# ── the rule itself ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("text", COMMAND_TEXTS)
def test_a_text_command_is_recognized(text):
    assert fluxer_adapter._is_text_command(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "/",  # a lone slash names nothing
        "/usr/bin/python script.py",  # a pasted path is not a command
        "what about /new?",  # not command-shaped
        "hello /new",
        "//",
    ],
)
def test_a_non_command_is_not_recognized(text):
    assert fluxer_adapter._is_text_command(text) is False


def test_a_command_may_carry_the_bot_name_suffix():
    """Telegram-style ``/cmd@botname`` is still the same command."""
    assert fluxer_adapter._is_text_command("/new@claire") is True
    assert fluxer_adapter._is_text_command("  /help@claire args  ") is True


# ── admission under an active mention gate ──────────────────────────────────────


@pytest.mark.parametrize("text", COMMAND_TEXTS)
def test_a_command_is_admitted_without_a_mention(text):
    adapter = _adapter(mention_gated_channels=[GATED_CHANNEL])

    assert _gate(adapter, text=text) is True


@pytest.mark.parametrize("text", PLAIN_TEXTS)
def test_ordinary_text_still_needs_a_mention(text):
    """The bypass covers commands only; the gate is otherwise untouched."""
    adapter = _adapter(mention_gated_channels=[GATED_CHANNEL])

    assert _gate(adapter, text=text) is False


def test_a_command_is_admitted_under_strict_mention():
    adapter = _adapter(strict_mention=True)

    assert _gate(adapter, text="/new") is True
    assert _gate(adapter, text="hello") is False


def test_a_command_is_admitted_when_the_text_is_only_a_command_ip_marker():
    """Whitespace and a leading alias suffix must not defeat the rule."""
    adapter = _adapter(mention_gated_channels=[GATED_CHANNEL])

    assert _gate(adapter, text="   /stop   ") is True
    assert _gate(adapter, text="/title@claire renamed") is True


def test_a_command_does_not_override_the_channel_allowlist():
    """A hard allowlist excludes the channel for every message, commands too."""
    adapter = _adapter(allowed_channels=["chan-allowed"])

    assert _gate(adapter, text="/new") is False
    assert _gate(adapter, channel_id="chan-allowed", text="hello") is False
    assert _gate(adapter, channel_id="chan-allowed", text="/new") is True


def test_a_dm_is_admitted_whatever_it_says():
    adapter = _adapter(mention_gated_channels=[GATED_CHANNEL])

    assert _gate(adapter, chat_type="dm", text="hello") is True


def test_a_command_is_admitted_without_a_guild_id():
    """A guildless event must not fall back to needing a mention for a command."""
    adapter = _adapter(mention_gated_channels=[GATED_CHANNEL])

    assert _gate(adapter, text="/new", guild=None) is True
    assert _gate(adapter, text="hello", guild=None) is False


# ── a refused reply reference must not cost the reply ───────────────────────────


def _http_400(body: str) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.fluxer.app/v1/channels/chan-1/messages")
    response = httpx.Response(400, request=request, text=body)
    return httpx.HTTPStatusError("400 Bad Request", request=request, response=response)


SYSTEM_MESSAGE_REFUSAL = (
    '{"code":"INVALID_FORM_BODY","message":"Invalid form body.","errors":['
    '{"path":"message_reference","message":"You can\'t reply to a system message.",'
    '"code":"CANNOT_REPLY_TO_SYSTEM_MESSAGE"}]}'
)


def test_the_system_message_refusal_is_recognized():
    assert _rejected(_http_400(SYSTEM_MESSAGE_REFUSAL)) is True


@pytest.mark.parametrize(
    "body",
    [
        '{"code":"UNKNOWN_CHANNEL","message":"Channel wasn\'t found."}',
        '{"code":"INVALID_FORM_BODY","message":"Invalid form body."}',
        "not json at all",
    ],
)
def test_other_failures_are_not_mistaken_for_a_reference_refusal(body):
    assert _rejected(_http_400(body)) is False


def test_a_missing_response_is_not_a_reference_refusal():
    assert _rejected(RuntimeError("connection reset")) is False


@pytest.mark.asyncio
async def test_a_refused_reference_is_retried_without_it():
    adapter = _adapter()
    refusal = _http_400(SYSTEM_MESSAGE_REFUSAL)
    calls: list[dict] = []

    async def fake_request(method, path, **kwargs):
        # Copy: the adapter mutates its chunk payload in place, so recording the
        # dict itself would show the post-fallback state for every attempt.
        calls.append(dict(kwargs.get("json") or {}))
        if len(calls) == 1:
            raise refusal
        return {"id": "msg-2", "content": "still delivered"}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    result = await adapter.send("chan-1", "still delivered", reply_to="system-msg-1")

    assert result.success is True
    assert result.message_id == "msg-2"
    assert len(calls) == 2
    assert calls[0].get("message_reference") == {"message_id": "system-msg-1"}
    assert "message_reference" not in calls[1]


@pytest.mark.asyncio
async def test_a_refused_reference_does_not_leak_into_later_chunks():
    """The reference is dropped from the shared payload, not just this chunk."""
    adapter = _adapter(reply_to_mode="all")
    refusal = _http_400(SYSTEM_MESSAGE_REFUSAL)
    calls: list[dict] = []

    async def fake_request(method, path, **kwargs):
        body = dict(kwargs.get("json") or {})
        calls.append(body)
        if len(calls) == 1:
            raise refusal
        return {"id": f"msg-{len(calls)}", "content": body.get("content")}

    adapter._request = fake_request
    adapter._verify_delivery = AsyncMock(return_value=None)

    long_text = "x" * 4500
    result = await adapter.send("chan-1", long_text, reply_to="system-msg-1")

    assert result.success is True
    retried = [body for body in calls if body.get("content", "").startswith("x")]
    assert all("message_reference" not in body for body in retried[1:])


@pytest.mark.asyncio
async def test_an_unrelated_failure_is_still_reported_as_a_failure():
    """Only the reference refusal is absorbed; a real delivery error still fails."""
    adapter = _adapter()
    adapter._request = AsyncMock(side_effect=_http_400('{"code":"SERVER_ERROR"}'))
    adapter._verify_delivery = AsyncMock(return_value=None)

    result = await adapter.send("chan-1", "boom", reply_to="msg-1")

    assert result.success is False
    assert result.retryable is True
    # One attempt: the reference refusal is the only failure worth a second try.
    assert adapter._request.await_count == 1
