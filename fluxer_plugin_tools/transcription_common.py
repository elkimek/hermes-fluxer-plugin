"""Standalone STT constants matching Hermes' public transcription module."""

from __future__ import annotations

import os

ELEVENLABS_STT_BASE_URL = os.getenv(
    "ELEVENLABS_STT_BASE_URL", "https://api.elevenlabs.io/v1"
)
