"""Plugin-private runtime helpers for the Fluxer voice scripts.

This package is deliberately **not** named ``tools``. A top-level ``tools``
package inside the plugin shadows the host's own ``tools`` package whenever the
plugin root precedes the Hermes root on ``sys.path`` — and the voice scripts put
the plugin root first so their own top-level modules (``adapter``,
``livekit_bridge``, ``xai_realtime``) win. Under that order a plugin ``tools``
package hid ``tools.hook_output_spill`` and every other host ``tools.*`` module
the adapter imports lazily, while the reverse order silently resolved the
scripts' STT helpers to the host's ``tools.transcription_tools``, which has no
``_extract_transcript_text`` / ``get_env_value`` / ``ELEVENLABS_STT_BASE_URL``.

The private name removes the collision in both directions: the scripts import
``fluxer_plugin_tools.*`` unambiguously, and host ``tools.*`` keeps resolving to
Hermes.
"""

from __future__ import annotations

__all__ = ["transcription_common", "transcription_tools"]
