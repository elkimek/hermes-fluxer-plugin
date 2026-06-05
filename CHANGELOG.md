# Changelog

All notable changes to the Hermes Fluxer plugin are recorded here.

This project uses simple semantic versioning while the plugin is young:

- patch versions for fixes, compatibility improvements, and safe UX polish;
- minor versions for new user-visible capabilities;
- major versions only for breaking configuration or runtime behavior.

## Unreleased

### Added

- Continued the Fluxer realtime voice spike with `REALTIME_VOICE.md`, documenting the discovered LiveKit/opcode-4 voice handshake and the staged path toward a live Žofka voice-room bridge.
- Added tested gateway seams for future realtime voice work: `_build_voice_state_update_payload(...)` and `send_voice_state_update(...)` can send Fluxer `VOICE_STATE_UPDATE` payloads over the existing gateway websocket.
- Added safe `VOICE_SERVER_UPDATE` capture for the spike path: the adapter tracks pending voice joins, records non-secret LiveKit endpoint/connection metadata, and only stores token presence — never the token itself.
- Added an in-memory `VOICE_SERVER_UPDATE` bridge hook so a future LiveKit bridge can receive the raw ephemeral token payload without the adapter persisting or logging that token.
- Started a transport-only LiveKit smoke bridge module with optional `realtime` dependency support; it connects from a raw `VOICE_SERVER_UPDATE` payload and keeps the ephemeral token out of returned/stored/logged state.
- Added a muted/deaf `scripts/fluxer_livekit_smoke.py` probe for the first real Fluxer voice-room presence test.
- Verified the presence-only LiveKit smoke path against hosted Fluxer without logging or storing the ephemeral token.
- Added and verified low-amplitude sine-tone publishing to hosted Fluxer LiveKit for the first audible bot smoke path.
- Added mono 16-bit PCM WAV publishing and verified a generated Žofka TTS clip through hosted Fluxer LiveKit.
- Added a minimal xAI Realtime websocket client plus smoke-probe flags for generating Grok Voice audio and publishing it through Fluxer LiveKit.
- Verified `grok-voice-latest` text-to-voice output through hosted Fluxer LiveKit with `xai_realtime_published: true`.
- Added one-turn duplex smoke plumbing: Fluxer remote audio capture → xAI Realtime audio input → streamed Grok Voice PCM deltas → Fluxer LiveKit publish.
- Verified live remote human-speaker capture and streamed response publishing against hosted Fluxer; first assistant audio arrived at about 2.0s after capture, with only 0.84s final playout drain.
- Tuned live VAD defaults to reduce end-of-turn latency while avoiding false bursts: 600ms silence stop, 180ms retained final silence, and 750ms minimum segment; diagnostics now separate wall-clock capture time from captured audio duration.
- Tightened realtime voice instructions to one short default answer and no multiple follow-up questions.
- Added a first-audio timeout for streamed xAI responses so no-audio provider turns fail fast instead of blocking the room until the full response timeout.
- Added and live-verified barge-in interruption: sustained user speech during assistant output clears the LiveKit audio queue, stops further xAI delta publishing, records an interrupted turn, and resumes listening.
- Added barge-in carryover: the interrupting utterance is captured as short-lived PCM, surfaced only as byte/duration diagnostics, and fed directly into the next xAI turn so Elkim does not need to repeat the interruption after Žofka stops.
- Hardened barge-in interruption after live testing showed assistant speech could continue: streamed xAI deltas are now checked between 20ms LiveKit frames, and interrupt stops/unpublishes the local LiveKit track in addition to clearing the audio queue.
- Added `--diagnose-barge-in`, a silent LiveKit-only probe that keeps a local output track active while measuring remote mic RMS/detection, plus richer xAI failure messages for first-audio latency/debugging.

## [0.1.1] - 2026-06-05

### Added

- Added a documented voice-message roundtrip path: inbound Fluxer voice messages are treated as spoken user input, while outbound Hermes voice replies use Fluxer voice-message upload metadata when supported by the deployment.
- Added regression tests for inbound Fluxer voice-message normalization, MIME inference, outbound `send_voice` payload shape, and zero-duration non-voice audio handling.
- Added GitHub Actions CI, quality, and security workflows for Python matrix tests, linting, dependency audit, secret-shaped placeholder scanning, and CodeQL.
- Added standalone test doubles for Hermes gateway types so the plugin test suite can run in clean CI without installing the full Hermes repository.

### Changed

- Fluxer attachments with voice-message shape (`VOICE_MESSAGE` flags, explicit voice markers, or duration/waveform metadata) now normalize to Hermes `MessageType.VOICE` instead of generic audio.
- Voice-shaped WebM attachments now normalize `video/webm` filename guesses to `audio/webm`, and inbound voice events carry safe `fluxer_voice_message` metadata (`content_type`, duration, waveform presence) for downstream STT/logging without storing waveform blobs.
- Voice metadata extraction now skips non-voice attachments that may precede the actual voice file and preserves an explicit zero-duration value instead of falling through to fallback duration keys.
- Native approval and slash-confirm controls now share one component registration path, keeping Fluxer button state consistent with reaction/text fallbacks.
- Attachments without an explicit MIME type now infer the MIME type from the filename, so files such as `voice-message.ogg` are cached as `audio/ogg` and can enter the normal Hermes STT path.
- Documentation now uses safe placeholders for Fluxer bot tokens instead of token-shaped examples.

### User impact

- Sending a native Fluxer voice message is now reliably handled like spoken chat and can be transcribed by Hermes.
- Generic audio files remain generic audio attachments, so music, podcasts, and other non-chat audio are not automatically treated as spoken input.

### Verification

- `python -m pytest -q` → 25 passed
- `python -m compileall -q .`
- `python -m ruff check .`
- `git diff --check`
- Strict secret-shape scan → 0 hits
- Live Fluxer smoke test: a post-restart Fluxer voice message was transcribed and delivered into the Hermes prompt.

## [0.1.0] - 2026-06-04

### Added

- Initial standalone Fluxer platform plugin for Hermes Agent.
- Fluxer bot REST sends, Gateway WebSocket inbound messages, message gating, replies, edits/deletes, pins, reactions, components where supported, media delivery, backlog recovery, channel discovery, home-channel delivery, and optional native slash-command registration.
- Safety defaults: deny-by-default user access, mention-gated group behavior, allowed-channel controls, and outbound broad-mention sanitization.
- Human and agent setup docs: `README.md`, `INSTALL_FOR_AGENTS.md`, `AGENTS.md`, and `after-install.md`.
