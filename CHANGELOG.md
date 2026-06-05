# Changelog

All notable changes to the Hermes Fluxer plugin are recorded here.

This project uses simple semantic versioning while the plugin is young:

- patch versions for fixes, compatibility improvements, and safe UX polish;
- minor versions for new user-visible capabilities;
- major versions only for breaking configuration or runtime behavior.

## Unreleased

### Added

- Started the Fluxer realtime voice spike with `REALTIME_VOICE.md`, documenting the discovered LiveKit/opcode-4 voice handshake and the staged path toward a live Žofka voice-room bridge.
- Added tested gateway seams for future realtime voice work: `_build_voice_state_update_payload(...)` and `send_voice_state_update(...)` can now send Fluxer `VOICE_STATE_UPDATE` payloads over the existing gateway websocket.
- Added safe `VOICE_SERVER_UPDATE` capture: the adapter tracks pending voice joins, records non-secret LiveKit endpoint/connection metadata, and only stores token presence — never the token itself.

## [0.1.1] - 2026-06-05

### Added

- Added a documented voice-message roundtrip path: inbound Fluxer voice messages are treated as spoken user input, while outbound Hermes voice replies use Fluxer voice-message upload metadata when supported by the deployment.
- Added regression tests for inbound Fluxer voice-message normalization, MIME inference, outbound `send_voice` payload shape, and zero-duration non-voice audio handling.
- Added GitHub Actions CI, quality, and security workflows for Python matrix tests, linting, dependency audit, secret-shaped placeholder scanning, and CodeQL.
- Added standalone test doubles for Hermes gateway types so the plugin test suite can run in clean CI without installing the full Hermes repository.

### Changed

- Fluxer attachments with voice-message shape (`VOICE_MESSAGE` flags, explicit voice markers, or duration/waveform metadata) now normalize to Hermes `MessageType.VOICE` instead of generic audio.
- Attachments without an explicit MIME type now infer the MIME type from the filename, so files such as `voice-message.ogg` are cached as `audio/ogg` and can enter the normal Hermes STT path.
- Documentation now uses safe placeholders for Fluxer bot tokens instead of token-shaped examples.

### User impact

- Sending a native Fluxer voice message is now reliably handled like spoken chat and can be transcribed by Hermes.
- Generic audio files remain generic audio attachments, so music, podcasts, and other non-chat audio are not automatically treated as spoken input.

### Verification

- `PYTHONPATH=. python -m py_compile adapter.py tests/test_plugin_package.py`
- `PYTHONPATH=. pytest -q` → 20 passed
- Live Fluxer smoke test: a post-restart Fluxer voice message was transcribed and delivered into the Hermes prompt.

## [0.1.0] - 2026-06-04

### Added

- Initial standalone Fluxer platform plugin for Hermes Agent.
- Fluxer bot REST sends, Gateway WebSocket inbound messages, message gating, replies, edits/deletes, pins, reactions, components where supported, media delivery, backlog recovery, channel discovery, home-channel delivery, and optional native slash-command registration.
- Safety defaults: deny-by-default user access, mention-gated group behavior, allowed-channel controls, and outbound broad-mention sanitization.
- Human and agent setup docs: `README.md`, `INSTALL_FOR_AGENTS.md`, `AGENTS.md`, and `after-install.md`.
