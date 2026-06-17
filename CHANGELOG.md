# Changelog

All notable changes to the Hermes Fluxer plugin are recorded here.

This project uses simple semantic versioning while the plugin is young:

- patch versions for fixes, compatibility improvements, and safe UX polish;
- minor versions for new user-visible capabilities;
- major versions only for breaking configuration or runtime behavior.

## [0.2.3] - 2026-06-17

### Fixed

- Restored visible tool-progress and streaming update bubbles in Fluxer chat after a compatibility regression made them silently disappear.

### User impact

- You again see live "⚙️ terminal" and other tool-progress messages while Hermes is working, instead of only seeing the final answer after it finishes.

### Verification

- `python -m py_compile adapter.py` → clean
- `pytest -q tests/test_plugin_package.py -k 'code_block or platform or adapter'` → 4 passed
- Live Hermes/Fluxer gateway restart verified `fluxer` connected and the voice auto-join sidecar running.
- Confirmed a tool-triggering message from Fluxer now renders the progress bubble.

## [0.2.2] - 2026-06-09

### Changed

- Advertised Fluxer's markdown code-block capability to Hermes so terminal tool-progress messages render full commands as clean fenced code blocks instead of truncated one-line previews on Hermes versions that support this display mode.

### User impact

- When Hermes runs a terminal command from Fluxer, the live progress bubble is easier to read and copy: Fluxer shows the command in its native code-block UI while the final answer remains unchanged.

### Verification

- `PYTHONPATH=<hermes-agent-checkout>:. pytest -q tests/test_plugin_package.py` → 57 passed
- `PYTHONPATH=<hermes-agent-checkout>:. python -m py_compile adapter.py tests/test_plugin_package.py`
- Live Hermes/Fluxer gateway restart verified `fluxer` connected and `supports_code_blocks=True` for the standalone plugin.

## [0.2.1] - 2026-06-08

### Changed

- Improved realtime Fluxer voice interruption behavior for speaker/soundbar setups: the assistant now ignores short echo bursts more reliably while still allowing the user to interrupt with natural stop phrases like “stop”, “stop counting”, “enough”, or “wait”.
- Kept the assistant in the voice room after a barge-in interruption, so stopping speech no longer forces the user to leave and rejoin before continuing the conversation.
- Made voice auto-join and barge-in handling more stable across reconnects, duplicate voice updates, and restart/recovery paths.
- Expanded the public voice configuration docs with the new barge-in tuning options.

### Verification

- `pytest -q` → 190 passed
- Live Fluxer voice-room testing with soundbar echo: no-interrupt echo test passed; short/long/natural stop interruptions worked; assistant stayed joined after interruption.

## [0.2.0] - 2026-06-06

### Added

- Continued the Fluxer realtime voice spike with `REALTIME_VOICE.md`, documenting the discovered LiveKit/opcode-4 voice handshake and the staged path toward a live assistant voice-room bridge.
- Added tested gateway seams for future realtime voice work: `_build_voice_state_update_payload(...)` and `send_voice_state_update(...)` can send Fluxer `VOICE_STATE_UPDATE` payloads over the existing gateway websocket.
- Added safe `VOICE_SERVER_UPDATE` capture for the spike path: the adapter tracks pending voice joins, records non-secret LiveKit endpoint/connection metadata, and only stores token presence — never the token itself.
- Added an in-memory `VOICE_SERVER_UPDATE` bridge hook so a future LiveKit bridge can receive the raw ephemeral token payload without the adapter persisting or logging that token.
- Started a transport-only LiveKit smoke bridge module with optional `realtime` dependency support; it connects from a raw `VOICE_SERVER_UPDATE` payload and keeps the ephemeral token out of returned/stored/logged state.
- Added a muted/deaf `scripts/fluxer_livekit_smoke.py` probe for the first real Fluxer voice-room presence test.
- Verified the presence-only LiveKit smoke path against hosted Fluxer without logging or storing the ephemeral token.
- Added and verified low-amplitude sine-tone publishing to hosted Fluxer LiveKit for the first audible bot smoke path.
- Added mono 16-bit PCM WAV publishing and verified a generated assistant TTS clip through hosted Fluxer LiveKit.
- Added a minimal xAI Realtime websocket client plus smoke-probe flags for generating Grok Voice audio and publishing it through Fluxer LiveKit.
- Verified `grok-voice-latest` text-to-voice output through hosted Fluxer LiveKit with `xai_realtime_published: true`.
- Added one-turn duplex smoke plumbing: Fluxer remote audio capture → xAI Realtime audio input → streamed Grok Voice PCM deltas → Fluxer LiveKit publish.
- Verified live remote human-speaker capture and streamed response publishing against hosted Fluxer; first assistant audio arrived at about 2.0s after capture, with only 0.84s final playout drain.
- Tuned live VAD defaults to reduce end-of-turn latency while avoiding false bursts: 600ms silence stop, 180ms retained final silence, and 750ms minimum segment; diagnostics now separate wall-clock capture time from captured audio duration.
- Tightened realtime voice instructions to one short default answer and no multiple follow-up questions.
- Added a first-audio timeout for streamed xAI responses so no-audio provider turns fail fast instead of blocking the room until the full response timeout.
- Added barge-in interruption plumbing: sustained user speech during assistant output can clear the LiveKit audio queue, stop further xAI delta publishing, record an interrupted turn, and resume listening.
- Added barge-in carryover: the interrupting utterance is captured as short-lived PCM, surfaced only as byte/duration diagnostics, and fed directly into the next xAI turn so the user does not need to repeat the interruption after the assistant stops.
- Hardened barge-in interruption after live testing showed assistant speech could continue: streamed xAI deltas are now checked between 20ms LiveKit frames, and interrupt stops/unpublishes the local LiveKit track in addition to clearing the audio queue.
- Added `--diagnose-barge-in`, an audible LiveKit-only probe that keeps a local output track active while measuring remote mic RMS/detection, plus richer xAI failure messages for first-audio latency/debugging.
- Fixed the xAI realtime audio-input path for captured Fluxer voice by sending captured PCM as an explicit `conversation.item.create` `input_audio` item before `response.create`; the older `input_audio_buffer.append`/`commit` path committed speech but did not produce responses for live Fluxer captures.
- Added LiveKit subscriber confirmation before streaming local audio frames, so smoke tests now distinguish "track published" from "Fluxer client subscribed and can hear it".
- Added `scripts/fluxer_stt_voice_loop.py`, a separate STT-backed realtime loop that targets the user's LiveKit participant prefix, captures a fixed audio window, transcribes with Hermes STT, prompts a text-grounded answer, and speaks it through xAI Voice back into Fluxer. This avoids the direct xAI audio-understanding path that live-tested as generic filler.
- Reduced STT-backed loop latency by shortening the reliable fixed capture window to 3s and streaming text-grounded xAI voice deltas directly into LiveKit instead of waiting for a full WAV before publishing; after live Groq/xAI tests proved fast but less accurate on Fluxer room captures, the default STT was restored to local `medium.en` for accuracy, with `--stt-provider groq|xai` still available as explicit overrides.
- Added LiveKit participant-identity-prefix filtering so realtime capture can target the user's Fluxer user track (`user_<id>_*`) instead of listening to every remote audio source; targeted fixed-window STT verified the captured prompt as "Shevka, what is two past two?" while the unfiltered/VAD path produced generic/empty understanding.
- Added explicit `--stt-provider elevenlabs` support for ElevenLabs Scribe (`scribe_v2`) alongside local/Groq/xAI STT; live test was very fast (~0.86s STT) but misheard the Fluxer room capture as "Asia FC", confirming provider swaps are not enough and the next focus should be LiveKit capture timing/quality.
- Hardened the STT-backed loop against non-spoken recalled-context wrappers by stripping `<memory-context>` and `[System note: ...]` blocks before prompting Hermes or writing safe turn summaries; this keeps live voice answers focused on what the user actually said.
- Added a production-safe realtime voice configuration surface: `FLUXER_VOICE_*` optional environment variables in `plugin.yaml`, plus equivalent `platforms.fluxer.extra.voice` YAML mapping through the Hermes platform adapter.
- Added `docs/voice-configuration.md` with the full realtime voice dashboard/config reference, provider and hardware tuning notes, VAD guidance, troubleshooting, and public-repo hygiene rules.
- Added regression coverage for the YAML-to-env bridge, plugin-managed voice supervisor lifecycle, and for keeping private dogfood IDs, local paths, and deployment context files out of the public tree.
- Added persisted Hermes voice-session headers so live voice turns can appear in normal Hermes session history instead of only local JSONL diagnostics.
- Added crash-recovery and live barge-in coverage for the auto-join supervisor and STT loop, including child restart, tuned barge-in argument propagation, and independent interruption while xAI/audio streaming stalls.

### Changed

- Realtime voice auto-join is now disabled by default, plugin-managed after Fluxer gateway connect, stopped during adapter disconnect, and refuses to join arbitrary voice rooms unless explicitly enabled and scoped with configured channel IDs.
- Replaced dogfood-specific voice defaults with generic assistant prompts, deployment-local context file support, safe home-relative paths, and environment-driven STT/TTS/VAD/timeout knobs.
- Removed the tracked deployment-local voice context cache; operators should provide private context via `FLUXER_VOICE_CONTEXT_FILE` or `platforms.fluxer.extra.voice.context_file`.
- Defaulted realtime voice to the persisted Hermes brain for consistent assistant behavior, with faster provider modes still available as explicit tuning choices.
- Hardened LiveKit smoke playback, xAI force-message generation, the legacy xAI room loop, nested barge-in config bridging, Python 3.10 timeout handling, timeout/cleanup API consistency, diagnostic-path redaction, supervisor spawn isolation, stale voice-join state cleanup, voice child credential forwarding, STT reconnect handling, and release metadata after review: one-shot audio tracks are now unpublished/stopped after playout, forced text-to-speech requests explicitly trigger `response.create`, xAI server error events preserve the `XAIRealtimeStreamError` diagnostic envelope, remote stream end exits cleanly, final partial speech segments are preserved, YAML `voice.barge_in.*` settings reach both the gateway and child supervisor, asyncio timeouts are caught/suppressed compatibly across supported Python versions, public WAV and sink APIs wrap timeouts consistently, sink APIs preserve provider event tails by using outer safety timeouts, failed publisher setup is closed defensively, early remote-audio collection explicitly closes async generators even after collection timeout with bounded cleanup, the barge-in diagnostic suppresses the closed-publisher race, LiveKit connection exceptions are redacted before JSON output, stale pending voice joins are cleared on gateway reconnect and explicit leave, config.yaml-only `bot_token`/base/gateway settings are forwarded to the gateway and child voice supervisor without overriding env vars, YAML `channel_ids: null` stays unscoped instead of starting the supervisor, STT LiveKit connect/reconnect work is serialized outside the gateway handler and cancels stale sessions before second `VOICE_SERVER_UPDATE` joins, the legacy xAI room loop now schedules LiveKit handoff outside the gateway handler and cancels in-flight xAI tasks before publisher close, unmatched delayed voice-server updates do not trigger the bridge handoff, the voice supervisor uses an absolute script path, restart-watches unexpected outer supervisor exits without respawning after stop, suppresses process-exit races during shutdown, a misconfigured voice supervisor can no longer bring down the text gateway, and the Fluxer User-Agent now matches the 0.2 release line.

### Verification

- `PYTHONPATH=. pytest -q` → 160 passed
- `python3 -m py_compile adapter.py livekit_bridge.py xai_realtime.py scripts/*.py`
- `git diff --check`
- Greptile local review against `origin/main` → accepted findings fixed: one-shot/streaming LiveKit track cleanup, xAI force-message `response.create`, release version bump, Python 3.10 asyncio timeout compatibility, supervisor spawn isolation from text gateway connectivity, LiveKit publish-failure cleanup, remote audio stream task cleanup including bounded timeout cleanup, stale pending voice-join cleanup, xAI error-event/malformed-event diagnostic wrapping, YAML credential forwarding to the gateway and child voice process, YAML null-value skipping, YAML `channel_ids: null` supervisor scoping, explicit leave pending-join cleanup, delayed unmatched voice-server handoff suppression, outer supervisor absolute script paths, outer supervisor restart watcher, restart-window cancellation on stop, shutdown ProcessLookupError suppression, clean timeout exits in the continuous xAI room loop, non-blocking xAI room LiveKit handoff including second-update cancellation, diagnostic publish-task exception suppression, non-blocking and serialized STT voice server update handling outside the gateway handler, second `VOICE_SERVER_UPDATE` stale-session cancellation, xAI task cancellation before publisher close, sink timeout event-tail preservation, tracked auto-join process watcher cleanup, non-blocking auto-join start/stop transitions, WAV timeout event-tail preservation, typed barge-in interrupt propagation, and empty ElevenLabs language-code omission
- Voice env/config audit: 44 `FLUXER_VOICE_*` variables used by code, declared in `plugin.yaml`, and documented in `docs/voice-configuration.md`
- Private dogfood grep audit for user IDs, voice/guild IDs, local paths, context-cache filename, and assistant-specific names → 0 shippable hits

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
