# Fluxer realtime voice spike

This is the first design note for making the standalone plugin do more than Discord-style text, files, and voice-message STT. The target is a real-time “talk with Žofka in a Fluxer voice room” path.

## What Fluxer exposes

Verified from Fluxer docs/source on 2026-06-05:

- Fluxer voice/video is backed by **LiveKit**.
- Voice channel join starts on the main Fluxer gateway with **opcode 4** (`VOICE_STATE_UPDATE`).
- The client sends a payload shaped like:
  - `guild_id`: guild snowflake or `null` for DM/group call
  - `channel_id`: voice channel snowflake, or `null` to disconnect
  - `self_mute`, `self_deaf`, `self_video`, `self_stream`
  - `viewer_stream_keys`: array
  - `connection_id`: existing connection id or `null`
- Fluxer then dispatches `VOICE_SERVER_UPDATE` containing a LiveKit `endpoint`, `token`, `connection_id`, and channel/guild context.
- The browser client connects to LiveKit with that endpoint/token and publishes/subscribes audio/video tracks.

The public `topics/voice` doc is still `TBD`, so this is grounded mainly in Fluxer source:

- `packages/constants/src/GatewayConstants.tsx`: `VOICE_STATE_UPDATE: 4`
- `fluxer_app/src/lib/GatewaySocket.tsx`: `updateVoiceState(...)`
- `fluxer_app/src/stores/voice/VoiceConnectionManager.tsx`: waits for `VOICE_SERVER_UPDATE`, then `room.connect(endpoint, token, ...)`
- `fluxer_gateway/src/gateway/gateway_rpc_voice.erl`: voice RPCs and LiveKit confirm path

## Architecture target

Smallest useful production shape:

```text
Fluxer gateway websocket
  ├─ MESSAGE_CREATE voice messages → existing Hermes STT path
  └─ VOICE_STATE_UPDATE / VOICE_SERVER_UPDATE
        ↓
Fluxer realtime voice bridge
  ├─ joins Fluxer LiveKit room as the bot
  ├─ subscribes to allowed user audio
  ├─ streams audio into xAI Realtime / other realtime STT+LLM backend
  ├─ receives assistant audio chunks
  └─ publishes Žofka audio back into the LiveKit room
```

Keep this as a separate bridge/service at first, with the standalone plugin exposing tested gateway seams. That avoids turning the platform adapter into a full media engine too early and lets Hermes reuse the pattern later for other voice-capable platforms.

## Safety boundaries

- Only listen in explicitly configured voice channels.
- Default to deaf/listen-disabled until an explicit `join` action is triggered.
- Keep live-session memory exposure tightly bounded: voice-room context should pass platform/channel/session metadata, not broad memory dumps.
- Fail closed if no Fluxer gateway websocket is connected or if the `VOICE_SERVER_UPDATE` channel/guild does not match the requested join.
- Log token presence only, never LiveKit tokens or bot tokens.

## Implementation phases

### Phase 1 — gateway handshake scaffold

Status: started in this repo.

- Add a tested `_build_voice_state_update_payload(...)` helper for opcode 4.
- Add a tested `send_voice_state_update(...)` adapter method that writes the payload to the existing Fluxer gateway websocket.
- Do not connect to LiveKit yet.

### Phase 2 — observe `VOICE_SERVER_UPDATE`

Status: implemented in this repo.

- Teach `_handle_gateway_dispatch` to recognize `VOICE_SERVER_UPDATE`.
- Keep a small pending-join map keyed by `(guild_id, channel_id)` and `connection_id`.
- Store only non-secret metadata: endpoint, connection id, guild/channel ids, and token presence.
- Never retain or log the LiveKit token.
- Expose an in-memory bridge callback/hook for the raw `VOICE_SERVER_UPDATE` payload so a future LiveKit bridge can consume the ephemeral token without writing it into persistent adapter state.

### Phase 3 — LiveKit smoke bridge

Status: xAI Realtime text-to-voice publishing verified against hosted Fluxer on 2026-06-05.

- Added `livekit_bridge.py`, a minimal transport-only smoke bridge that can connect to the Fluxer LiveKit room from a raw `VOICE_SERVER_UPDATE` payload and disconnect cleanly.
- The bridge uses the ephemeral token only as the local `Room.connect(endpoint, token, ...)` argument; it never stores, returns, or logs the token.
- Added optional dependency group: `pip install 'hermes-fluxer-plugin[realtime]'` installs the Python LiveKit SDK.
- Added `scripts/fluxer_livekit_smoke.py` to run real probes against a configured Fluxer voice channel; it joins muted/deaf by default, can publish a short low-amplitude sine tone, prints only safe metadata, then leaves.
- Verified the smoke probe against hosted Fluxer: the bot received a sanitized `VOICE_SERVER_UPDATE`, connected to `wss://*.fluxer.media`, entered `guild_..._channel_...` with a LiveKit participant identity, then left cleanly. Token presence was reported only as `has_token: true`.
- Verified audible publishing against hosted Fluxer: the probe joined unmuted/deaf, published a short low-amplitude test tone with `tone_published: true`, then disconnected.
- Added mono 16-bit PCM WAV publishing and verified a generated Žofka TTS clip against hosted Fluxer with `wav_published: true`.
- Added a minimal xAI Realtime websocket client that can request `grok-voice-latest` PCM16 audio from either a text prompt or xAI `force_message`, write it as WAV, and hand it to the Fluxer LiveKit publisher.
- Verified xAI Realtime end-to-end into Fluxer: `grok-voice-latest` produced PCM16 audio over `wss://api.x.ai/v1/realtime`, the smoke probe joined Fluxer LiveKit, and published it with `xai_realtime_published: true`.
- Added the first one-turn duplex probe: subscribe to remote LiveKit audio, collect PCM16, send `input_audio_buffer.append`/`commit` into xAI Realtime, stream Grok Voice output deltas directly into Fluxer LiveKit, and drain final playout.
- Verified live remote human-speaker capture and streamed response publishing against hosted Fluxer. The measured turn started assistant audio at `first_audio_seconds: 1.999`; final playout drain was `0.840s`, replacing the earlier full-WAV publish wait.
- Tightened end-of-turn capture defaults after live testing: the loop now waits for `600ms` silence, keeps only `180ms` final silence in the PCM sent to xAI, and requires `750ms` minimum captured audio to avoid false/too-short bursts. Timing now reports both wall-clock `capture_seconds` and `captured_audio_seconds`.
- Tightened realtime answer instructions to default to one short sentence and avoid multiple follow-up questions, reducing generated-audio length.
- Added `--xai-first-audio-timeout` so a provider turn that emits no audio delta fails fast and the room can resume listening instead of waiting for the full response timeout.
- Added first barge-in support: while assistant audio is streaming, a fresh sustained user speech detector clears the LiveKit `AudioSource` queue, aborts further xAI audio deltas, marks the turn as interrupted, and reopens listening. Live smoke verified `interrupted_turn_count: 1` followed by a newly captured response turn.
- Optimized barge-in carryover: the interrupting utterance is now retained as PCM, reported as `barge_in_carryover_pcm_bytes`, and used immediately as the next xAI prompt instead of forcing a fresh post-interrupt capture/repeat.
- Next: live-test carryover timing in the hosted room, then tune echo/noise thresholds if Fluxer speaker playback leaks into the interruption detector.
- Hardened the interrupt path after live testing showed Žofka could keep speaking: interruption now checks between 20ms frames inside large xAI audio deltas and stops/unpublishes the LiveKit local track instead of only clearing the `AudioSource` queue.

### Phase 4 — real-time Žofka loop

- Subscribe to allowed user audio tracks.
- Stream audio to **xAI Realtime** or a selected realtime backend.
- Publish assistant audio back to the Fluxer LiveKit room.
- Add controls: join, leave, mute, listen-only, push-to-talk / wake-word mode.

## Open questions

- Does hosted Fluxer allow bot accounts to join voice with the same gateway flow as users, or does it require additional bot permissions/intents?
- Are Fluxer LiveKit tokens scoped to publish audio for bots, or only for user sessions?
- Should xAI Realtime be direct from the bridge, or should Hermes expose a generic realtime voice engine so Fluxer only handles transport?
- What UX should trigger joining: slash command, Fluxer component button, text command in this dev channel, or explicit config-only channel?
