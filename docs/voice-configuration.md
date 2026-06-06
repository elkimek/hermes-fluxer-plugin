# Realtime voice configuration

Realtime voice lets the Fluxer plugin join a configured Fluxer voice room, listen only to configured participant audio, run STT/brain/TTS, and publish the assistant audio back into the room.

This feature is intentionally opt-in. A fresh install will not join voice rooms or listen to anyone until you enable it and scope the allowed users/channels.

## Where to edit it

You can configure realtime voice in either place:

1. **Hermes Dashboard → Config → Fluxer**
2. `~/.hermes/config.yaml` under:

```yaml
platforms:
  fluxer:
    extra:
      voice:
        enabled: true
```

Environment variables still win over `config.yaml`. If a value looks correct in the dashboard but runtime behaves differently, check `~/.hermes/.env` and the gateway service environment for the matching `FLUXER_VOICE_*` variable.

## Minimal setup

```yaml
platforms:
  fluxer:
    enabled: true
    extra:
      voice:
        enabled: true
        auto_join: true
        target_user_ids: "your_fluxer_user_id"
        channel_ids: "your_voice_channel_id"
        guild_ids: "your_guild_id"   # optional for DMs/group calls, recommended for guild voice channels
```

Equivalent environment variables:

```bash
FLUXER_VOICE_ENABLED=true
FLUXER_VOICE_AUTO_JOIN=true
FLUXER_VOICE_TARGET_USER_IDS=your_fluxer_user_id
FLUXER_VOICE_CHANNEL_IDS=your_voice_channel_id
FLUXER_VOICE_GUILD_IDS=your_guild_id
```

Use strings for Fluxer IDs. Many IDs are 64-bit snowflakes and should not be edited as browser/JavaScript numbers.

## Safety model

Realtime voice starts only when all of these are true:

- Fluxer platform is enabled: `platforms.fluxer.enabled=true`
- voice is enabled: `voice.enabled=true` / `FLUXER_VOICE_ENABLED=true`
- auto-join is enabled: `voice.auto_join=true` / `FLUXER_VOICE_AUTO_JOIN=true`
- at least one allowed voice channel is configured
- the target user/channel/guild checks pass

The supervisor is plugin-managed. Do not run `scripts/fluxer_voice_auto_join.py` manually in production; the adapter starts and stops it with the Fluxer gateway connection.

## Configuration reference

### Scope and lifecycle

| Dashboard / YAML key | Env var | Default | What it does | Tune when |
| --- | --- | --- | --- | --- |
| `enabled` | `FLUXER_VOICE_ENABLED` | `false` | Enables the realtime voice bridge. | Always required for voice rooms. |
| `auto_join` | `FLUXER_VOICE_AUTO_JOIN` | `false` | Join automatically when an allowed target user is in an allowed voice channel. | Enable for follow-me behavior; leave off for manual/smoke testing. |
| `target_user_ids` | `FLUXER_VOICE_TARGET_USER_IDS` | empty | Comma-separated Fluxer user IDs whose audio may be captured. | Add only the humans the assistant should listen to. |
| `channel_ids` | `FLUXER_VOICE_CHANNEL_IDS` | empty | Comma-separated Fluxer voice channel IDs the bot may join. | Required for auto-join. Keep narrow. |
| `guild_ids` | `FLUXER_VOICE_GUILD_IDS` | empty | Optional comma-separated guild/community IDs. | Use when the same channel ID shape can appear across scopes, or to fail closed to one guild. |
| `participant_prefix` | `FLUXER_VOICE_PARTICIPANT_PREFIX` | auto | LiveKit participant identity prefix to capture. Usually inferred as `user_<targetUserId>_` for one target user. | Only if your Fluxer/LiveKit identity format differs. |
| `supervisor_disabled` | `FLUXER_VOICE_SUPERVISOR_DISABLED` | `false` | Recursion guard for child scripts. | Normally do not set. Child processes set it internally. |
| `python` | `FLUXER_VOICE_PYTHON` | current Python | Python executable for the child voice process. | Use a venv with realtime deps if Hermes itself runs under a different Python. |

### Brain, STT, and TTS

| Dashboard / YAML key | Env var | Default | What it does | Tune when |
| --- | --- | --- | --- | --- |
| `brain_provider` | `FLUXER_VOICE_BRAIN_PROVIDER` | `hermes` | Chooses response brain: `hermes`, `auto`, `xai-fast`, or `xai`. | Keep `hermes` for the full Hermes/session/Honcho experience; use `auto`/`xai-fast` only as explicit performance trade-offs. |
| `stt_provider` | `FLUXER_VOICE_STT_PROVIDER` | `auto` | STT provider: `auto`, `local`, `groq`, `xai`, or `elevenlabs`. | Switch if language/accent/latency is poor. |
| `stt_model` | `FLUXER_VOICE_STT_MODEL` | provider default | STT model name, e.g. `scribe_v2` for ElevenLabs. | Use smaller/faster models for low-power machines. |
| `elevenlabs_language_code` | `FLUXER_VOICE_ELEVENLABS_LANGUAGE_CODE` | empty | Optional Scribe language code; empty means autodetect. | Set when autodetect hurts accuracy. |
| `tts_voice` | `FLUXER_VOICE_TTS_VOICE` | provider default | Realtime TTS voice name. | Change assistant voice/personality. |
| `context_file` | `FLUXER_VOICE_CONTEXT_FILE` | empty | Optional local text context loaded by fast voice modes. | Use for deployment-local assistant identity or room instructions. Never commit personal context. |
| `session_db` | `FLUXER_VOICE_SESSION_DB` | empty / Hermes default | Optional Hermes session DB for recall in full-brain mode. | Use only on the machine that owns that Hermes state. |

### Full Hermes brain mode

These only matter when the voice loop calls the Hermes API for responses.

| Dashboard / YAML key | Env var | Default | What it does | Tune when |
| --- | --- | --- | --- | --- |
| `hermes_url` | `FLUXER_VOICE_HERMES_URL` | `http://127.0.0.1:8642` | Hermes API base URL. | Change for remote API or non-default port. |
| `hermes_model` | `FLUXER_VOICE_HERMES_MODEL` | `Hermes` | Model label sent to the Hermes API voice route. | Usually leave alone. |
| `hermes_timeout_seconds` | `FLUXER_VOICE_HERMES_TIMEOUT_SECONDS` | `90` | Max wait for one Hermes brain response. | Raise for slower hardware; lower if voice must recover quickly. |
| `hermes_max_tokens` | `FLUXER_VOICE_HERMES_MAX_TOKENS` | `90` | Response length budget. | Lower for snappier voice; raise for longer explanations. |
| `hermes_temperature` | `FLUXER_VOICE_HERMES_TEMPERATURE` | `0.4` | Sampling temperature for brain responses. | Lower for predictable answers; higher for more playful conversation. |
| `hermes_session_id` | `FLUXER_VOICE_HERMES_SESSION_ID` | derived from Fluxer guild/channel/participant | Stable Hermes short-term session ID for persisted full-brain voice turns. | Set only if you need a custom session identity; leave derived for normal room-scoped sessions. |
| `hermes_session_key` | `FLUXER_VOICE_HERMES_SESSION_KEY` | derived from Fluxer guild/channel/participant | Stable Hermes long-term memory scope key sent as `X-Hermes-Session-Key`. | Set only if you need a custom Honcho/memory scope; leave derived for normal room-scoped memory. |

### Capture, VAD, and timing

| Dashboard / YAML key | Env var | Default | What it does | Tune when |
| --- | --- | --- | --- | --- |
| `max_turns` | `FLUXER_VOICE_MAX_TURNS` | script default | Max conversation turns handled by one loop process. | Lower for supervised tests; higher for long sessions. |
| `initial_settle_seconds` | `FLUXER_VOICE_INITIAL_SETTLE_SECONDS` | script default | Wait after joining before listening. | Raise if the first utterance is clipped after join. |
| `sample_rate` | `FLUXER_VOICE_SAMPLE_RATE` | `24000` | PCM sample rate for capture/publish. | Keep at 24000 unless your provider or hardware path needs another rate. |
| `vad.frame_ms` | `FLUXER_VOICE_FRAME_MS` | `20` | Audio frame size for VAD. | Rarely tune. Smaller reacts faster but costs more CPU. |
| `vad.energy_threshold` | `FLUXER_VOICE_ENERGY_THRESHOLD` | `300` | RMS threshold for speech detection. | Raise if background noise/echo triggers false speech; lower if quiet users are missed. |
| `vad.silence_ms` | `FLUXER_VOICE_SILENCE_MS` | `850` | Silence duration that ends a user turn. | Lower for faster replies; raise if it cuts off slow speakers. |
| `vad.end_padding_ms` | `FLUXER_VOICE_END_PADDING_MS` | `180` | Final silence kept in the submitted audio. | Raise if STT misses word endings; lower to shave latency. |
| `vad.min_segment_ms` | `FLUXER_VOICE_MIN_SEGMENT_MS` | `1200` | Minimum speech segment accepted as a turn. | Lower for short acknowledgements; raise to ignore coughs/clicks. |
| `vad.max_segment_seconds` | `FLUXER_VOICE_MAX_SEGMENT_SECONDS` | `9` | Hard cap for one captured user segment. | Raise for long monologues; lower for snappier back-and-forth. |
| `timeouts.capture_seconds` | `FLUXER_VOICE_CAPTURE_TIMEOUT_SECONDS` | `90` | Max wait for user speech before the loop times out waiting. | Lower for unattended rooms; raise for sparse conversations. |
| `timeouts.connect_seconds` | `FLUXER_VOICE_CONNECT_TIMEOUT_SECONDS` | `30` | Max wait for gateway/LiveKit setup. | Raise on slow self-hosted deployments. |
| `timeouts.xai_seconds` | `FLUXER_VOICE_XAI_TIMEOUT_SECONDS` | `45` | Max wait for one realtime audio response. | Lower to fail fast when provider hangs; raise for slow networks. |
| `timeouts.xai_first_audio_seconds` | `FLUXER_VOICE_XAI_FIRST_AUDIO_TIMEOUT_SECONDS` | `12` | Max wait for first audio delta. | Lower if you prefer quick recovery over waiting. |
| `barge_in.disable` | `FLUXER_VOICE_DISABLE_BARGE_IN` | `false` | Disable user interruption while assistant audio is speaking. | Use only for diagnosis; normal voice should leave barge-in enabled. |
| `barge_in.energy_threshold` | `FLUXER_VOICE_BARGE_IN_ENERGY_THRESHOLD` | `700` | RMS threshold for speech that interrupts assistant playback. | Raise if echo/noise stops the assistant; lower if quiet interruptions are missed. |
| `barge_in.min_ms` | `FLUXER_VOICE_BARGE_IN_MIN_MS` | `180` | Sustained voiced duration required before interrupting. | Lower for faster stop; raise to avoid false triggers. |
| `barge_in.capture_timeout_seconds` | `FLUXER_VOICE_BARGE_IN_CAPTURE_TIMEOUT_SECONDS` | `2` | Wait to retain the interrupt utterance diagnostics after stop. | Raise only if you want richer diagnostics/carryover. |
| `barge_in.after_first_audio_only` | `FLUXER_VOICE_BARGE_IN_AFTER_FIRST_AUDIO_ONLY` | `true` | Arms barge-in only after assistant audio starts, avoiding the tail of the user's prompt. | Set false only when testing pre-audio interruption. |
| `timeouts.max_runtime_seconds` | `FLUXER_VOICE_MAX_RUNTIME_SECONDS` | `3600` | Max lifetime for one loop process. | Lower for watchdog-style cycling; raise for long sessions. |
| `timeouts.start_cooldown_seconds` | `FLUXER_VOICE_START_COOLDOWN_SECONDS` | `5` | Cooldown between supervisor starts. | Raise if reconnect loops are too aggressive. |
| `timeouts.stop_timeout_seconds` | `FLUXER_VOICE_STOP_TIMEOUT_SECONDS` | `8` | Wait for graceful child shutdown before force-kill. | Raise only if clean disconnect routinely needs longer. |

### Diagnostics and local paths

| Dashboard / YAML key | Env var | Default | What it does | Tune when |
| --- | --- | --- | --- | --- |
| `turn_log_jsonl` | `FLUXER_VOICE_TURN_LOG_JSONL` | empty / temp path | Optional JSONL log for voice turns and timings. | Enable while tuning latency/STT/VAD. Keep logs local. |
| `context_file` | `FLUXER_VOICE_CONTEXT_FILE` | empty | Local context file for fast modes. | Keep deployment-specific identity/personality here, outside the repo. |
| `python` | `FLUXER_VOICE_PYTHON` | current Python | Child process Python executable. | Point to a venv with LiveKit/audio deps. |

## Hardware and provider tuning

### Low-power CPU or HDD-backed VM

Use faster/smaller providers and shorter response budgets:

```yaml
brain_provider: xai-fast
stt_provider: elevenlabs
hermes_max_tokens: 60
vad:
  silence_ms: 700
  min_segment_ms: 900
timeouts:
  xai_first_audio_seconds: 8
```

If full Hermes mode is slow on your hardware, `auto` or `xai-fast` can reduce latency for casual room talk, but they are no longer the recommended default: fast turns may feel less like the user's real Hermes and are not full persisted Hermes/Honcho session turns.

### Noisy room or speaker echo

Raise the speech threshold and minimum segment length:

```yaml
vad:
  energy_threshold: 450
  min_segment_ms: 1400
  silence_ms: 900
```

### Quiet microphone or soft speaker

Lower the threshold and allow shorter segments:

```yaml
vad:
  energy_threshold: 180
  min_segment_ms: 800
```

### STT misses the end of words

Increase end padding:

```yaml
vad:
  end_padding_ms: 250
```

### Replies feel sluggish

Lower silence detection and output length first:

```yaml
hermes_max_tokens: 60
vad:
  silence_ms: 650
  end_padding_ms: 120
timeouts:
  xai_first_audio_seconds: 8
```

Do not set `silence_ms` too low if users pause naturally mid-sentence; it will split one thought into multiple turns.

## Public repo hygiene

Do not commit:

- real Fluxer bot tokens
- real user/channel/guild IDs in examples or tests
- local `~/.hermes/config.yaml`
- `.env`
- personal assistant context files
- voice turn logs or captured audio

Use placeholders like `your_fluxer_user_id` or inert fake snowflakes such as `1234567890123456789` in public docs/tests.

## Troubleshooting

### Fluxer filter does not appear in the dashboard

Restart the dashboard after updating Hermes core, then reload the page:

```bash
systemctl --user restart hermes-dashboard.service
```

The filter appears only when the runtime config contains `platforms.fluxer.*` fields.

### Voice is enabled but the bot does not join

Check these first:

1. `platforms.fluxer.enabled=true`
2. `voice.enabled=true`
3. `voice.auto_join=true`
4. `voice.channel_ids` is non-empty
5. target user/channel/guild IDs match the live Fluxer room
6. gateway was restarted after changes

### The bot joins but does not hear the user

Check:

- `target_user_ids` matches the actual Fluxer user ID
- `participant_prefix` is correct or unset for auto-detection
- VAD threshold is not too high
- STT credentials/provider are configured

### The bot hears but replies slowly

Check turn timing logs if enabled, then tune in this order:

1. STT provider/model
2. `vad.silence_ms`
3. `hermes_max_tokens`
4. `brain_provider`
5. hardware/API latency

### The bot keeps talking over the user

Raise barge-in reliability by reducing playback echo and tuning VAD:

- use headphones where possible during tests
- raise `vad.energy_threshold` if speaker playback is triggering capture
- raise `vad.min_segment_ms` to ignore short echo bursts
