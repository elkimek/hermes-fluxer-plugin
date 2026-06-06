# Fluxer plugin installed

Fluxer support is installed, but it still needs a bot token and an explicit user allowlist before Hermes will answer messages.

## 1. Enable the platform

```bash
hermes config set platforms.fluxer.enabled true
```

## 2. Add your bot token

Append this to `~/.hermes/.env`. Do not replace an existing `.env` file wholesale:

```bash
FLUXER_BOT_TOKEN=<fluxer-bot-token>
```

Keep the real token private.

## 3. Allow users

The adapter is deny-by-default. If you skip this, Hermes can connect to Fluxer but will ignore inbound messages.

Recommended:

```bash
FLUXER_ALLOWED_USERS=your_fluxer_user_id
```

Development only:

```bash
FLUXER_ALLOW_ALL_USERS=true
```

## 4. Optional home channel

For cron jobs and default Fluxer notifications:

```bash
FLUXER_HOME_CHANNEL=your_default_channel_or_dm_id
FLUXER_HOME_CHANNEL_NAME="Fluxer Home"
```

## 5. Optional realtime voice auto-join

Realtime voice is disabled by default. To let the plugin-managed supervisor automatically join specific voice rooms after the Fluxer gateway connects, add:

```bash
FLUXER_VOICE_ENABLED=true
FLUXER_VOICE_AUTO_JOIN=true
FLUXER_VOICE_TARGET_USER_IDS=your_fluxer_user_id
FLUXER_VOICE_CHANNEL_IDS=your_voice_channel_id
```

Or configure the equivalent `fluxer.voice` block in `~/.hermes/config.yaml`. Auto-join refuses to arm unless both `FLUXER_VOICE_TARGET_USER_IDS` and `FLUXER_VOICE_CHANNEL_IDS` are set, so an enabled plugin cannot silently listen to arbitrary users. Keep any deployment-local assistant/personality context outside the repo and point to it with `FLUXER_VOICE_CONTEXT_FILE` only if needed. `FLUXER_VOICE_SUPERVISOR_DISABLED` is an internal child-process recursion guard and should normally be left empty/false.

## 6. Restart and test

```bash
hermes gateway restart
```

Test delivery from Hermes:

```python
send_message(target="fluxer:<channel_id>", message="Fluxer plugin is alive.")
```

For the full explanation of Fluxer, Hermes wiring, safety defaults, group-chat behavior, and troubleshooting, read `README.md`.
