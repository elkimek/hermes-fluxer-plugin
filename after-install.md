# Fluxer plugin installed

Fluxer support is installed, but it still needs a bot token and an explicit user allowlist before Hermes will answer messages.

## 1. Enable the platform

```bash
hermes config set platforms.fluxer.enabled true
```

## 2. Add your bot token

Append this to `~/.hermes/.env`. Do not replace an existing `.env` file wholesale:

```bash
FLUXER_BOT_TOKEN=your_application_id.your_secret
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

## 5. Restart and test

```bash
hermes gateway restart
```

Test delivery from Hermes:

```python
send_message(target="fluxer:<channel_id>", message="Fluxer plugin is alive.")
```

For the full explanation of Fluxer, Hermes wiring, safety defaults, group-chat behavior, and troubleshooting, read `README.md`.
