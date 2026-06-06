# Install this plugin for a Hermes user

Use this when a user hands you `elkimek/hermes-fluxer-plugin` and asks you to make Fluxer work with Hermes.

## Inputs you need

Ask the user for any missing values:

- Fluxer bot token: `FLUXER_BOT_TOKEN` in `<applicationId>.<secret>` form.
- Allowed Fluxer user ID(s): `FLUXER_ALLOWED_USERS`.
- Optional default channel/DM ID: `FLUXER_HOME_CHANNEL`.
- Optional realtime voice auto-join: `FLUXER_VOICE_ENABLED`, `FLUXER_VOICE_AUTO_JOIN`, `FLUXER_VOICE_TARGET_USER_IDS`, and `FLUXER_VOICE_CHANNEL_IDS`.
- For self-hosted Fluxer only: `FLUXER_BASE_URL` and optionally `FLUXER_GATEWAY_URL`.

Do not guess these values. Do not print the token.

## Install

```bash
hermes plugins install elkimek/hermes-fluxer-plugin --enable
hermes config set platforms.fluxer.enabled true
```

If the plugin is already installed, inspect before changing anything:

```bash
hermes plugins list
hermes config get platforms.fluxer.enabled
```

## Configure environment

Append missing keys to `~/.hermes/.env`. Do not replace the file.

Minimum live configuration:

```bash
FLUXER_BOT_TOKEN=<fluxer-bot-token>
FLUXER_ALLOWED_USERS=your_fluxer_user_id
```

Optional default destination:

```bash
FLUXER_HOME_CHANNEL=your_default_channel_or_dm_id
FLUXER_HOME_CHANNEL_NAME="Fluxer Home"
```

Self-hosted Fluxer example:

```bash
FLUXER_BASE_URL=https://your-fluxer.example
FLUXER_GATEWAY_URL=wss://your-fluxer.example/gateway
```

Development-only permissive mode:

```bash
FLUXER_ALLOW_ALL_USERS=true
```

Use that only in a private/dev space, then switch back to `FLUXER_ALLOWED_USERS`.

## Optional realtime voice

Realtime voice is disabled by default and should be enabled only when the user explicitly wants it. Do **not** hand-start `scripts/fluxer_voice_auto_join.py` for production installs; the Fluxer adapter manages the supervisor after gateway connect.

Minimum auto-join env:

```bash
FLUXER_VOICE_ENABLED=true
FLUXER_VOICE_AUTO_JOIN=true
FLUXER_VOICE_TARGET_USER_IDS=your_fluxer_user_id
FLUXER_VOICE_CHANNEL_IDS=your_voice_channel_id
```

Equivalent YAML can be placed under `platforms.fluxer.extra.voice` / `fluxer.voice` depending on the user's config shape; see `README.md` for the complete voice matrix. Keep deployment-local assistant context outside the repo and point to it with `FLUXER_VOICE_CONTEXT_FILE` only if needed.

## Restart

Tell the user before restarting a live gateway:

```bash
hermes gateway restart
```

## Verify

```bash
hermes plugins list
hermes config get platforms.fluxer.enabled
```

Expected:

- `fluxer-platform` is present and enabled.
- `platforms.fluxer.enabled` is `true`.

From a Hermes session, test outbound delivery:

```python
send_message(target="fluxer:<channel_id>", message="Fluxer plugin is alive.")
```

If `FLUXER_HOME_CHANNEL` is configured:

```python
send_message(target="fluxer", message="Fluxer home delivery works.")
```

Then test inbound by sending a Fluxer message from an allowed user. In a group/channel, mention the bot unless `FLUXER_FREE_RESPONSE_CHANNELS` or home-guild free response is configured.

## Troubleshooting order

1. Plugin installed and enabled?
2. `platforms.fluxer.enabled=true`?
3. Gateway restarted after env/config edits?
4. Token present in Hermes runtime? Do not print it.
5. Allowed user configured?
6. Channel allowed by `FLUXER_ALLOWED_CHANNELS`, if that var is set?
7. Group message mentions the bot, unless free-response behavior is configured?
8. For self-hosted deployments, are `FLUXER_BASE_URL` and `FLUXER_GATEWAY_URL` correct?

Report exactly what you verified and what remains missing.
