# Hermes Fluxer Plugin

Fluxer support for [Hermes Agent](https://hermes-agent.nousresearch.com/docs), packaged as a standalone community plugin for [Fluxer](https://fluxer.app).

This repository lets a Hermes installation talk to [Fluxer](https://fluxer.app) through Fluxer's bot API: Hermes can receive Fluxer gateway events, decide whether a message should wake the agent, and send replies back to Fluxer channels or DMs. You can install it without waiting for Fluxer support to land in Hermes core.

If you are a human setting this up, start with **Quick start**. If you are handing the repo to an AI agent, point it at [`INSTALL_FOR_AGENTS.md`](INSTALL_FOR_AGENTS.md) first.

## What is Fluxer?

Fluxer is a chat platform with Discord-like concepts: users, guilds or communities, channels, direct messages, message reactions, gateway events, and bot/application tokens. It can be used as a hosted service or as a self-hosted chat surface, depending on your deployment.

For Hermes users, Fluxer is another place where the agent can live. Once configured, Hermes can answer messages in Fluxer, send notifications there, deliver cron-job results to a home channel, and use reactions/buttons for approval flows when the Fluxer deployment supports them.

## What this repo provides

This repo is not Fluxer itself and it is not a full Hermes fork. It is the adapter layer between the two systems. It does not create a Fluxer bot for you, provide a token, or know your Fluxer user/channel IDs; those come from your own Fluxer deployment.

| File | Purpose |
| --- | --- |
| `plugin.yaml` | Hermes plugin manifest. Declares this as a `platform` plugin and lists configuration env vars. |
| `adapter.py` | The Fluxer platform adapter: REST sends, gateway receive loop, message gating, approvals, media handling, backlog recovery, and channel discovery. |
| `__init__.py` | Registration shim loaded by Hermes when the plugin is enabled. |
| `pyproject.toml` | Python package metadata and runtime dependencies. |
| `after-install.md` | Short post-install checklist shown after plugin installation. |
| `AGENTS.md` | Safety and execution contract for AI agents working in this repo. |
| `INSTALL_FOR_AGENTS.md` | Short install/configure/verify runbook for agents helping users set Fluxer up. |
| `tests/` | Source-level package and regression tests for the adapter and manifest. |

## Mental model

Hermes platform adapters do three jobs:

1. **Listen** to an external platform for inbound user activity.
2. **Normalize** that activity into Hermes `MessageEvent` objects.
3. **Send** Hermes responses back through the external platform.

This plugin does that for Fluxer:

```text
Fluxer user
  -> Fluxer Gateway WebSocket
  -> FluxerAdapter
  -> Hermes handle_message(...)
  -> FluxerAdapter
  -> Fluxer REST API
  -> Fluxer channel or DM
```

The adapter is deliberately conservative. Group chats are mention-gated by default, user access is deny-by-default unless configured, and deployment-dependent Fluxer features fall back to plain text/reaction flows where possible.

## Current support

Implemented:

- outbound text sends through Fluxer's bot REST API
- inbound `MESSAGE_CREATE` events through Fluxer Gateway WebSocket
- direct messages, channels, groups, forums, and thread-like channel types where Fluxer exposes them
- replies / referenced message context
- message edits and deletes
- pins, when the Fluxer server supports pin routes
- media and document delivery where Fluxer's API supports uploads or attachment URLs
- reactions for approval and slash-confirm flows
- component buttons for approval flows, with reaction fallback
- channel-directory enumeration for Hermes delivery targets
- home-channel delivery for cron jobs and notifications
- mention-gated group-chat behavior
- reconnect handling, heartbeat tracking, and recent-message backlog recovery
- optional native Fluxer slash-command registration

Deployment-dependent / best-effort:

- native slash commands
- component buttons
- thread behavior
- media upload limits and supported content types
- pin routes
- exact gateway event shapes on self-hosted Fluxer builds

If a deployment does not support native commands or components, Hermes approval prompts still include visible text and reaction fallback paths so the user is not trapped.


## Before you start

You need four things:

1. **Hermes CLI installed** on the machine where the gateway runs.
2. **A Fluxer bot/application token** in `<applicationId>.<secret>` form.
3. **At least one Fluxer user ID** to allow. The plugin is deny-by-default.
4. Optional but useful: **a Fluxer channel or DM ID** for `FLUXER_HOME_CHANNEL`, so Hermes has a default place for notifications and cron-job output.

For self-hosted Fluxer, also know your HTTP base URL and, if gateway discovery is not available, the WebSocket gateway URL.

If you do not know the IDs yet, do not guess. Use your Fluxer UI/admin tooling/API, or temporarily use `FLUXER_ALLOW_ALL_USERS=true` only in a local/dev space long enough to identify the right IDs, then switch back to `FLUXER_ALLOWED_USERS`.

## Safety defaults

The important bit: **Fluxer users are deny-by-default.**

Set one of these, or Hermes will connect but ignore inbound messages from everyone:

```bash
# Recommended: allow only specific Fluxer user IDs
FLUXER_ALLOWED_USERS=fluxer_user_id_1,fluxer_user_id_2

# Development only: allow anyone who can reach the bot
FLUXER_ALLOW_ALL_USERS=true
```

This is intentional. A chat bot connected to an agent can run tools, read configured context, and trigger approval flows. The safe default is silence until the operator explicitly chooses who may talk to it.

Group channels are also quiet by default:

```bash
FLUXER_REQUIRE_MENTION=true
```

That means channel messages need a bot mention or direct-address pattern unless the channel/guild is configured as free-response.

## Quick start

### 1. Install the plugin

Install from GitHub:

```bash
hermes plugins install elkimek/hermes-fluxer-plugin --enable
```

If you are testing an unpublished branch before it is merged, clone that branch directly into Hermes' plugin directory instead:

```bash
git clone --branch <branch-name> \
  https://github.com/elkimek/hermes-fluxer-plugin.git \
  ~/.hermes/plugins/hermes-fluxer-plugin
hermes plugins enable fluxer-platform
```

### 2. Enable the Fluxer platform

The install command enables the plugin package. This setting enables the actual Fluxer platform adapter:

```bash
hermes config set platforms.fluxer.enabled true
```

### 3. Add Fluxer credentials

Put your token in `~/.hermes/.env`. Append to the file; do not replace an existing `.env` wholesale:

```bash
FLUXER_BOT_TOKEN=your_application_id.your_secret
```

The token is sensitive. Do not paste a real token into GitHub issues, logs, screenshots, or chat transcripts.

### 4. Allow at least one user

For normal use:

```bash
FLUXER_ALLOWED_USERS=your_fluxer_user_id
```

For local/dev testing only:

```bash
FLUXER_ALLOW_ALL_USERS=true
```

If neither is set, the adapter will reject inbound user messages and slash commands by design.

### 5. Optional: choose a home channel

A home channel gives Hermes a default Fluxer destination for cron jobs, notifications, and `send_message(target="fluxer", ...)`.

```bash
FLUXER_HOME_CHANNEL=your_default_channel_or_dm_id
FLUXER_HOME_CHANNEL_NAME="Fluxer Home"
```

### 6. Restart Hermes Gateway

```bash
hermes gateway restart
```

### 7. Test outbound delivery

From a Hermes session with the Fluxer platform loaded:

```python
send_message(target="fluxer:<channel_id>", message="Fluxer plugin is alive.")
```

If `FLUXER_HOME_CHANNEL` is configured, you can also use:

```python
send_message(target="fluxer", message="Fluxer home delivery works.")
```

Then test inbound by sending a Fluxer message from an allowed user. In a group/channel, mention the bot unless you configured free-response behavior.

## Hand this repo to an agent

This README is written for humans, but the repo also includes an agent runbook:

- [`AGENTS.md`](AGENTS.md) — execution contract and safety boundaries for AI agents working in this repo.
- [`INSTALL_FOR_AGENTS.md`](INSTALL_FOR_AGENTS.md) — short install/configure/verify checklist for a Hermes agent helping a user set Fluxer up.

Useful instruction to paste to an agent:

```text
Install elkimek/hermes-fluxer-plugin into my Hermes setup. Read INSTALL_FOR_AGENTS.md first. Do not print or overwrite secrets. Do not replace my existing ~/.hermes/.env or config.yaml. Ask me for the Fluxer bot token, allowed user IDs, and optional home channel ID if they are missing. Restart the Hermes gateway only after telling me what will change. Verify with hermes plugins list, hermes config get platforms.fluxer.enabled, and a test send_message call.
```

## Configuration reference

### Required

| Env var | Meaning |
| --- | --- |
| `FLUXER_BOT_TOKEN` | Fluxer bot token in `<applicationId>.<secret>` form. |

### Access control

| Env var | Default | Meaning |
| --- | --- | --- |
| `FLUXER_ALLOWED_USERS` | empty | Comma-separated Fluxer user IDs allowed to talk to the bot. Empty means nobody is allowed unless `FLUXER_ALLOW_ALL_USERS=true`. |
| `FLUXER_ALLOW_ALL_USERS` | `false` | Allow any Fluxer user to talk to the bot. Useful for development, risky for shared servers. |
| `FLUXER_ALLOWED_CHANNELS` | empty | Comma-separated channel IDs where Hermes is allowed to respond. Empty means all channels are eligible, but user allowlist and mention gates still apply. |

Access control applies to normal messages, reactions, component interactions, and native slash-command interactions.

### Hosted and self-hosted Fluxer

The hosted API defaults to:

```text
https://api.fluxer.app/v1
```

For self-hosted Fluxer, set:

```bash
FLUXER_BASE_URL=https://your-fluxer.example
```

The adapter accepts either a plain web origin or an already-scoped API URL:

- `https://your-fluxer.example` -> normalized to `https://your-fluxer.example/api`
- `https://your-fluxer.example/api` -> used as-is
- `https://your-fluxer.example/api/v1` -> used as-is
- `https://api.fluxer.app/v1` -> used as-is

Gateway discovery normally happens through `/gateway/bot`. If your deployment needs an explicit WebSocket URL:

```bash
FLUXER_GATEWAY_URL=wss://your-fluxer.example/gateway
```

### Group-chat response behavior

By default Hermes should not jump into every group conversation. Useful knobs:

| Env var | Default | Meaning |
| --- | --- | --- |
| `FLUXER_REQUIRE_MENTION` | `true` | Require a bot mention/direct address in normal channels. DMs do not need mentions. |
| `FLUXER_STRICT_MENTION` | `false` | Require a fresh mention on every channel message instead of remembering mentioned threads. |
| `FLUXER_FREE_RESPONSE_CHANNELS` | empty | Channels where Hermes may respond without a mention. |
| `FLUXER_MENTION_PATTERNS` | empty | Extra comma-separated regexes that count as bot mentions/direct address patterns. |
| `FLUXER_HOME_GUILD_ID` / `FLUXER_HOME_GUILDS` | empty | One or more trusted guild/community IDs used with `FLUXER_AUTO_FREE_RESPONSE_HOME_GUILD`. |

For a trusted home guild/community where Hermes may respond naturally:

```bash
# one home guild/community
FLUXER_HOME_GUILD_ID=guild_id

# or several home guilds/communities
FLUXER_HOME_GUILDS=guild_id_1,guild_id_2

FLUXER_AUTO_FREE_RESPONSE_HOME_GUILD=true
FLUXER_MENTION_GATED_CHANNELS=channel_id_that_still_requires_mention
```

Use this carefully. It changes group behavior from "answer when called" to "answer naturally" in configured spaces.

### Mentions in outbound messages

The adapter sanitizes broad mentions by default so an agent response does not accidentally ping a whole server.

| Env var | Default | Meaning |
| --- | --- | --- |
| `FLUXER_ALLOW_MENTION_EVERYONE` | `false` | Permit outbound `@everyone` / `@here`. |
| `FLUXER_ALLOW_MENTION_ROLES` | `false` | Permit outbound role mentions. |
| `FLUXER_ALLOW_MENTION_USERS` | `true` | Permit outbound user mentions. |
| `FLUXER_ALLOW_MENTION_REPLIED_USER` | `true` | Permit reply notifications to the replied-to user when Fluxer supports allowed mentions. |

### Delivery verification and backlog recovery

Optional reliability settings:

```bash
FLUXER_DELIVERY_VERIFICATION=true
FLUXER_BACKLOG_RECOVERY=true
FLUXER_BACKLOG_LIMIT=25
FLUXER_BACKLOG_BOOTSTRAP_SECONDS=120
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `FLUXER_DELIVERY_VERIFICATION` | `true` | Read back sent/edited messages where possible to verify delivery. |
| `FLUXER_BACKLOG_RECOVERY` | `true` | Scan recent known-channel messages after startup/reconnect. |
| `FLUXER_BACKLOG_LIMIT` | `25` | Maximum recent messages to scan per known channel. |
| `FLUXER_BACKLOG_BOOTSTRAP_SECONDS` | `120` | Startup lookback window when no previous disconnect time exists. |

Backlog recovery is a safety net for short disconnects. It is not a full historical importer.

### Native command registration

If your Fluxer deployment supports application/slash-command registration:

```bash
FLUXER_REGISTER_NATIVE_COMMANDS=true
FLUXER_APPLICATION_ID=your_application_id
FLUXER_NATIVE_COMMAND_GUILDS=guild_id_1,guild_id_2
```

| Env var | Default | Meaning |
| --- | --- | --- |
| `FLUXER_REGISTER_NATIVE_COMMANDS` | `false` | Register Hermes slash commands with Fluxer on gateway connect. |
| `FLUXER_APPLICATION_ID` | token prefix | Fluxer application ID for native command registration. Must be an ID, not the whole token. |
| `FLUXER_NATIVE_COMMAND_GUILDS` | empty | Comma-separated guild IDs for guild-scoped registration. Empty means global. |

Leave native registration disabled if your Fluxer deployment does not support those routes. Text fallback still works.

## How approvals work

Hermes sometimes needs user approval before running sensitive tools or confirming slash-command actions. This adapter supports several approval surfaces:

1. **Component buttons**, if the Fluxer deployment supports message components.
2. **Reactions**, added by the bot to the approval message.
3. **Visible text fallback**, so the prompt still tells the user what to do if components are unavailable.

Allowed-user checks apply to approval reactions and component clicks. A random server member should not be able to approve another user's agent action unless explicitly allowed.

## Troubleshooting

### Plugin does not appear

```bash
hermes plugins list
```

Expected: `fluxer-platform` appears and is enabled.

Also check:

```bash
hermes config get platforms.fluxer.enabled
```

Expected: `true`.

### Bot connects but never responds

Most common cause: no allowed users are configured, or the message is in a group/channel and does not mention the bot.

Set one:

```bash
FLUXER_ALLOWED_USERS=your_fluxer_user_id
```

Or, only for development:

```bash
FLUXER_ALLOW_ALL_USERS=true
```

Then restart the gateway.

Also check group behavior: in channels, Hermes may require a mention unless the channel/guild is configured for free response.

### Outbound sends work, inbound does not

Check, in order:

1. Gateway restarted after env/config changes.
2. The message author is in `FLUXER_ALLOWED_USERS`, or `FLUXER_ALLOW_ALL_USERS=true` is set.
3. The channel is not excluded by `FLUXER_ALLOWED_CHANNELS`.
4. In group channels, the message mentions the bot or matches `FLUXER_MENTION_PATTERNS`.
5. The Fluxer Gateway WebSocket URL is discoverable through `/gateway/bot`, or `FLUXER_GATEWAY_URL` is set explicitly.

### Self-hosted instance cannot connect

Set both URLs explicitly while debugging:

```bash
FLUXER_BASE_URL=https://your-fluxer.example
FLUXER_GATEWAY_URL=wss://your-fluxer.example/gateway
```

Then check Hermes Gateway logs for Fluxer connection errors.

### Native slash commands do not appear

Native command registration is optional and deployment-dependent.

Check:

```bash
FLUXER_REGISTER_NATIVE_COMMANDS=true
FLUXER_APPLICATION_ID=your_application_id
```

If guild-scoped commands are required:

```bash
FLUXER_NATIVE_COMMAND_GUILDS=guild_id
```

If registration is unsupported by your Fluxer deployment, leave it disabled and use normal text messages / approval fallback text.

### Approval buttons do not appear

Component buttons are deployment-dependent. The adapter falls back to reactions and visible text. If reactions also fail, check whether the bot has permission to add reactions in that channel.

## Development

Install test dependencies in your Hermes/plugin development environment, then run:

```bash
python3 -m pytest -q
python3 -m py_compile adapter.py __init__.py
```

The tests intentionally include source-text assertions for the failure modes found during review:

- slash-confirm prompts must fail closed if Fluxer omits a message ID
- REST error logging must redact token-like data
- deduplication must use ordered eviction
- inbound text messages must enforce `FLUXER_ALLOWED_USERS`
- native slash-command interactions must enforce `FLUXER_ALLOWED_USERS`
- the connect guard must match the documented default base URL behavior

## Security notes

- Treat `FLUXER_BOT_TOKEN` as a secret.
- Prefer `FLUXER_ALLOWED_USERS` over `FLUXER_ALLOW_ALL_USERS` on shared servers.
- Keep `FLUXER_REQUIRE_MENTION=true` unless you intentionally want natural group-chat participation.
- Broad outbound mentions are neutralized by default.
- Approval reactions/buttons are checked against the same user allowlist as messages.
- Backlog recovery only scans recent known channels; it should not be treated as an audit log.

## License

MIT. This adapter is maintained as a standalone community-installable Fluxer integration.
