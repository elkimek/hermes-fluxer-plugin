# Agent instructions for hermes-fluxer-plugin

This is Elkim's standalone Fluxer platform plugin for Hermes. Treat it as a public repo: no private IDs, no local paths, no real tokens in commits, issues, logs, or summaries.

## Read first

1. `README.md` for the human explanation and full config reference.
2. `INSTALL_FOR_AGENTS.md` for the install/configure/verify runbook.
3. `plugin.yaml` for the authoritative env-var surface.
4. `adapter.py` only after you understand the safety defaults.

## Safety contract

- Do not overwrite a user's existing `~/.hermes/.env` or `~/.hermes/config.yaml`.
- Do not print `FLUXER_BOT_TOKEN` or any other secret. Report only whether a key is present.
- Do not enable `FLUXER_ALLOW_ALL_USERS=true` on a shared/live server unless the user explicitly asks for a dev-only permissive test.
- Prefer `FLUXER_ALLOWED_USERS` and `FLUXER_REQUIRE_MENTION=true`.
- Tell the user before restarting a live Hermes gateway.
- If Fluxer IDs are missing, ask for them or explain exactly how to find them; do not invent IDs.

## Safe inspection commands

```bash
git status --short --branch
hermes plugins list
hermes config get platforms.fluxer.enabled
python3 -m pytest -q
python3 -m py_compile adapter.py __init__.py tests/test_plugin_package.py
```

When checking environment files, print only key names, never values.

## Release notes

- Every user-visible plugin update must add or update `CHANGELOG.md` in the same commit.
- Keep changelog entries human-readable: what changed, why it matters, user impact, and verification.
- Bump `pyproject.toml` and `plugin.yaml` versions together when cutting a new plugin release.

## Verification before reporting success

A setup is not done until you have verified:

1. `fluxer-platform` appears in `hermes plugins list` and is enabled.
2. `platforms.fluxer.enabled` is `true`.
3. `FLUXER_BOT_TOKEN` is present in the Hermes environment.
4. Either `FLUXER_ALLOWED_USERS` or, for dev only, `FLUXER_ALLOW_ALL_USERS=true` is configured.
5. The gateway has been restarted after config/env changes.
6. Outbound delivery works with `send_message(target="fluxer:<channel_id>", ...)` or `send_message(target="fluxer", ...)` if `FLUXER_HOME_CHANNEL` is configured.
7. Inbound messages from an allowed user reach Hermes. In group channels, mention the bot unless free-response behavior is configured.
