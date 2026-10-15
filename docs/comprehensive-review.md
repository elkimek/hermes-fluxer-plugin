# Plugin routing and comprehensive review

## Scope and delivery boundary

Reviewed inbound admission, text commands, thread/session normalization, typing,
approval UI, attachments, outbound transport, heartbeat/reconnect, optional voice,
and packaging against plugin baseline 7345ab8 and the installed Hermes contracts.
All changes are in the plugin. No Hermes harness edits, live plugin installation,
gateway restart, paid provider calls, or live /stop tests were performed.

## Repaired

- Parent-origin turns and in-thread follow-ups share the existing per-user thread
  session-key shape. Commands do not create threads on their own message ids.
- Parent-channel /stop targets a unique same-user executing auto-thread, including
  a prospective thread before creation. Ambiguous multiple runs are not guessed.
- Typing, messages, uploads and approvals resolve thread destinations consistently;
  cold metadata restores bot-owned thread admission without widening allowlists.
- Missing auto-threads fall back to their recorded parent and subsequent sends
  retain the redirect. Unknown cold untyped containers with failed metadata reads
  are not dispatched into a guessed, different session.
- Thread naming reads the anchor session first, preserves author isolation, and
  retains the human-rename guard. Legacy parent lookup is a fallback only when
  the anchor session is absent.
- Split sends expose the last message id and earlier continuation ids. Final edits
  split instead of silently dropping their tail. Partial failures report source
  prefixes in original-text coordinates, including mentions and fenced code.
- Class-level length budgeting uses 4000 UTF-16 units. Permanent HTTP errors no
  longer claim retryability, and ambiguous read/write timeouts are not blindly
  retried. Missing heartbeat ACKs can now expire the watchdog window.
- Attachments are downloaded after admission. Document bodies use connect-time
  SSRF-safe transport, per-hop validation and a streaming size cap. Manual
  redirects strip authorization on origin changes and never restore it.
- Approval UI implements host policy flags and exact request binding. Unbound
  cards resolve nothing, including on deletion. TTL is checked on consumption,
  channel/message identities are required, and expired/error states do not claim
  success.
- Optional voice helpers use a private package rather than shadowing host tools.
  Tests isolate operator environment files. Private speech WAVs use restrictive
  temporary files with cleanup; read-only SQLite URIs escape special characters;
  multipart filenames are sanitized and the undeclared requests dependency removed.

## Parent-executed final verification

- Real installed Hermes imports, full suite with no excluded files: **579 passed,
  1 skipped**.
- Standalone CI mode with HERMES_AGENT_ROOT=/nonexistent-hermes-root: **532 passed,
  15 skipped**. Host-only integration checks skip when the host is unavailable.
- Ruff across adapter, tests, scripts and private helpers: **all checks passed**.
- All five optional voice script --help commands exit 0 with real host imports.
- Regression guards were exercised against broken implementations during the
  individual repairs. Parent additionally reproduced the source-map sanitizer and
  pre-creation active-run registration failures before fixing them.

## Known limits / remaining audit observations

- A typing indicator creates a prospective thread; an aborted turn can leave an
  empty thread. No live gateway end-to-end /stop or voice-room test was attempted.
- Two identical pending approval commands cannot be bound uniquely through this
  host's prompt contract; those cards fail closed and direct users to text commands.
- Older hosts without the connect-time SSRF helper drop document attachments rather
  than fetch through an unsafe client.
- Cold unknown containers whose metadata lookup fails are logged and dropped; a
  later event can retry classification. This avoids silently creating wrong sessions.
- Per-user thread history is preserved intentionally. Existing host chat-scoped
  in-thread stop policy is unchanged; this patch is not a new authorization model.
- Reconnect backlog still uses process-local dedupe and can replay recent history
  on a cold restart. Persistent cursors were not added in this repair.
- Some older channel/thread/typing caches remain without full TTL/LRU lifecycle
  coverage. Delivery read-back remains diagnostic and adds requests. Connection
  pooling and broader lifecycle cleanup were not redesigned.
- Wheel packaging still contains the three runtime modules, not the optional
  repository voice scripts/private helpers. Optional voice is verified from a
  repository install, not promised as a wheel-installed supervisor.
- LiveKit audio fidelity, VAD quality, provider integration and native application
  commands were not exercised. This deployment exposes text commands, not a native
  slash-command registration API.

These are reviewed fixes with explicit exclusions, not a claim of zero remaining
bugs or a penetration-test certification.
