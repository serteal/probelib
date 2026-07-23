# Pack: engineering

Per-lens signature extensions for engineering codebases — CLIs, tools,
services, automation, infra glue — where the cost of a bug is a user
(the owner or a friend) hitting an error, losing data, or trusting wrong
output. Activated by the usage model's domain profile (globally or
per-module).

## wrong-math
- Time arithmetic: epoch seconds vs milliseconds, naive vs aware
  datetimes, DST boundaries, duration math across midnight.
- Size/rate parsing and formatting: KiB vs KB, bits vs bytes; percent vs
  fraction in thresholds.
- Accounting-style totals that double-count retried or partially
  processed items.

## data-alignment
- Pagination cursors skipping/duplicating across page boundaries;
  resumable jobs re-processing the boundary item.
- CSV/TSV column drift between writer and reader; header assumed but
  absent; sort applied for display but relied on by logic.
- Records matched by un-normalized keys (case, trailing slash, unicode).

## state-staleness
- Caches (files, memo dicts, HTTP caches) not invalidated when config or
  inputs change; long-lived daemon/service state leaking between
  requests; stale lockfiles blocking or, worse, being ignored.

## api-contract
- `--help`/README documenting flags the parser doesn't accept (or
  defaults it doesn't use); exit codes not reflecting failure; JSON
  output shape drifting from what downstream scripts parse.

## boundary-degenerate
- Empty file, missing directory, zero search results, no matching rows —
  paths that should no-op cleanly but crash or, worse, act on everything.

## config-plumbing
- Precedence between flag, env var, and config file differing from docs;
  a typo'd config key silently falling back to default; an option applied
  in one subcommand but not another.

## persistence-roundtrip
- Config/state-file version skew read without migration or error; CRLF
  and encoding issues; `~` and relative-path assumptions; atomic-write
  discipline (temp+rename) missing where interrupts are realistic.

## resource-environment
- Cross-platform paths and case-sensitivity; missing external binaries
  producing cryptic errors; network timeouts unhandled on the paths the
  usage model says run unattended (cron, CI).

## ordering-nondeterminism
- Two concurrent invocations racing on a shared file/directory; retry of
  a non-idempotent step (send, charge, append) duplicating effects;
  output ordering depending on filesystem/dict iteration where users
  diff or parse it.

## silent-failure
- Subprocess return codes unchecked; HTTP non-2xx swallowed; `rm`/copy
  loops continuing past failures and reporting success; errors printed
  to a log nobody watches while the exit code stays 0.

## Severity anchors (for `06-severity-report`)
- **critical**: a sync/cleanup tool's matcher mis-normalizes paths and
  deletes files outside the intended set — silent data loss in the
  default workflow.
- **high**: the retry wrapper re-runs a non-idempotent step, so a
  routine flake duplicates records; common path, hours to notice and
  clean up.
- **medium**: exit code 0 on partial failure — scripts that check it
  proceed, but the log states the errors plainly.
- **low**: `--verbose` documented but ignored by one subcommand.
