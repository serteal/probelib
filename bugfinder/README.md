# bugfinder — multi-agent background bug detection

Prompts and design for a long-running multi-agent pipeline that finds bugs in
private research/engineering codebases. The pipeline is precision-first: a
false positive costs the owner trust and attention; a missed bug costs nothing
today, because the system runs continuously and gets another chance.

This directory contains the *prompts* (the scaffolding harness — cron loop,
agent spawning, sandboxing — is whatever you run them with: Claude Agent SDK,
a Workflow script, a shell loop). Prompts use `{{VARIABLE}}` placeholders that
the orchestrator fills in at spawn time.

## Pipeline

```
             ┌────────────────────────────────────────────────┐
             │ Phase 0 (once + refresh): USAGE MODEL          │
             │  01-usage-model  →  02-usage-model-review      │
             └────────────────────────┬───────────────────────┘
                                      │ usage-model.md
             ┌────────────────────────▼───────────────────────┐
             │ Phase 1: PLAN ROUND (03-round-planner)         │
             │  usage model + ledger + churn + coverage map   │
             │  → N investigation briefs, pairwise-           │
             │    decorrelated (seed × method × question)     │
             └────────────────────────┬───────────────────────┘
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
  Phase 2: HYPOTHESIS fanout (04-hypothesis × N, one brief each)
        │                             │                             │
        └─────────────┬───────────────┴─────────────────────────────┘
                      ▼ findings.json (≤5 each, structured)
             Phase 3: DEDUP + TRIAGE (05-dedup-triage, 1 agent,
                      mechanical fingerprint clustering first)
                      ▼ top-K merged findings
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
  Phase 4: VERIFY (06-verify × K, one finding each, adversarial,
                   executable repro against the real package)
        └─────────────┬─────────────┘
                      ▼ verified reports (confirmed AND refuted)
             Phase 5: CALIBRATE + REPORT (07-severity-report)
                      ▼
             digest for the owner  +  ledger updates
                      │
                      └──► LEDGER (persistent, feeds every phase of
                           every future round; owner feedback lands here)
```

## Changes from the original design, and why

The original sketch (usage model → hypothesis fanout → dedup → verify fanout →
severity sort) is structurally right — it matches what works in practice:
Google's Big Sleep insists on a sandboxed reproduction before reporting,
Anthropic's Claude Code review runs finder agents per issue-class and then a
verification agent that tries to disprove each finding before surfacing it,
and hypothesis-validation pipelines like VulAgent cut FPR ~36% with the same
shape. The changes below are where the sketch would underperform:

1. **The usage model gets a review pass and a care map.** A wrong usage model
   poisons every downstream agent, so a second agent adversarially checks it
   against the code before it is adopted (`02`). Its most load-bearing output
   is the *care map* — per-module care levels including an explicit ignore
   list — and the *invariants* section, which becomes the verification oracle
   ("expected behavior" must be grounded somewhere; this is what verifiers
   cite instead of their own intuition). Refresh it incrementally when git
   churn touches uncharted areas; don't regenerate from scratch each round.

2. **Hypothesis diversity is planned per round, not hardcoded.** Different
   starting files alone give shallow diversity — every agent still
   gravitates to the same salient code and the same salient bug types. A
   *round planner* (`03`) turns the usage model, ledger, git churn, and
   coverage history into N pairwise-decorrelated *investigation briefs*:
   seed (where to start) × method (how to explore, from a domain-neutral
   catalog of procedures) × focus question (what to ask, codebase-specific,
   derived from invariants and failure signatures). Churn seeding matters:
   bugs correlate with recent change, and "variant analysis" from a recent
   diff is the regime where LLM bug-finding has actually worked in the
   wild. See "Focus without a bug taxonomy" below for why briefs replaced
   bug-type lenses.

3. **Findings are structured, capped, and self-refuted at the source.** Most
   FP reduction is cheapest at generation time: every finding must carry
   exact `file:line` evidence, a concrete trigger, a *user-visible
   consequence tied to the usage model*, the *source* of the expected
   behavior, and a note on what the agent already tried in order to refute
   it. No consequence → not a finding. Max 5 findings per agent forces
   internal triage; zero findings is an acceptable output.

4. **Dedup is mechanical first, semantic second — and it also triages.**
   Findings carry a fingerprint (files + symbol + failure mode); the
   orchestrator clusters exact matches in code before the dedup agent merges
   semantic duplicates. Two additions: (a) independent rediscovery count is
   recorded and used as a prior (two agents finding the same bug from
   different seeds is signal); (b) the dedup agent checks the ledger and
   drops findings matching previously *refuted* or *intended* entries —
   without this, a long-running system re-litigates the same false positives
   every round and never converges. It forwards only the top {{K}} to
   verification (verification is the expensive phase) and gives every
   dropped finding a disposition — nothing vanishes silently.

5. **Verification imports the real package; "copy parts of the codebase into
   a script" is a trap.** A copy can diverge from the real code — you end up
   verifying a bug in the copy — and it rots as the repo moves. Repros
   import the actual package. The repro must be *discriminating*: it fails
   now because of this defect and would pass under the correct behavior
   (Meta's ACH makes the same demand of generated tests — a test that can't
   tell buggy from fixed code is worthless). Verifiers start from a REFUTED
   default, re-derive the finding from fresh code reads, check reachability
   from a real entry point, and run the "correct usage" control to make sure
   the repro isn't failing for an unrelated reason. Refuted reports are a
   first-class deliverable — they are what the ledger learns from.

6. **Severity is calibrated with anchors, and severity ≠ confidence ≠
   likelihood.** The final agent scores impact-given-real on a rubric with
   concrete anchor examples per level, reports likelihood (how often real
   usage hits it) and confidence separately, and never blends them into one
   number. It expects a skew toward medium/low — if a round comes back >30%
   critical/high, the prior is miscalibration, not a bug bonanza. It is also
   the last FP gate: it reads each repro and demotes any whose assertion
   merely encodes the verifier's own assumption. Silently-wrong-results
   outranks a loud crash at equal reach: a crash announces itself, a wrong
   number ends up in a paper.

7. **A persistent ledger is what makes "running constantly" work.** Without
   memory, a background system rediscovers the same findings forever. Every
   finding gets a ledger row: fingerprint, status (`open` / `confirmed` /
   `refuted` / `reported` / `fixed` / `intended` / `stale`), commit SHA at
   discovery and last verification, refutation reason, and owner feedback.
   Owner marking a report "intended" or "don't care" is the single
   highest-leverage FPR input the system gets — it flows into the dedup and
   hypothesis prompts as the known-non-bugs digest. Confirmed-open bugs get
   re-verified when the code they touch changes; stale entries expire.

8. **Rounds, not a hot loop.** Run in rounds; after `D` consecutive rounds
   with no new confirmed findings, sleep until new commits arrive, then wake
   and seed the planner from the diff. Track per-method and per-area
   confirmed vs refuted rates over time; the planner shifts budget away
   from whatever only produces noise on this codebase.

9. **Spend the strong model where it pays.** Hypothesis agents parallelize
   well on a cheaper/faster model tier; verification and final calibration
   are the precision bottleneck and deserve the strongest model and the most
   reasoning effort. Dedup clustering is mostly mechanical.

## Focus without a bug taxonomy

Hypothesis agents need two things: *decorrelation* (so N agents don't all
converge on the same salient code) and *focus* (so each goes deep). An
earlier iteration got both from assigned bug-type lenses with per-domain
signature packs — but that hardcodes one domain's failure taxonomy into
supposedly general prompts (every new domain needs a new pack), and a
per-agent bug-type assignment quietly pressures agents to produce
lens-shaped findings: an agent sent hunting "ordering bugs" in a codebase
with none is tempted to stretch. So focus is procedural and generated
instead, along three axes, none of which is a bug-type list:

1. **Seeds** (*where*): entry points, churned areas, coverage gaps,
   high-fan-in utilities, invariants — inventoried fresh each round.
2. **Methods** (*how*, `prompts/methods.md`): exploration procedures —
   trace a workflow end-to-end, attack an invariant, audit a module's
   contracts, review recent diffs, check all callers of a utility, probe
   test gaps, follow the README with fresh eyes, sweep boundaries.
   Procedures are domain-neutral by construction: "does what flows out
   match what the next step assumes?" works identically on tensors and
   on CSV rows.
3. **Focus questions** (*what*): concrete, codebase-specific questions
   the planner derives from the usage model's invariants and failure
   signatures, churn, and hit-rate history. A question can be answered
   "it holds — here's what I checked", which a lens assignment never
   gracefully could; cleared ground is recorded and not re-plowed.

What carries the domain priors instead of packs: the usage model's
invariants and generated failure signatures (per-codebase, vetted by the
review agent), and the `invariant-attack` method, which aims agents at
exactly the silent-wrongness properties worth defending. The planner is
also told to vary the failure classes its questions implicitly point at
(wrong values, stale state, contract drift, edge handling, swallowed
errors) — portfolio balance, never per-agent quotas.

Known trade-off: the planner is a single point of correlation — a dull
planner makes a dull round. Mitigations: hard pairwise-decorrelation
constraints in its prompt, past briefs and coverage in its input so
repetition is visible, and per-method/area hit-rate feedback. If rounds
still homogenize, interleave two planner variants.

## What counts as a bug (scope, in one place)

In scope, in priority order:
1. **Silently wrong results** — code runs, output is wrong (wrong axis
   reduction, leaking split, mask misalignment, stale cache, ignored
   config). Worst class: it corrupts research conclusions without a trace.
2. **Failures on realistic usage** — crashes, hangs, data loss on inputs and
   workflows the usage model says actually happen.
3. **Contract violations** — docstring/README/type hints promise X, code
   does Y.

Out of scope: style, lint, performance (unless it makes results wrong or the
tool unusable), hypothetical adversarial misuse, security posture (private
non-adversarial codebases), missing features, bugs in ignore-listed code.

## Ledger schema

One JSONL row per finding (or SQLite with the same fields):

```json
{
  "id": "bf-2026-0142",
  "fingerprint": {"files": ["probelab/masks.py"], "symbol": "assistant", "failure_mode": "off-by-one-boundary"},
  "title": "assistant() mask includes the final user token when template lacks trailing newline",
  "status": "refuted",
  "severity": null,
  "first_seen": {"round": 12, "commit": "f59f1c3"},
  "last_verified": {"round": 12, "commit": "f59f1c3"},
  "independent_discoveries": 2,
  "refutation_reason": "caller-validates: tokenize_dataset re-aligns mask before use (tokenization.py:88)",
  "owner_feedback": null,
  "report_path": "reports/round-012/bf-2026-0142.md"
}
```

Status lifecycle: `open → confirmed → reported → fixed|intended`, or
`open → refuted`, or `* → stale` (touched code changed; re-verify or expire).

## Per-round orchestration sketch

1. Refresh usage model if churn touched modules it doesn't cover (else skip).
2. Run the round planner with churn, ledger, hit rates, and the coverage
   map → ~8–16 investigation briefs.
3. Run hypothesis fanout (one brief each) → collect findings JSON and
   focus-question answers (the answers go into the coverage map).
4. Mechanical fingerprint clustering → dedup/triage agent → top-K.
5. Verify fanout (one agent per finding, sandboxed, time-budgeted).
6. Calibration agent → digest + ledger updates.
7. Deliver digest only if it contains something new; otherwise log silently.
8. Dry-round counter; sleep on `D` dry rounds until new commits.

## Files

| File | Role | Cardinality |
|---|---|---|
| `prompts/00-common.md` | Shared preamble prepended to every agent prompt | — |
| `prompts/01-usage-model.md` | Build the usage model document | 1, on setup + refresh |
| `prompts/02-usage-model-review.md` | Adversarial check of the usage model | 1, after 01 |
| `prompts/03-round-planner.md` | Turn usage model + ledger + churn + coverage into N decorrelated briefs | 1 per round |
| `prompts/methods.md` | Exploration-method catalog (domain-neutral procedures) consumed by 03/04 | — |
| `prompts/04-hypothesis.md` | Execute one brief: explore, hypothesize, self-refute | N per round |
| `prompts/05-dedup-triage.md` | Merge duplicates, apply ledger, rank, select top-K | 1 per round |
| `prompts/06-verify.md` | Adversarially verify + reproduce one finding | K per round |
| `prompts/07-severity-report.md` | Final gate, calibrated severity, owner digest | 1 per round |

## References

- Google Project Zero — Project Naptime / Big Sleep: human-like workflow
  (browse, hypothesize, run in sandbox, reproduce before reporting); their
  SQLite find came from variant analysis of a recent change.
  https://googleprojectzero.blogspot.com/2024/06/project-naptime.html
- Anthropic — Claude Code review/security-review architecture: per-class
  finder agents + a verification agent that tries to disprove each finding;
  their reported precision hinges on that verify step.
  https://code.claude.com/docs/en/code-review ·
  https://github.com/anthropics/claude-code-security-review
- Anthropic — building effective agents & multi-agent research system:
  orchestrator–worker, self-contained subagent tasks, no cross-talk.
  https://www.anthropic.com/engineering/building-effective-agents ·
  https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them
- VulAgent (arXiv:2509.11523): hypothesis-validation multi-agent pipeline;
  ~36% FPR reduction from explicit trigger-path validation.
- Meta ACH — mutation-guided LLM test generation (arXiv:2501.12862):
  discriminating tests ("would catch the described fault") as the assurance
  primitive.
- Tambon et al., "Silent bugs in deep learning frameworks" (EMSE 2023,
  arXiv:2112.13314) and "Investigating and Detecting Silent Bugs in PyTorch
  Programs" (SANER 2024): most damaging bugs in ML/scientific code are
  silent wrong outputs, not crashes — why the scope ranks silent
  wrongness first and why `invariant-attack` is the flagship method.
- LLM-as-judge calibration surveys (anchored rubrics, position-bias
  mitigation, drift): the basis for the anchored severity rubric and the
  pairwise consistency pass in 06.
