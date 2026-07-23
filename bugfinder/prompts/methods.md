# Exploration methods (domain-neutral)

A method is a *procedure* for exploring the codebase — how to walk it and
what question to keep asking — not a list of bug types to find. This is
deliberate: bug-type taxonomies hardcode one domain's priors and pressure
agents to produce taxonomy-shaped findings. Procedures are domain-neutral
by construction; the codebase-specific "what" comes from the brief's
focus question, which the round planner derives from the usage model's
invariants, failure signatures, and churn.

The orchestrator tracks per-method confirmed-vs-refuted rates on each
codebase; the planner shifts budget toward methods that pay there.

---

## workflow-trace

Follow one real usage mode (from the usage model) end-to-end: the user's
call, every transformation, every handoff, the final output. At each
handoff ask the only question that matters: *does what flows out match
what the next step assumes flows in?* — shapes, units, ordering,
nullability, ownership (who may mutate), error states.

Picks up: cross-module mismatches, assumptions that hold in one caller's
context but not another's. The single best method for bugs no
file-at-a-time read can see. Planner fit: usage modes not traced
recently; workflows crossing recently-changed boundaries.

## invariant-attack

Take one or two invariants from the usage model and actively try to
construct a realistic input, state, or call sequence that violates them.
Work backwards from "what would have to be true for this to break" to
"can real usage make that true?".

Picks up: silent wrong-results bugs — the invariant *is* the grounded
oracle, so findings arrive pre-grounded for the verifier. Planner fit:
the highest-yield method wherever invariants exist; rotate through
invariants over rounds.

## contract-audit

Enumerate every public promise a module makes — docstrings, type hints,
docs, examples, help text — and verify the implementation delivers each
one, including defaults, units, error behavior, and edge wording. The
delta is the finding (sometimes the fix is doc-side; say so).

Picks up: drift between promise and behavior, sibling implementations
that diverge. Planner fit: modules with rich public surface; after
refactors that touched signatures.

## diff-review

Review the commits since the last round like a hostile reviewer: for each
change, what did it break, what did it forget to update (callers, docs,
serialized formats, sibling branches)? Then variant analysis: for any
mistake fixed in history, hunt the same mistake in code the fix didn't
touch.

Picks up: regressions and incomplete changes — the highest-density bug
habitat there is; variant analysis is the regime where LLM bug-finding
has worked in the wild. Planner fit: always, when there is churn.

## consumer-check

Pick one high-fan-in utility (many callers). Enumerate the callers; for
each, compare what the caller assumes against what the utility actually
guarantees — and the reverse: does the utility handle every input shape
its real callers send?

Picks up: the one caller that predates a behavior change; guarantees
weaker than everyone believes. Planner fit: utilities with many callers,
especially recently-modified ones.

## test-gap-probe

In a high-care module, map which behaviors the tests actually pin down,
and scrutinize what they don't. Optionally write throwaway probe tests in
scratch — properties, edges, round-trips — and see what falls out.
Probe tests are exploration tools, not deliverables; repro discipline
belongs to the verifier.

Picks up: behavior nobody ever asserted, which is where silent wrongness
hides longest. Planner fit: critical-care modules with thin coverage
(the usage model flags these).

## fresh-eyes-run

Follow the README, examples, and documented workflows literally, as a
first-time user on a machine that isn't the author's — no folklore, no
implicit environment. Note every divergence between documentation and
reality and every step that only works by accident.

Picks up: onboarding breakage, doc rot, environment assumptions — the
bugs the owner's friends hit and the owner never does. Planner fit:
after doc or setup changes; periodically for any codebase other people
run.

## boundary-sweep

For one entry point, enumerate the degenerate-but-realistic inputs the
usage model implies — empty, exactly one, zero-length after filtering,
missing optional, last partial chunk — and reason (or execute in
scratch) through each. Realism bound: only edges a real workflow
reaches, not adversarial pathologies.

Picks up: the no-op path that crashes, the partial batch handled wrong.
Planner fit: entry points whose edge behavior no test pins; small
codebases where other methods are exhausted.
