# Lenses

Failure-mode classes for hypothesis agents. Each round, the orchestrator
instantiates `03-hypothesis` with one lens × one seed. Lenses are grounded
in the empirical literature on silent bugs in ML/scientific code (Tambon et
al. 2023; SANER'24 PyTorch study): most damaging bugs surface as *wrong
outputs*, not crashes, which is why the majority of lenses here target
silent wrongness.

Per-lens budgets should adapt over time: the orchestrator tracks each
lens's confirmed-vs-refuted ratio on this codebase and shifts fanout toward
lenses that pay.

---

## numeric-shape — Numerical & shape correctness

Wrong math that still runs. The flagship lens for research code.

Signatures:
- Reductions/indexing over the wrong axis, especially after a transpose,
  squeeze, or layout change upstream; `dims`/`axis` defaults that assume a
  layout the caller didn't provide.
- Broadcasting that silently "works": `[B, 1] vs [B]` producing `[B, B]`,
  masks broadcast against the wrong dimension.
- Off-by-one in slices, windows, boundaries: `[:n]` vs `[:n+1]`, fencepost
  in span/offset math (token↔char alignment is a classic host).
- dtype/precision: silent float64→float32, integer division, bool→int
  arithmetic, accumulating in low precision.
- NaN/inf swallowed by a downstream `mean`/`nanmean`/`clip`; division by a
  count that can be zero.
- Aggregation weighting: mean-of-means vs pooled mean over ragged batches.

Where it hides: pooling/masking/aggregation utilities, metric
implementations, anything that reshapes between library conventions.

Worked example shape: a `mean` over padded sequences that divides by max
length instead of true lengths — every score is biased toward zero, no
error is ever raised, and downstream AUROC quietly drops a few points.

## data-integrity — Splits, labels, leakage

The experiment-invalidating class.

Signatures:
- Split before/after shuffle inconsistencies; stratification that silently
  falls back to unstratified; seeds not reaching the split.
- Filtering/sorting/dedup applied to features but not labels (or after
  pairing was established) — silent misalignment.
- Train/test contamination: normalization fit on all data, vocabulary or
  thresholds derived from test, cache shared across splits.
- Label handling: implicit class-order assumptions, `[0,1]` vs `[-1,1]`,
  positive-class index conventions differing between train and metric code.

Where it hides: dataset loaders, split/filter/merge utilities, metric glue.

## state-staleness — Caching, mutation, staleness

Signatures:
- Cache keys missing an input that affects the result (config field, layer
  index, model revision) — first call wins forever.
- Mutable default arguments; functions mutating caller-owned
  tensors/frames/dicts in place while appearing pure.
- Module/global state carrying across calls or runs (registries,
  `functools.cache` on methods, class attributes as scratch).
- Files: results written then re-read without invalidation on code/config
  change; partial writes on interrupt read back as complete.

## api-contract — Docs/contract drift

Signatures:
- Docstring or README says X; code does Y (defaults, shapes, units,
  return types). Renamed/reordered params where old call sites or docs
  survive; `**kwargs` swallowing typos of real parameter names.
- Type hints that lie (Optional not handled, narrower return than
  declared).
- Copy-paste drift between sibling implementations that promise the same
  interface (two backends, two probe classes) but diverge in behavior.

Note: the *fix* may be to the doc rather than the code — still a finding,
but say so in `consequence`.

## boundary-degenerate — Edges users actually hit

Signatures: empty batch/dataset, single sample, single class present,
sequence length 1 or 0 after masking, zero-variance features, a batch
smaller than requested pool/topk. Constrained by the usage model: only
report edges a real workflow reaches (small pilot runs, aggressive
filtering, last partial batch) — not adversarial pathologies.

## config-plumbing — Parameters that don't arrive

Signatures:
- A kwarg accepted, stored, and never read; a config field read but not
  applied on some code path; plumbing lost at a call boundary (`layers=`
  accepted but only the first used).
- Silent fallback to defaults on lookup failure (`dict.get(k, default)`
  hiding a typo'd key); precedence bugs between config sources.
- Flags honored in one backend/branch but not its sibling.

Cheap mechanical assist: for a public function's parameters, trace each one
to a use; a parameter with no reachable use is a strong candidate.

## serialization-roundtrip — Save/load asymmetry

Signatures: fields dropped or re-defaulted on load; device/dtype not
restored; version/format skew read without error; path handling breaking
off the author's machine; save-then-load producing an object whose
behavior (not just repr) differs — predictions from a loaded probe
diverging from the in-memory one is the canonical catch.

## resource-environment — The world outside the process

Signatures: device mismatches on multi-device setups; accidental
materialization of the full activation tensor that OOMs at realistic scale
(this is the perf-exception: report when realistic scale *fails*, not when
it is merely slow); file-handle/tmpdir leaks in long jobs; assumptions
(network, HF cache, CUDA) that turn into cryptic failures on the cluster
per the usage model's infra section.

## nondeterminism — Seeds and ordering

Signatures: a seed parameter that doesn't reach all RNG consumers (numpy
vs torch vs python; dataloader workers); determinism promised but
order-dependent iteration (set/dict ordering feeding results); parallel
writes racing to one path; results depending on batch size where the docs
imply invariance (a real finding when the usage model promises
reproducibility).

## silent-failure — Error handling that lies

Signatures: broad `except` returning a plausible default (empty list, 0.0,
None scored downstream); errors logged at debug and execution continuing
with partial data; a warning where the usage model implies users need an
error (e.g., requested layer absent → silently fewer layers); partial
success reported as success (3 of 5 shards loaded, no indication).
