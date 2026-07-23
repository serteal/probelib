# Pack: ml-research

Per-lens signature extensions for ML/scientific research code — libraries
and experiment code whose outputs feed analyses, plots, and papers.
Activated by the usage model's domain profile (globally or per-module).
Grounded in the silent-bugs-in-DL literature (Tambon et al. 2023;
SANER'24 PyTorch study): most damaging bugs here surface as wrong
outputs, not crashes.

## wrong-math
- Reductions/indexing over the wrong axis, especially after a transpose,
  squeeze, or layout change upstream; `dims`/`axis` defaults assuming a
  layout the caller didn't provide.
- Broadcasting that silently "works": `[B, 1]` vs `[B]` producing
  `[B, B]`; masks broadcast against the wrong dimension.
- dtype/precision: silent float64→float32, bool→int arithmetic,
  low-precision accumulation over long sequences.
- NaN/inf swallowed by a downstream `mean`/`nanmean`/`clip`; division by
  a count that can be zero (empty mask, fully-padded row).
- Pooling over padded sequences dividing by max length instead of true
  lengths; mean-of-means over ragged batches.

## data-alignment
- Split before/after shuffle inconsistencies; stratification silently
  falling back to unstratified; the split not receiving the seed.
- Train/test contamination: normalization or vocabulary fit on all data,
  thresholds derived from test, caches shared across splits.
- Features filtered/sorted without their labels (or vice versa) after
  pairing; class-order and label-convention mismatches (`[0,1]` vs
  `[-1,1]`, positive-class index differing between train and metric code).

## state-staleness
- Cache keys missing a layer index, model revision, tokenizer, or config
  field that changes the result.
- Hook/registry state surviving across models or runs.

## boundary-degenerate
- Single-class labels reaching a metric or stratified split; sequence
  length 0/1 after masking; batch smaller than a pool/top-k parameter.

## config-plumbing
- `layers=[...]` accepted but only the first used; a flag honored in one
  collection backend but not its sibling.

## persistence-roundtrip
- Device/dtype not restored on load; a loaded model/probe whose
  predictions diverge from the in-memory original (the canonical catch).

## resource-environment
- Device mismatches on multi-device setups; accidental materialization
  of a full activation/attention tensor that OOMs at realistic scale;
  assumptions (HF cache, network, CUDA version) that fail on the cluster.

## ordering-nondeterminism
- A seed parameter not reaching every RNG consumer (python vs numpy vs
  torch vs dataloader workers); results depending on batch size where
  the docs imply invariance.

## silent-failure
- A requested layer/feature absent → silently fewer returned; some
  shards/files failing to load with the rest reported as the full set.

## Severity anchors (for `06-severity-report`)
- **critical**: pooled activations averaged with padding included — every
  score from the default path biased, no error, results in flight are
  suspect.
- **high**: stratified split silently unstratified when a class is rare —
  misleading eval for users who passed `stratified=True` on skewed data.
- **medium**: save→load drops a non-default dtype; only users who changed
  it and reload notice, and the divergence is visible when it happens.
- **low**: `__repr__` shows a pre-normalization layer index.
