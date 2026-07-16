# Full-discrete ViT / ScaleLogic audit

Date: 2026-07-16

Source: `/data2/wangmeiqi/learnable_logic_hadamard_mixer_20260716` on server 34
Curated destination: `chaochao825/learnable_logic`

## Executive judgment

This branch is worth continuing. Its durable value is the explicit bridge from
trainable fake quantization to a hardened integer/bit-plane inference payload:
signed bit-plane weights, A8 activation codes, exact product-LUT transactions,
packed XNOR/popcount routing, deterministic Top-K, alternative normalization
contracts, and export-time removal of training-only state.

The evidence does **not** yet show that Hadamard or LHVM is the winning global
mixer. The only imported Hadamard/LHVM measurements are 1k-step smoke runs, and
all are below the matched attention smoke control. The active 50k ScaleLogic
run is also not a Hadamard-global model: it uses attention globally and
depthwise shift-add in the first four blocks.

The most defensible main line is therefore:

1. preserve and test the full-discrete backend;
2. finish paired ScaleLogic accuracy experiments;
3. complete the exponent-only integer requantization boundary;
4. compare attention, Hadamard/LHVM, and frequency-domain baselines under the
   same data, model size, step budget, seed set, and deployment accounting.

## Provenance and claim boundary

The directory mixes established ingredients with project-specific integration
and experiments. The following classification describes provenance at the
method-family level; it is not a novelty or authorship claim.

### Established/reference ingredients

- Transformer attention, residual blocks, and RMS-style normalization
- signed integer quantization, bit planes, STE/QAT, and power-of-two scales
- XNOR/popcount similarity, Top-K routing, lookup tables, and Hadamard
  transforms
- depthwise/local spatial mixing and Boolean LUT trees

### Project-specific implementation and attempts in this snapshot

- the Wmag7/A8 ViT integration and its exact A8-by-U4 transaction reference
- Wmag7 reconstruction from low U4 and high U3 magnitude chunks
- a stable hard-routing ABI shared by QAT, evaluation, export, and logic-LUT
  execution
- Q0.15 RMS-LUT, Shift-RMS, requant-only, and identity/no-norm controls in one
  training/evaluation protocol
- deployment payload export without FP32 shadows, optimizer state, or
  surrogate-gradient-only objects
- local depthwise shift-add, Hadamard/LHVM, expert/LUT/spatial, and shared
  logic-tree variants
- paired ablations and source/protocol hashing

The repository snapshot alone is insufficient to claim that every
project-specific combination is original research. A publication should add a
formal related-work search and cite the source of every reused component.

## Current method

### Numeric path

- Activations are signed A8 codes with a signed power-of-two exponent.
- Accuracy-first weights use a sign and seven magnitude planes
  `{64,32,16,8,4,2,1}`, representing the range `[-127,127]`.
- The compression point uses four magnitude planes `{8,4,2,1}`.
- The exact product reference addresses a 4096-entry A8-by-U4 table. Wmag7 is
  reconstructed as `low4_product + (high3_product << 4)` before sign handling.
- Integer products accumulate exactly in int64 in the Python oracle; the
  backend contract derives narrower hardware widths from fan-in.

### Routing and value path

- Q/K become threshold bits and are compared using packed XNOR/popcount.
- Hard Top-K sorts by score descending and then key index ascending, which
  freezes tie behavior across QAT, evaluation, and export.
- Selected values use score-gap weights `{8,4,2,1}`.
- FFN hidden controls are binary MUX/AND decisions over quantized values.

### Normalization

- `rms_lut`: Q0.15 reciprocal-square-root ROM over the clamped A8 mean square.
- `shift_rms`: exact sum-of-squares plus constant comparisons and an exponent
  rewrite; no reciprocal ROM or elementwise normalization multiply.
- `requant`: activation range repair only.
- `none`: strict control without block normalization.

The experiments show that block range conditioning is necessary. Removing
block normalization collapses a frozen checkpoint, and from-scratch
requant-only/no-norm controls remain materially below the RMS baseline.

### Optional mixer and local branches

The package includes attention, Hadamard/LHVM variants, depthwise shift-add,
spatial/LUT/expert modules, and a shared 3x3 Boolean logic tree. These are
alternatives around the full-discrete carrier, not equally validated methods.

The shared logic tree has a fixed 8-to-4-to-2-to-1 topology for each
channel/bitplane. Its hard LUTs initialize to projection A (`0xC`). In the
completed paired experiment, all deployed LUTs stayed at `0xC` and the hard
root never changed, so the apparent one-layer gain cannot be attributed to a
new deployed Boolean expression.

## Evidence audit

Evidence classes in
[`../tables/full_discrete_results_20260716.csv`](../tables/full_discrete_results_20260716.csv):

- `source_document`: values recorded in the imported source README or norm
  probe. The original per-run result files were not present in the 34-server
  snapshot, so these are author-recorded rather than independently reparsed.
- `result_json`: values parsed from small smoke `result.json` files copied into
  `docs/evidence/full_discrete_smoke/` with their matching protocols.
- `checkpoint_history`: values read from the active checkpoint's serialized
  history. They are intermediate, not final.
- `test_execution`: tests rerun directly against the imported source.

### Completed 50k findings

- Wmag7 plus Q15 RMS reaches `75.30%` validation accuracy.
- Shift-RMS in both block and final positions reaches `71.80%`.
- Shift-RMS blocks with no final norm reach `74.60%`.
- RMS blocks with no final norm reach `74.74%`.
- Requant-only reaches `70.10%`; no norm reaches `70.64%`.
- Logic-tree local layers 0/1/3 reach `75.30%`, `75.76%`, and `58.28%`.
  The one-layer delta is only `+0.46` percentage points for one seed, while
  every hard LUT and root output remains unchanged.

### Smoke findings

At 1k steps on the d6/e192 smoke protocol:

- attention local-6 control: `49.30%`
- LHVM parallel: `47.42%`
- LHVM hybrid: `31.92%`
- LHVM: `26.66%`

The d12/e384 ScaleLogic smoke reaches `48.86%`. These runs are useful for code
and optimization-path validation only; they are not final model comparisons.

### Active 50k ScaleLogic run

Protocol SHA256:
`fe9b968f3be661a3ed8cbf77649978774ba525afddf725fe79c127523ec37198`

Configuration:

- CIFAR-10, seed 42, 50k planned steps
- d12/e384/h12, Top-K 8, Wmag7/A8, seven Q/K lanes
- Q15 RMS-LUT block/final normalization
- global mixer: attention
- first four local blocks: depthwise shift-add

Checkpoint history captured during this audit:

| Step | Validation accuracy | Train loss | Peak GiB | Seconds/step |
|---:|---:|---:|---:|---:|
| 5,000 | 0.5958 | 1.4691 | 11.7573 | 0.4120 |
| 10,000 | 0.6664 | 1.1960 | 11.7559 | 0.5500 |
| 15,000 | 0.6830 | 1.0756 | 11.7559 | 0.7802 |
| 20,000 | 0.6960 | 1.0678 | 11.7559 | 0.4145 |

The run was still active when this snapshot was prepared. No final-accuracy
claim should be made from these rows. The extracted lightweight history is in
`docs/evidence/scalelogic_d12e384_h12_local4_seed42_50k.checkpoint_history.json`;
the large checkpoint remains on server 34.

## Verification status

- 116 CPU tests passed directly on server 34.
- The test suite covers core full-discrete modules, enhancements, exporter,
  transaction backend, and protocol hashing.
- Before publication, the 11 protocol-tracked core files matched the active
  protocol byte for byte. The GitHub publish tree then normalizes CRLF line
  endings to LF and uses exactly one terminal newline; applying the same text
  normalization to the source gives an exact source-to-publish comparison. The
  resulting hashes are recorded in
  `docs/protocols/full_discrete_publish_source_hashes_20260716.json`.
- One warning remains in `test_enhancements_expert.py`: a tensor requiring
  gradients is converted to a Python float in a test assertion.
- The source uses FP32 shadow parameters, AdamW, and STE during training by
  design. Export removes those training objects.

This remains a transaction-level reference, not finished RTL. Missing pieces
include:

- an end-to-end integer exponent-only accumulator/residual-to-A8 requantizer
- packed high-throughput kernels or a cycle-accurate executor
- frozen fixed widths and residual/exponent alignment across the whole model
- synthesis, timing, area, energy, and memory-traffic measurements

## Relationship to frequency-domain work

`FFTNet/KV_FFT`, `Freq_KV`, and related frequency-compression directories are
comparison families. They should be compared with this branch, not silently
merged into its evidence. A fair comparison needs a shared table containing:

- task, split, seed, and training budget
- model dimensions and parameter count
- token/KV compression ratio and retained spectrum
- hard/deployable accuracy
- multiply/add/LUT/popcount/shift counts
- state and activation memory
- measured latency/energy or synthesis estimates

Without that protocol, frequency compression and logic discretization answer
different questions and their headline accuracies are not directly comparable.

## Continue, hold, or stop

### Continue now

1. Finish the paired d12/e384 ScaleLogic `local_layers=4` versus
   `local_layers=0` 50k runs without changing their frozen source.
2. Repeat the winning pair for at least three seeds and report mean/std.
3. Implement and exhaustively test the final integer exponent-only
   requantizer, then compare fake-quant, transaction, and exported paths layer
   by layer.
4. Persist stdout/JSONL for every long run. The active run stored protocol and
   checkpoint history but no durable standalone log.
5. Run attention, Hadamard/LHVM, and frequency baselines under one matched
   protocol only after the above deployment path is stable.

### Hold until matched evidence exists

- claims that Hadamard/LHVM improves accuracy or efficiency
- Wmag4 as the default accuracy claim
- packed-kernel or RTL performance claims
- conclusions from 1k smoke runs

### Deprioritize in the current form

- three repeated shared-logic-tree layers
- interpreting the one-layer logic-tree delta as a deployed logic gain
- requant-only or no-norm as the primary model

## Repository boundary

Committed:

- source, launchers, tests, backend contract, probe notes, normalized evidence,
  selected small smoke JSON evidence, and active-run protocol/history metadata

Kept on server 34:

- CIFAR-10 data
- `runs/`
- optimizer/checkpoint binaries
- raw diagnostics and caches

This separation keeps the GitHub repository auditable without publishing
large, mutable training state.
