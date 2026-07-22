# Fully discrete ViT and logic-gate status, 2026-07-15

This report freezes the current method inventory, completed 50k results, probe
evidence, and failure analysis.  It replaces the earlier queue-status notes.
No result below is inferred from a partial run.

## Executive conclusion

The current method family is a **hard-discrete numerical ViT**, not yet a
fully executable network of learned two-input gates.  Its main capacity comes
from dense Wmag7 shift/add projections.  The canonical depth-6, width-192
control reaches 75.30% on the fixed 5,000-example CIFAR-10 validation split;
the highest observed scale row is d12/e192 at 76.10%, from one seed.

Three conclusions are supported by completed experiments:

1. Score-gap selected-V aggregation is effective in its matched d2/e96
   experiment.  Across seeds 42/43/44, final accuracy is 63.66% versus 57.82%
   for uniform selected-V, a paired gain of 5.84 +/- 1.20 percentage points
   with 3/3 wins.
2. The minimal learned logic tree did **not** add hard inference capacity.
   After 50k, every truth table was still projection-A (`0xC`), every root code
   change rate was zero, and learned-hard accuracy equaled forced-A accuracy.
   Its apparent one-layer gain is therefore a training-surrogate effect or
   single-seed noise, not a deployed logic-function gain.
3. Width scaling is broken by the current protocol.  d6/e384 loses 6.70 points
   relative to d6/e192, while d12/e192 gains only 0.80 point.  A frozen
   first-block probe shows that doubling head dimension shifts the selected-V
   weight-1 bucket from 7.54% to 30.34%.  Wmag7 weight relative RMSE actually
   falls from 3.81% to 2.79%; worse aggregate Wmag7 quantization error at the
   final checkpoint therefore cannot explain the width regression.  A matched
   float/Wmag7 ladder is still required to exclude weight quantization as a
   contributor to absolute accuracy loss or to interactions during training.

The remaining hardware work is also substantive: integer exponent
requantization, fixed-width residual alignment, the V denominator, a stable
Top-K comparator network, a packed whole-model executor, and C++/CUDA/RTL
bit-exact validation are not closed.

## Scope and terminology

All new 75.xx% values in this report are accuracies on a deterministic
5,000-example validation subset of the CIFAR-10 training set.  They are not
official CIFAR-10 test accuracy.  The split seed is `20260711`; training seed
is 42 unless a row says otherwise.  The common full-discrete recipe uses
50,000 steps, batch 128, crop/flip augmentation, AdamW, cosine decay from
`3e-4` to `1e-5`, 2,000 warmup steps, and validation every 5,000 steps.

Use these three terms separately:

| Level | Meaning | Current status |
|---|---|---|
| Hard-discrete numerical model | The evaluated forward value is defined by integer codes, bitplanes, hard routing, and quantization, although PyTorch may carry it in floating tensors. | Implemented; 50k accuracy measured. |
| Integer transaction specification | Hardened payloads and slow reference primitives define exact integer operations for selected kernels. | Partial: Wmag LUT accumulation, packed XNOR, RMS ROM, export schema. |
| Complete logic-gate executor | The exported whole model runs without unspecified floating bridges or generic framework operators and matches the checkpoint layer by layer; RTL/PPA are validated. | Not implemented. |

FP32 shadow parameters, STEs, and AdamW during training do not by themselves
invalidate discrete inference.  They become a deployment problem only if the
exported hard payload cannot execute independently.  That independent
whole-model closure is what is currently missing.

## Method inventory

### Score-gap selected-V

Q/K threshold bits produce an integer XNOR-popcount score.  Hard Top-K first
selects K keys.  Only inside that set, the best score is subtracted from each
selected score:

```text
raw_gap = best_selected_score - selected_score
bucket  = clip(raw_gap >> 1, 0, 3)
weight  = 2 ** (3 - bucket) = {8,4,2,1}
```

The selected V numerator is a weighted integer sum and the denominator is the
sum of the selected integer weights.  This preserves more routing confidence
than uniform averaging while retaining shift/add-friendly weights.

### Full-discrete ViT base

- Activations are signed A8 codes with a runtime power-of-two scale.
- Dense patch, QKV, attention-output, FFN, and classifier matrices use a sign
  plus Wmag bitplanes and per-output power-of-two scales.
- Q/K use seven threshold lanes per head feature, XNOR-popcount, and hard K=8.
- V uses the score-gap rule above and rounded integer division.
- The FFN has a hard binary gate, but its gate/up/down maps remain dense
  shift/add projections.
- Every attention and FFN residual is requantized to A8.
- Block normalization is an integer mean-square plus Q0.15 reciprocal-sqrt
  ROM by default.

Wmag7 is the accuracy-first configuration: magnitude planes
`{1,2,4,8,16,32,64}` plus a sign.  Wmag4 is a compression point.  In the
transaction backend, Wmag7 is exactly reconstructed from a low U4 and high U3
ROM access, a fixed shift, an add, and a conditional negate.

### Tested enhancements

| Enhancement | Hard-forward intent | Important training/deployment caveat |
|---|---|---|
| 256-entry group LUT | Per-group A8-to-A8 activation table, with one runtime power-of-two scale per group. | Learns an FP32 shadow table; hard table is rounded. |
| Learned head gap LUT | Per-head monotonic map from gaps 0--63 to `{0,1,2,4,8}`. | Absolute gap address is not normalized by head dimension. |
| Repeated local 3x3 branch | Depthwise W4/A8 local residual stencil in early layers. | Reference/training path still uses framework convolution; packed executor is absent. |
| BitSlice LogicExpert | A8 bitplanes, two learned hard input connections, and a 4-bit truth table per gate, in parallel with the ordinary FFN. | Training uses dense connection softmax/matmul; the branch still ends in a dense W7 projection. |
| Two-bit state-selected FFN | Dynamic/static/script state selects one of four connection/LUT banks. | Adds large shadow connection state and a discontinuous controller. |
| Fixed local logic tree | One fixed 8-to-4-to-2-to-1 tree for every `(channel, bitplane)`, seven two-input LUTs, spatial sharing, no learned routing or root projection. | The hard tables never left identity in the completed run. |

### Logic transaction backend

The backend defines an A8-by-U4 product ROM, Wmag chunk reconstruction,
checked integer accumulation, packed XNOR-popcount, stable score/key ordering
in the newer backend, and a Q0.15 reciprocal-sqrt ROM.  It is a slow CPU
oracle, not a throughput implementation or cycle model.  The complete numeric
contract is in `vit_lgn/full_discrete/LOGIC_BACKEND_CONTRACT.md`.

## Completed 50k results

### Score-gap versus uniform, d2/e96, three seeds

| Variant | Seed 42 | Seed 43 | Seed 44 | Final mean +/- sample std |
|---|---:|---:|---:|---:|
| Uniform selected-V, final A1 | 57.54 | 56.90 | 59.02 | 57.82 +/- 1.09 |
| Score-gap selected-V, final A1 | 64.76 | 62.20 | 64.02 | 63.66 +/- 1.32 |
| Paired gain | +7.22 | +5.30 | +5.00 | **+5.84 +/- 1.20** |

This is the cleanest evidence that score-gap is useful.  It is a matched
selected-V experiment, but it is not the full d6/e192 model and must not be
used to attribute all of the latter model's accuracy to score-gap.

### Scale matrix, Wmag7/A8, seed 42

| Model | Trainable shadow parameters | Best validation | Final validation | Delta from d6/e192 |
|---|---:|---:|---:|---:|
| d6/e192 | 3.56M | 75.34 | 75.30 | 0.00 |
| d12/e192 | 7.10M | 76.30 | 76.10 | +0.80 |
| d6/e384 | 14.20M | 69.16 | 68.60 | -6.70 |
| d12/e384 | 28.36M | 67.92 | 66.92 | -8.38 |

Depth gives a small single-seed improvement; width causes a large regression.
The interaction is also negative: adding the separate depth and width deltas
would predict 69.40%, but d12/e384 reaches 66.92%, another -2.48 points.

An earlier standalone d12/e384 run at commit `2984167` reached 68.06%.  It is
not part of the paired scale matrix, which used commit `47e0250`; the two
numbers are retained separately and are not averaged.

The same early standalone generation also contains a d6/e192 result of 74.68%
at commit `2984167`.  It is retained because its checkpoint is the subject of
the frozen normalization probe, but the newer paired/canonical d6/e192 control
is 75.30% at commits `47e0250`/`4523f31`/`376aba42`.  Cross-generation controls
are not treated as repeat seeds.

Across most mid- and late-run checkpoints the wide rows have higher training loss.
At 50k it is 0.6043 for d12/e192, 0.7493 for d6/e384, and 0.8666 for
d12/e384.  This does not look like ordinary wide-model overfitting; the wide
hard-discrete models are harder to optimize under the frozen numeric and
training protocol.  The first checkpoints are not uniformly ordered, so this
is a late-run trend rather than an all-step statement.

### Enhancement queue, d6/e192, seed 42

| Method | Trainable shadow parameters | Best | Final | Delta from control |
|---|---:|---:|---:|---:|
| Control | 3.56M | 75.34 | 75.30 | 0.00 |
| Group activation LUT, 48 groups | 3.64M | 73.40 | 73.12 | -2.18 |
| Learned monotonic head gap LUT | 3.57M | 74.44 | 74.06 | -1.24 |
| Three repeated depthwise local branches | 3.57M | 75.30 | 75.14 | -0.16 |
| Two 1024-wide LogicExperts | 43.72M | 54.02 | 53.80 | -21.50 |
| Dynamic two-bit state, expert width 512 | 43.72M | 60.68 | 60.68 | -14.62 |
| Script state, expert width 512 | 43.72M | 63.88 | 63.88 | -11.42 |
| Static state, expert width 512 | 43.72M | 62.96 | 62.96 | -12.34 |

None of the tested large learned-connection/state mechanisms improved the
control.  Parameter count is misleading here: most added parameters are
continuous connection/controller shadows around discontinuous hard choices,
not smoothly usable hard capacity.

### Weight width and normalization, d6/e192, seed 42

| Weight / block norm / final norm | Best | Final | Delta from Wmag7 RMS |
|---|---:|---:|---:|
| Wmag7 / RMS LUT / RMS LUT | 75.34 | 75.30 | 0.00 |
| Wmag4 / RMS LUT / RMS LUT | 74.22 | 73.74 | -1.56 |
| Wmag7 / Shift-RMS / Shift-RMS | 71.88 | 71.80 | -3.50 |
| Wmag7 / Shift-RMS / none | 74.60 | 74.60 | -0.70 |
| Wmag7 / RMS LUT / none | 74.74 | 74.74 | -0.56 |
| Wmag7 / requant only / requant only | 70.24 | 70.10 | -5.20 |
| Wmag7 / no norm / no norm | 71.20 | 70.64 | -4.66 |

The useful result is not that normalization can be deleted.  It is that
Shift-RMS in the blocks with no final Shift-RMS is only 0.14 point below RMS
blocks with no final RMS.  A cheap exponent-only block normalizer is plausible;
the final normalization/calibration path is the larger Shift-RMS problem.

### Minimal fixed logic tree, d6/e192, seed 42

| Local tree layers | Trainable shadow parameters | Learned hard | Forced-A | LUT flips | Root code change | Delta from no tree |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3.56M | 75.30 | N/A | N/A | N/A | 0.00 |
| 1 | 3.61M | 75.76 | 75.76 | 0 | 0 | +0.46 |
| 3 | 3.69M | 58.28 | 58.28 | 0 in every branch | 0 in every branch | -17.02 |

Final minimum absolute truth-logit margins were 0.4196 for the one-layer tree
and 0.3608/0.9256/0.7442 for the three-layer branches.  The tables were not
about to cross a hard boundary: the current optimizer and surrogate never
entered a new hard function.

The one-layer +0.46 point result is only 23 examples on a 5,000-example split,
with one seed.  Because forced-A and learned-hard outputs are identical, it
cannot be called a logic-expression gain.  Three repeated soft surrogates
severely damage backbone optimization even though the deployed hard branches
are still wires.

## Probe evidence

### Frozen-checkpoint normalization counterfactual

On the same d6/e192 checkpoint, replacing RMS without retraining gives 74.68%
for RMS, 73.60% for Shift-RMS, and 15.30% for requant-only or identity block
normalization.  From-scratch no-norm training recovers to 70.64%, so the two
experiments answer different questions: the checkpoint cannot tolerate a
post-hoc deletion, while training can partially adapt but still loses 4.66
points.  Full details are in `vit_lgn/full_discrete/NORM_PROBE_RESULTS.md`.

### Width probe on frozen scale checkpoints

This is a bounded diagnostic, not another accuracy run: the first attention
block was evaluated on the same first 16 validation images for the paired
d6/e192 and d6/e384 checkpoints.

| Metric | d6/e192, h6 | d6/e384, h6 |
|---|---:|---:|
| Head dimension / XNOR width | 32 / 224 | 64 / 448 |
| Selected raw-gap mean / median | 2.36 / 2 | 4.34 / 4 |
| Weight bucket 8 | 38.33% | 25.83% |
| Weight bucket 4 | 37.38% | 23.07% |
| Weight bucket 2 | 16.75% | 20.76% |
| Weight bucket 1 | **7.54%** | **30.34%** |
| Mean number of keys tied at the Top-K boundary | 7.30 | 3.08 |
| Queries with more than one boundary-tied key | 96.27% | 80.98% |
| QKV-input A8 saturation | 0.82% | 1.40% |
| Aggregate Wmag7 relative RMSE | 3.81% | 2.79% |

The frozen absolute gap thresholds are not scale invariant.  With h=6 fixed,
doubling embedding width doubles head dimension and XNOR width, approximately
doubles selected raw gaps, and moves four times as many selected values into
the minimum weight bucket.  The older scale commit also used framework
`topk` before the later stable score-descending/key-index-ascending ABI; ties
were extremely common in both rows.

This probe directly supports normalized-gap or constant-head-dimension tests.
It does **not** support explaining the width regression by *worse* aggregate
Wmag7 quantization error: the wider checkpoint has lower final-checkpoint
error.  It does not isolate weight quantization's absolute cost or its training
interactions; that requires the matched ladder described below.

The exact script, command, sample indices, checkpoint hashes, and raw JSON are
stored in `docs/repro/full_discrete_scale_probe_20260715/`.

## Why high-accuracy full logic-gate execution was not achieved

### 1. The accurate capacity is still dense shift/add capacity

A Wmag7 projection is multiplier-free, and every finite-width ROM/adder can be
synthesized to gates, but it is not a sparse learned two-input logic network.
Each output retains dense fan-in and every coefficient stores a sign plus seven
magnitude decisions.  Scaling e192 to e384 approximately quadruples dense
weight count and hardware work.  The 75.30% result therefore demonstrates a
useful hard-discrete arithmetic model, not ConvLogic-style gate scaling.

### 2. Hard connection and truth-table choices are not being optimized

The large LogicExpert/state variants add tens of millions of soft connection
parameters but train hard one-hot choices through mismatched softmax/sigmoid
surrogates.  The minimal tree removes learned routing and dense root
projection, but all hard truth tables remain identity.  Its shared hard loss
surface is flat between table flips and discontinuous at a flip; an STE
gradient is not evidence that flipping a truth entry will reduce hard loss.

The next logic-tree optimizer must first pass a local truth test: for sampled
entries, the sign of its training signal should correlate with the actual loss
change after a real hard bit flip.  If it does not, longer 50k queues are not
meaningful.

### 3. The scaling rules are not invariant to model width

The scale matrix fixes heads=6, qk lanes=7, K=8, `gap_shift=1`, clipping bucket
3, A8 vector-max scales, global gradient clipping, LR, and steps.  Changing
e192 to e384 therefore changes head dimension from 32 to 64 without rescaling
the score-gap map.  The measured bucket shift is already large in block 1.

Other plausible width penalties remain to be measured rather than asserted:
per-vector `amax` scales become more outlier-sensitive, residuals are
requantized after every branch, and four times as many parameters share the
same LR/step/clip budget.  The current results did not log the scale-matrix
gradient clipping coefficient, so clipping is a hypothesis, not a conclusion.

### 4. Dynamic-range control remains necessary

Bounded A8 codes do not make normalization redundant.  Each token also has a
dynamic exponent; dense accumulators and two residual branches must be aligned
and requantized; Q/K thresholds depend on the input scale distribution.  The
no-norm and requant-only losses show that range conditioning is carrying real
optimization and representation work.

### 5. The exported execution chain is incomplete

The ordinary accurate path still uses framework `F.linear` with quantized
weights.  `logic_lut` replaces learned matrix products only in a slow CPU
reference, then converts the exact integer accumulator back to a floating
power-of-two carrier before requantization.  The model also still spells out
scale selection with `log2/pow/round`, residuals as carrier additions, V
normalization as integer division, and Top-K as sorting/selection.

The remaining closure is concrete:

- an integer `(code, exponent)` requantizer with frozen rounding and overflow;
- fixed-width residual exponent alignment and addition;
- a bounded V denominator reciprocal/constant-divider network;
- a stable Top-K comparator network and tie ABI;
- packed executors for the enhancement/tree payloads;
- an independent whole-model payload runner;
- layerwise checkpoint-to-runner equality, then C++/CUDA/RTL and synthesis PPA.

### 6. The accuracy loss is not yet decomposed against a matched float model

There is no same-source, same-initialization, same-data-order 50k ladder from
FP32 to A8, Wmag7, hard Q/K, score-gap V, and binary FFN.  Consequently the
current experiments can rank discrete variants, but cannot state precisely
how many points each discretization step loses relative to a normal ViT.
Older attention-clean 200k test results use a different protocol and cannot
fill this causal gap.

## Recommended next work, in order

1. Build the matched accuracy ladder: float, A8-only, A8+Wmag7, hard Q/K,
   score-gap V, binary FFN.  Freeze initialization and data order.
2. Repair width invariance before any larger model claim:
   - d6/e384/h12 to keep head dimension 32;
   - d6/e384/h6 with head-dimension-normalized gap thresholds;
   - record bucket, tie, activation exponent, saturation/SQNR, and gradient
     clipping statistics.
3. Keep Wmag7 for the accuracy-first path.  Wmag4 costs only 1.56 points and
   remains a later compression target; it is not the dominant failure.
4. Use Shift-RMS in blocks with no final Shift-RMS, then add a small integer or
   power-of-two classifier calibration experiment.
5. Replace current logic-tree STE training before scaling it:
   - exponent-aligned or scale-invariant neighborhood encoding;
   - cross-channel/bitplane fixed mixing;
   - residual XOR/MUX enable rather than unconditional bit replacement;
   - low-margin/annealed categorical-16 gates or explicit hard coordinate
     search;
   - require LUT flips, nonzero root changes, and learned-hard versus forced-A
     separation before calling the mechanism active.
6. Complete the independent integer payload executor before using
   "fully logic-gate executable" as a result claim.

Every new architecture claim should still use 50k paired controls.  Offline
checkpoint probes should decide which hypotheses deserve those expensive
runs; they should not be treated as accuracy evidence.

## Reproducibility and raw evidence

Machine-readable final rows are in:

- `docs/tables/score_gap_50k_results_20260715.csv`
- `docs/tables/full_discrete_50k_results_20260715.csv`
- `docs/tables/full_discrete_scale_history_20260715.csv`
- `docs/tables/full_discrete_scale_probe_20260715.csv`

The main full-discrete result table retains the frozen commit, protocol hash,
result SHA256, and raw server path.  The scale history joins that table by run
name.  The scale probe additionally records its script, sample-index,
checkpoint, protocol, and source hashes.  The score-gap upload was not a clean
Git checkout, so its table records the matrix protocol, per-result hashes, and
an exact hashed source snapshot instead of inventing a commit.  That snapshot
and the six raw result files are in
`docs/repro/score_gap_50k_20260712/`.

`trainable_shadow_parameters` counts FP32/QAT trainable scalars.  It is not a
deployment payload-bit count or a logic-gate count; Wmag4 and Wmag7 can have
the same scalar shape while requiring different payload bits and ROM reads.
Checkpoints and the remaining full run directories stay on the experiment
servers and are intentionally not committed.
