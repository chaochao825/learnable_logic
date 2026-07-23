# ScaleLogic completion and epoch-scaling audit

This artifact closes the previously incomplete 50k ScaleLogic comparison and
separates three questions that had been conflated: whether more optimization
steps help, whether added capacity helps, and whether the result is fully
deployable without real-valued matrix multiplication.

## Direct answer

There is a training-scalable architecture, but the strongest evidence is not
yet a pure one-bit LGN.

1. **Attention-clean K=8 with augmentation** is the clearest training-length
   result. Validation accuracy rises from `71.34%` at 20k to `79.13%` at 150k
   and the 200k test result is `78.93%`, with identical eval and inference
   predictions. Its q/k/v and output projections, MLP, residual carrier, and
   LayerNorm are still real-valued, so it is a hard-routing hybrid rather than
   a fully discrete network.
2. **Wmag7/A8 Full-Discrete ViT** is the strongest deployment candidate. Its
   hard forward uses bit-plane shift/add projections, A8 codes, XNOR/popcount
   Top-K, integer score-gap aggregation, binary FFN gates, and RMS-LUT
   normalization. The historical d6/e192 run reaches `75.34%` best validation
   accuracy at 50k; the separate depth-only d12/e192 row reaches `76.30%`.
   Those stored metrics used the old dyadic floating carrier. Schema v6 now
   provides a standalone integer executor, so strict deployment claims use a
   fresh `uint8 -> integer logits` replay rather than inheriting those numbers.
3. **ScaleLogic d12/e384/h12** does not yet demonstrate capacity scaling. The
   completed local0 and local4 runs reach only `71.14%` and `71.68%` best
   validation accuracy. They keep head dimension fixed at 32 and therefore
   repair the older Q/K-width confound, but still underperform the smaller
   historical model under the same 50k/128 sample budget.

The current answer is therefore: longer training can improve the hybrid
hard-routing model substantially, and depth scales modestly in the fully
discrete model; width-plus-depth scaling is still optimization limited.

## Strict no-real-value checkpoint replay

The schema-v6 executor has now replayed the complete fixed 5,000-image
validation split for the d6/e192 checkpoint. It obtains `75.30%`, exactly its
stored final accuracy. The dynamic audit covers `14,319,802` Torch operations
and sees zero floating or complex tensors. On an independent 100-image check,
all integer logits exactly match the offline QAT carrier, with maximum absolute
difference zero and no prediction mismatch.

The d12/e384/local4 checkpoint also passes a 100-image fixed-prefix strict
replay at `72.00%`, with zero real tensors across `614,916` audited operations.
Its first 20 integer logit rows exactly match the offline carrier. This prefix
check validates topology support but does not replace the stored 5,000-image
accuracy. Frozen evidence and hashes are in
[`../strict_integer_runtime_20260723/`](../strict_integer_runtime_20260723/README.md).

## Completed strict pair

The d12 local0 and local4 rows use identical source files, CIFAR payload,
split, seed, optimizer, schedule, topology, and 50k-step budget. Only
`local_layers` and `out_dir` differ.

| variant | parameters | best valid | best step | final valid | runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| historical d6/e192/h6 reference | 3.563M | **75.34%** | 40k | **75.30%** | 6,239.2 s |
| historical d12/e192/h6 depth reference | 7.102M | **76.30%** | 40k | **76.10%** | 14,838.2 s |
| d12/e384/h12, local0 | 28.359M | 71.14% | 40k | 70.80% | 20,086.9 s |
| d12/e384/h12, local4 | 28.373M | **71.68%** | 45k | **71.62%** | 24,227.7 s |

The four early depthwise shift-add branches add `0.54 pp` best and `0.82 pp`
final accuracy, but cost `20.62%` more wall time. This is a real but small
local-inductive-bias gain, not a capacity-scaling breakthrough.

Both variants improve through 40-45k and then flatten as cosine LR approaches
`1e-5`. The local4 curve moves from `59.58%` at 5k to `69.60%` at 20k,
`70.96%` at 30k, and `71.68%` at 45k. Additional training is plausible only
if the useful learning-rate window is extended; repeating low-LR tail steps is
not supported by this curve.

## Optimization diagnostics

Early deep-model A8 saturation is severe but trainable. In fresh H200 smokes,
roughly 40% of late-layer codes initially hit an extreme. In the completed
local4 run, final-normalization saturation falls from `10.53%` at 5k to
`1.79%` at 30k and ends at `2.13%`. Saturation is therefore an early
conditioning cost, not the proven final ceiling.

Global gradient clipping remains active throughout the d12 runs. The mean
recorded clip coefficients are `0.291` for local0 and `0.314` for local4,
meaning the nominal optimizer update is usually reduced to about one third.
A fixed global norm of 1.0 is not scale invariant as parameter count grows;
per-layer clipping, adaptive gradient clipping, or a model-size-aware limit is
a higher-priority scale ablation than another logic-tree branch.

## Methods that currently work

- **Redundant multi-bit state:** A8 carriers and Wmag bit planes preserve
  magnitude/count information. The pure one-bit persistent-state LGN remains
  near 28% on CIFAR-10 despite a small soft-hard gap.
- **Hard forward, surrogate backward:** exact hard XNOR routing and quantized
  values are used during QAT; STE is confined to optimization. This prevents a
  final argmax conversion from changing the evaluated model.
- **K=8 hard routing:** K=8 consistently beats K=4. The K=4 attention-clean
  run reaches very high training accuracy but lower test accuracy, indicating
  insufficient fan-in and overfit.
- **Augmentation plus sustained optimization:** augmented K=8 improves from
  `71.34%` at 20k to `79.13%` at 150k. The no-augmentation K=8 run peaks near
  `72%` by 30-50k and gains nothing useful from 200k steps.
- **RMS-LUT range conditioning:** Wmag7/RMS reaches `75.30%`; from-scratch
  shift-RMS, requant-only, and no-norm variants reach `71.80%`, `70.10%`, and
  `70.64%`. Removing range conditioning is not a viable simplification.
- **Fixed head dimension while widening:** increasing heads with embedding
  width keeps each XNOR comparison at 224 encoded bits. This repairs the older
  h6 width confound, although it does not by itself recover small-model
  accuracy.
- **Small spatial bias:** four early shift-add depthwise branches give a
  reproducible paired gain. Repeated learned Boolean 3x3 trees did not: their
  LUTs stayed at projection A and three layers collapsed accuracy.
- **Blockwise replacement and distillation:** exact truth-table refitting is
  useful for small frozen prefixes, while K-expanded distilled students
  improve hard accuracy in 11/12 prior groups. These methods should replace
  one block at a time after an accurate multi-bit teacher is established.

## Methods not supported

- Scaling a pure binary BitState LGN by width lowers the gap but leaves
  accuracy low and most selected gates as literals.
- More epochs without augmentation do not improve attention-clean K=8.
- Width/depth increases under a fixed 50k sample budget do not guarantee a
  gain; d12/e384 remains below d6/e192.
- Gumbel-ST, aggressive entropy pressure, and anti-literal penalties can make
  gates look discrete or structurally complex while classification collapses.
- Wmag4, repeated logic trees, LUT activation replacement, and large logic
  experts all lose accuracy in the current form.

## Cancelled floating-carrier H200 test

Two source-identical runs from `main@325f249` briefly started on H200 NVL GPUs
on 2026-07-23. They were stopped before the first validation point once the
requirement was tightened to prohibit all real-valued inference state. Their
model forward still materialized dyadic scales in floating tensors at
requantization and residual boundaries. They produced no admissible accuracy
result and are excluded from every comparison above.

The replacement gate is schema v6's strict integer executor. It accepts only
`uint8` images, carries only integer codes/exponents, audits every Torch
operator for Boolean/integer dtypes, and fails closed on unsupported topology.
The d6 checkpoint has passed the complete replay described above. Longer or
larger training resumes only under this same deployment gate.

Follow-up: the separate depth-only d12/e192 checkpoint has since completed a
full 5,000-image strict integer replay at `76.10%`, with zero floating tensors
and exact carrier-logit agreement on 20 audited rows. That evidence is frozen
in [`../full_discrete_d12_strict_20260723/`](../full_discrete_d12_strict_20260723/README.md).

## Files

- `completed_50k_summary.csv`: final/best metrics, capacity, saturation, and
  clipping summaries.
- `completed_50k_curves.csv`: all stored 5k checkpoints.
- `attention_clean_augmented_200k_curve.csv`: the established long-training
  hybrid reference.
- `validation.json`: pair assertions and SHA-256 hashes.
- `build_scaling_evidence.py`: deterministic standard-library generator.
- `*_result.json` and `*_protocol.json`: raw compact evidence copied from the
  210 and 34 servers, including the depth-only d12/e192 reference;
  checkpoints remain remote.
