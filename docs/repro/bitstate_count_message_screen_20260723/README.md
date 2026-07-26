# BitState count-message paired screen

## Decision

`bitstate_count_message_v0` is **not promoted**. Its predeclared selection
metric, mean best validation hard accuracy, is `0.05 pp` lower than the
matched majority control (`20.0167%` versus `20.0667%`). The apparent
`+0.60 pp` mean hard-test difference is retained as evidence but is not used
to select the method. A full-CIFAR launch would therefore be test-set-driven
and is prohibited by the promotion policy.

This result also closes off a repeated direction. Earlier merge-only
nontrivial-gate penalties successfully removed identity routing but sharply
increased hidden-state mismatch and the soft-hard gap. The present
intervention changes the message itself without forcing gate diversity. It
still fails because the trained merge path largely discards that message.

## Matched protocol

All runs use CIFAR-10 with augmentation, 10,000 training images, a fixed
2,000-image validation split, a 2,000-image test subset, eight epochs, four
soft warm-up epochs, progressive hard-ST, width 4096, two local and two
global blocks, eight heads, 64 Q/K bits, Top-K 8, and seeds 0/1/2. The only
declared variable is the global value message:

- control: one majority bit for each selected value channel;
- candidate: 25% of lanes encode integer-count thresholds `{1,3,5,7}` and
  the remaining lanes retain majority messages.

Every run has protocol SHA-256
`d2e6aad30dd77d2aca41f2843578cb9459ec5c436a46ab1651a84c23c0b4d9a7`.
The manifests also prove identical source hashes across all six paired runs.

## Per-seed results

Validation values are from the epoch selected before test evaluation. Test
values are measured after restoring that checkpoint.

| method | seed | selected epoch | val hard acc | test soft acc | test hard acc | test gap | train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority | 0 | 7 | 21.05% | 19.65% | 20.90% | 1.25 pp | 207.97 s |
| majority | 1 | 7 | 19.25% | 20.10% | 19.55% | 0.55 pp | 158.24 s |
| majority | 2 | 7 | 19.90% | 18.45% | 20.55% | 2.10 pp | 208.29 s |
| count message | 0 | 7 | 20.70% | 20.65% | 21.10% | 0.45 pp | 209.96 s |
| count message | 1 | 8 | 19.30% | 18.95% | 21.25% | 2.30 pp | 129.03 s |
| count message | 2 | 7 | 20.05% | 17.90% | 20.45% | 2.55 pp | 129.37 s |

| metric, three-seed mean | majority | count message | candidate - control |
| --- | ---: | ---: | ---: |
| best validation hard accuracy | **20.0667%** | 20.0167% | **-0.05 pp** |
| test soft accuracy | **19.4000%** | 19.1667% | -0.2333 pp |
| test hard accuracy | 20.3333% | **20.9333%** | +0.60 pp |
| test accuracy gap | **1.3000 pp** | 1.7667 pp | +0.4667 pp |
| test soft loss | **2.1840** | 2.1888 | +0.0048 |
| test hard loss | 2.1722 | **2.1650** | -0.0072 |
| test loss gap | **0.0138** | 0.0238 | +0.0100 |
| unused-gate ratio | **0.9504%** | 1.0083% | +0.0579 pp |
| inactive-neuron ratio | **0.0979%** | 0.1491% | +0.0512 pp |
| maximum layer flip ratio | **25.8032%** | 26.3351% | +0.5319 pp |

The wall-clock means are recorded in `screen_results.csv`, but run ordering
and shared-server contention were not randomized, so their difference is not
treated as a method speedup.

All per-epoch values are finite. Four of six runs reach the predeclared 20%
validation target; both seed-1 runs do not. Candidate late-validation drop is
`0.75 pp` on average and at most `1.15 pp`, versus `0.28 pp` on average for
control. This stays inside the 2 pp health limit but provides no evidence that
the candidate converges more cleanly. The eight-epoch budget is a registered
screen, not a claim of full CIFAR convergence; failure on its matched
validation metric is why it is not expanded to that more expensive stage.

## Structural diagnosis

The count intervention is active, rather than an identity/no-op ablation:

- all nine Top-K count levels from 0 through 8 occur in every block;
- count entropy is `3.1251` to `3.1539` bits out of a `log2(9)` maximum;
- non-extreme counts make up `67.45%` to `71.78%` of observations;
- thresholds have distinct activation rates, approximately `81.5%` to
  `85.4%`, `60.6%` to `65.4%`, `43.4%` to `48.4%`, and `25.2%` to `30.3%`;
- count encoding changes `12.30%` of message bits relative to majority.

Despite that information, only `6.6650%` of candidate merge gates depend on
the message input, compared with `6.2988%` for control. Candidate merges are
still the exact identity-state gate in `92.9810%` of lanes. The added count
signal reaches the merge input but is almost entirely rejected there. This
explains why information-rich messages do not produce validation gains and
why another anti-literal penalty would repeat an already failed experiment.

## Capacity and strict deployment

| capacity metric | value |
| --- | ---: |
| input bits per patch | 192 |
| persistent state | 4096 Boolean bits/token |
| tokens | 65 |
| trainable shadow parameters | 247,424 |
| selected two-input gates | 14,976 |
| learned gate depth | 9 |
| deployed depth lower bound, excluding Top-K | 31 |
| maximum fanout | 228 |
| hard payload tensor bits | 36,614,432 |
| fixed XNOR operations/sample | 4,326,400 |
| value-count inputs/sample | 4,259,840 |
| count-threshold outputs/sample | 133,120 |

Each run has a standalone `deployment_payload.pt` on server 236. The
candidate payload is 4,591,229 bytes and contains exactly 3,904 Boolean,
1,138,504 int32, and 18,884 uint8 tensor elements. The strict executor accepts
uint8 images, keeps persistent state as bool, emits int32 class logits, and
uses no real-valued tensor or scalar. Reloaded payload logits exactly match
the checkpoint hard path; operator audits report 205 operations for control
and 219 for count message with zero floating tensors.

The optimization checkpoint is deliberately separate: differentiable
training uses real-valued shadow logits and gradients, while none of those
values are present in the deployable model. Claiming that the optimizer itself
is real-free would be false and incompatible with the differentiable-training
question being studied.

## Reproduction artifacts

- `screen_results.csv`: all six rows, including validation, test, capacity,
  utilization, runtime audit, artifact hash, and source hash fields.
- `aggregate.json`: paired deltas, means, capacity, and the failed promotion
  decision.
- `*/summary.json`: unmodified training summaries.
- `*/run_manifest.json`: protocol, environment, source, result, capacity, and
  strict deployment provenance.
- `*/count_message_diagnostics.json`: count distributions and exact
  truth-table dependency analysis for every global merge block.
- `*/export_payload.json`: serialized payload SHA-256, dtype inventory, source
  checkpoint hash, exporter hash, and exact-logit verification.

Regenerate the aggregate locally with:

```bash
python vit_lgn/bitstate/summarize_count_message_screen.py \
  docs/repro/bitstate_count_message_screen_20260723
```
