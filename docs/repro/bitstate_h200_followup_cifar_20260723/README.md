# Full-CIFAR H200 follow-up

This artifact extends the seed-0, width-4096 full-CIFAR comparison with the
strong annealing baseline. It uses the same architecture, fixed wiring, 45,000
training images, 5,000-image validation split, official 10,000-image test
split, 30 epochs, seed 0, and H200 NVL hardware as the strong DLGN row. Both
strong baselines use N(0,1) gate logits, Adam at constant learning rate 0.01,
and no custom state/gate regularization. Training source is `5aaee9d`.

## Final test metrics

| method | soft acc | hard acc | acc gap | soft loss | hard loss | loss gap | train time | time to 20% | entropy-unused | activation-inactive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong DLGN | 30.23% | 28.13% | 2.10 pp | 1.9259 | 1.9704 | 0.0445 | 1766.6 s | 60.5 s | 40.24% | 29.41% |
| strong annealing | 29.09% | **28.81%** | **0.28 pp** | 1.9457 | 1.9505 | **0.0048** | **1766.4 s** | 413.3 s | 42.63% | 30.63% |
| progressive scale16 | 27.02% | 26.44% | 0.58 pp | 1.9917 | 2.0050 | 0.0133 | 1994.7 s | 263.9 s | **0.06%** | **3.61%** |
| progressive scale16, width 8192 | 27.93% | 27.88% | **0.05 pp** | 1.9631 | 1.9740 | 0.0110 | 3161.1 s | 521.2 s | **0.03%** | **1.90%** |
| strong Gumbel-ST | 14.99% | 15.71% | 0.72 pp | 2.2547 | 2.2551 | 0.0003 | 1849.7 s | not reached | 79.74% | 22.01% |

Annealing is the strongest final classifier in this seed: it improves hard
accuracy by `0.68 pp` and lowers the gap by `1.82 pp` versus strong DLGN. It is
also `2.37 pp` more accurate and has a `0.30 pp` smaller gap than progressive
scale16. The proposed method therefore cannot claim a seed-0 win over simple
annealing. It still reaches 20% hard accuracy earlier than annealing and has
far fewer entropy-unused and activation-inactive gates, although DLGN reaches
that target fastest.

## Width scaling

| width | gates | predicates | fanout max | hard acc | acc gap | max layer flip | train time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 14,976 | 3,904 | 228 | 26.44% | 0.58 pp | **6.85%** | 1994.7 s |
| 8,192 | 27,264 | 8,000 | 429 | **27.88%** | **0.05 pp** | 9.05% | 3161.1 s |

Doubling state width improves hard test accuracy by `1.44 pp`, lowers the
final accuracy gap by `0.53 pp`, and nearly halves activation-inactive gates.
It costs `82.05%` more trainable gates, `88.16%` higher maximum fanout, and
`58.47%` more training time. It also reaches 20% later and increases peak
internal bit mismatch from `6.85%` to `9.05%`, despite the smaller final
accuracy gap. Width therefore supplies useful redundant Boolean capacity, but
does not eliminate depth-wise mismatch or yield linear accuracy scaling.

The larger model is even more literal-dominated: all-literal selection rises
from `89.28%` to `93.98%`, and nontrivial two-input selection falls from
`7.48%` to `4.51%`. Its two global merge layers copy the old-state input on
`89.56%` and `90.38%` of channels. The scaling gain is consequently best
interpreted as wider Boolean routing redundancy, not broader logic-function
use. A still wider full run is not justified before testing the targeted
global-merge anti-bypass regularizer.

## Structural and depth diagnostics

| method | constant gates | all literals | nontrivial two-input | max layer flip | vote flip |
| --- | ---: | ---: | ---: | ---: | ---: |
| strong DLGN | 14.38% | 28.49% | 57.14% | 19.11% | 10.37% |
| strong annealing | 15.24% | 28.75% | 56.01% | 19.80% | 8.95% |
| progressive scale16 | **3.24%** | 89.28% | 7.48% | **6.85%** | **6.85%** |

Annealing preserves roughly the same broad two-input function mix as DLGN,
whereas progressive hardening commits mostly to identity-like literals. The
annealed network nevertheless reaches a tiny final accuracy gap while its
internal binary mismatch still peaks at `19.80%`. Final accuracy-gap alone can
therefore hide large depth-wise disagreement through downstream cancellation.
Progressive hardening has materially lower internal mismatch, but that benefit
does not translate into the best final discrete accuracy in this comparison.

`anneal_normal_adam_w4096_seed0_summary.json` is the complete run summary.
`anneal_normal_adam_w4096_seed0_gate_function_metrics.json` contains aggregate
and named-layer 16-function histograms.
`anneal_normal_adam_w4096_seed0_posthoc_diagnostics.json` independently reloads
the checkpoint and records the full named-layer histograms, eight-batch depth
diagnostics, inactive ratio, and exact hard-carrier versus integer-bit
verification.

`hard_scale16_w8192_seed0_summary.json` and
`hard_scale16_w8192_seed0_gate_function_metrics.json` contain the corresponding
width-scaling evidence. `width_scaling_results.csv` is the two-row
machine-readable comparison.

The strongest-recipe DLGN/proposed comparison is now complete over seeds 0-2.
Strict low-margin AdamW DLGN/annealing rows are complete on the 4090s; their
H200 seed-0 timing replays and strict seeds 1-2 remain in progress. The first
table is therefore a strongest-baseline comparison; the separate
matched-protocol artifact supplies the one-factor accuracy/gap ablation.
