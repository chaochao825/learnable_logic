# Integer count-threshold global-token screen

This bounded CIFAR-10 experiment isolates the global-token information
bottleneck in the persistent Boolean-state model. The control reduces every
state channel over the 64 patch tokens with one fixed majority comparison. The
candidate learns one integer occurrence-count threshold per channel. Both
deployment paths use only popcount, integer comparison, Boolean gates,
XNOR-popcount Top-K, majority aggregation, and integer GroupSum.

## Protocol

- Source: commit `3b941c5`.
- Hardware: two RTX 4090 GPUs on server 236, one method per GPU.
- Data: CIFAR-10, deterministic 2,000-example validation split, bounded 10,000
  training examples, and a bounded 2,000-example test subset.
- Model: width 4096, two local blocks, two global blocks, eight heads, 64 Q/K
  bits, Top-K 8, 64 votes per class, seed 0, identical fixed wiring.
- Training: eight epochs, four soft warm-up epochs, then exact hard-forward
  straight-through training with a 16x logit hardening step.
- Selection: best hard validation checkpoint, followed by one test evaluation.

## Results

| global token | soft acc | hard acc | acc gap | loss gap | first 20% epoch | entropy-unused | final vote flip | train time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fixed majority | 19.65% | 20.90% | **1.25 pp** | **0.0154** | 7 | **1.03%** | **27.38%** | 202.4 s |
| learned count threshold | 19.30% | **22.00%** | 2.70 pp | 0.0925 | **4** | 1.07% | 27.94% | 203.3 s |

The learned count token improves hard test accuracy by `1.10 pp`, improves the
best hard validation score by `1.55 pp` (`22.60%` versus `21.05%`), and reaches
20% hard validation accuracy three epochs earlier. It does not reduce the
discretization gap: the accuracy gap increases by `1.45 pp`, while the loss gap
also grows. The useful conclusion is therefore narrower than an overall win:
integer count thresholds preserve more class-relevant global information, but
they do not solve soft-to-hard mismatch.

Both rows pass exact hard-carrier versus Boolean-reference checks, have zero
hard-path accuracy difference, and retain about 1% entropy-unused gates after
hardening. This is a one-seed, short-budget architecture screen, not part of the
primary matched Mind-the-Gap method ranking. A full-data run is justified only
as a capacity ablation.

`comparison.csv` is the compact comparison table. The two `*_summary.json`
files retain complete arguments, histories, state diagnostics, and layer-gap
measurements. One-row required-metric CSVs are included; complete training logs
remain in the remote run directory. No checkpoints or dataset files are
committed.
