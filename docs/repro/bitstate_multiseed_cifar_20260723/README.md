# Three-seed strongest-recipe CIFAR comparison

This artifact checks whether the seed-0 persistent Boolean result survives
seeds 1 and 2. Every run uses the same width-4096 architecture, full CIFAR-10
split, 30 epochs, fixed wiring generated from the matched seed, H200 NVL
hardware, and exact integer/Boolean deployment verification.

This is a strongest-recipe comparison, not a one-factor optimizer ablation.
Strong DLGN uses N(0,1) gate logits, Adam at constant learning rate 0.01, and
no custom regularization. Proposed progressive scale16 uses targeted
low-margin initialization, AdamW/cosine, state regularization, four soft epochs
before hard straight-through training, and the entropy schedule selected by
the bounded screen. The separate low-margin artifact now supplies the strict
four-method, three-seed shared-protocol comparison.

## Aggregate metrics

Values are mean plus or minus sample standard deviation over three seeds.

| method | soft acc | hard acc | acc gap | loss gap | train time | time to 20% | entropy-unused | inactive | max layer flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong DLGN | **30.18 +/- 0.41%** | 27.70 +/- 0.43% | 2.48 +/- 0.39 pp | 0.0485 +/- 0.0064 | **1764.6 +/- 2.0 s** | **80.0 +/- 34.0 s** | 39.57 +/- 0.65% | 28.47 +/- 0.89% | 19.86 +/- 0.65% |
| proposed scale16 | 28.16 +/- 1.14% | **27.79 +/- 1.17%** | **0.62 +/- 0.27 pp** | **0.0139 +/- 0.0053** | 1987.8 +/- 6.7 s | 284.3 +/- 38.7 s | **0.042 +/- 0.017%** | **3.56 +/- 0.18%** | **7.23 +/- 0.35%** |

The mean hard-accuracy difference is only `+0.09 pp` for proposed. Its paired
seed differences are `-1.69`, `+1.11`, and `+0.85 pp`, so these data do not
support a consistent accuracy improvement. DLGN has `2.02 pp` higher mean soft
accuracy; progressive hardening closes enough of the soft-to-hard mismatch to
make the final hard accuracies nearly equal.

The gap result is consistent across seeds. Proposed reduces accuracy gap by
`1.52`, `1.57`, and `2.50 pp`; the mean reduction is `1.86 pp`, or `75.03%`
relative to DLGN. Mean loss gap falls by `0.0346`, peak hidden-state flip ratio
falls by `12.63 pp`, entropy-unused gates fall by `39.53 pp`, and
activation-inactive gates fall by `24.90 pp`.

This stability is not a convergence-speed win. Proposed uses `12.65%` more
wall-clock training time and reaches 20% hard validation accuracy in `284.3 s`
versus `80.0 s` for DLGN, making it `3.55x` slower on this threshold. It does
not reproduce the Mind-the-Gap faster-convergence claim.

## Structural qualification

| method | constant | all literals | nontrivial two-input |
| --- | ---: | ---: | ---: |
| strong DLGN | 13.98 +/- 0.35% | 28.59 +/- 0.40% | **57.43 +/- 0.45%** |
| proposed scale16 | **3.26 +/- 0.09%** | 89.09 +/- 0.17% | 7.65 +/- 0.19% |

The function mix is highly stable across seeds even though proposed hard
accuracy varies by about two points. Its near-zero entropy-unused ratio means
the gates are committed, but most are committed one-input routes. The method's
repeatable benefit is hard-state stability and reduced mismatch, not broad use
of two-input Boolean computation.

`multiseed_runs.csv` contains all six per-seed rows. Seed-0 values are sourced
from the adjacent `bitstate_h200_long_cifar_20260723` artifact; this directory
retains the new seed-1/2 summaries and post-hoc diagnostics.
`multiseed_aggregate.csv` contains means and sample standard deviations, and
`paired_deltas.csv` records proposed-minus-DLGN differences for every seed.
The historical seed-0 empty target-epoch field is repaired from its stored
history, where it first crosses 20% at epoch 4.
