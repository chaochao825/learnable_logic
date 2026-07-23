# Nontrivial-gate regularization screen

This bounded CIFAR-10 screen tests whether directly discouraging literal gates
fixes the identity-routing collapse observed in the low-margin persistent
Boolean model. Every row uses the width-4096 architecture, fixed wiring,
10,000 training images, a 2,000-image validation split, 2,000 test images,
eight epochs, four soft warm-up epochs, seed 0, and an RTX 4090. The target
penalty is

`weight * relu(target - probability mass on genuine two-input functions)`.

The first four runs apply it to every trainable gate layer. They use source
`5145537`. The final two apply it only to the two global state-merge layers,
where selecting input `a` bypasses the cross-token message. They use source
`9a73976`. The scope change is the only implementation difference relevant to
this comparison.

Two commitment follow-ups retain the merge-only nontrivial target at weight
0.1 and raise the existing entropy-target weight over all gates. They also use
source `9a73976`. A final pair uses source `145099a` to apply that entropy
target only to global merges. All other architecture, data, optimization, and
selection settings remain fixed.

## Final metrics

| scope | target | weight | soft acc | hard acc | acc gap | loss gap | entropy-unused | max layer flip | selected nontrivial |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all, control | 0.5 | 0.0 | **19.65%** | 20.90% | **1.25 pp** | **0.0154** | **1.03%** | **27.38%** | 2.45% |
| all | 0.5 | 0.1 | 18.75% | 20.05% | 1.30 pp | 0.0250 | **1.03%** | 25.93% | 2.47% |
| all | 0.8 | 0.1 | 15.75% | 19.25% | 3.50 pp | 0.0677 | 88.55% | 48.10% | 97.94% |
| all | 0.8 | 0.5 | 16.40% | 18.40% | 2.00 pp | 0.0375 | 89.52% | 50.25% | 99.44% |
| global merges | 0.8 | 0.1 | 14.00% | 20.10% | 6.10 pp | 0.1223 | 47.72% | 48.36% | 56.35% all / 99.56% merge mass |
| global merges | 0.8 | 0.5 | 15.85% | **21.05%** | 5.20 pp | 0.1072 | 50.11% | 48.78% | 56.92% all / 99.94% merge mass |

The weak all-layer penalty changes neither the selected gate structure nor the
hard accuracy materially. The stronger target succeeds at its literal
objective but damages the optimization: selected nontrivial gates rise above
97%, while confidence falls below 17%, entropy-unused gates rise above 88%,
and peak hidden-state mismatch approaches 50%.

Restricting the penalty to global merges is more role-appropriate but does not
solve the optimization problem. The stronger merge-only run gains only
`0.15 pp` hard accuracy over control, while its accuracy gap grows from
`1.25 pp` to `5.20 pp`, its loss gap grows by almost 7x, and its peak layer
flip ratio rises by `21.40 pp`. It therefore fails the predeclared requirement
of preserving hard accuracy and hardening stability and is not promoted to a
full-CIFAR run.

## Commitment follow-up

| entropy scope | entropy weight | soft acc | hard acc | acc gap | loss gap | entropy-unused | max layer flip | confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.1 | 18.90% | 20.95% | 2.05 pp | 0.0340 | 46.51% | 47.03% | 54.11% |
| all | 0.5 | **19.00%** | 20.55% | **1.55 pp** | **0.0311** | **45.82%** | 45.47% | **56.53%** |
| global merges | 0.1 | 18.85% | **22.25%** | 3.40 pp | 0.0632 | 55.75% | 46.07% | 39.67% all / 22.63% merge |
| global merges | 0.5 | 17.35% | 20.50% | 3.15 pp | 0.0329 | 55.74% | **42.28%** | 41.39% all / 25.91% merge |

Stronger all-layer entropy regularization recovers much of the final output
gap lost by the structural penalty, but it does not recover the control's
`1.25 pp` gap, `1.03%` entropy-unused ratio, or `27.38%` peak layer flip. Its
best hard accuracy differs from control by only `+0.05 pp`.

Scope-only entropy isolates the merge commitment term but removes the baseline
entropy pressure from every other gate. The 0.1 row gains `1.35 pp` hard
accuracy, yet its gap, unused ratio, and hidden-state mismatch all worsen
materially. Even the merge layers remain weakly committed. This is useful
capacity evidence, not a successful gap-closing method. A future combined
objective would need to retain the all-layer base term and add a separate
merge-specific margin; the present screen does not justify a full-CIFAR run.

## Layer selection

| run | merge 1 input-a | merge 1 nontrivial | merge 2 input-a | merge 2 nontrivial |
| --- | ---: | ---: | ---: | ---: |
| control | 93.58% | 1.20% | 93.19% | 2.08% |
| all, target 0.8, weight 0.1 | 0.02% | 99.95% | 0.02% | 98.61% |
| merge-only, target 0.8, weight 0.1 | 0.00% | 100.00% | 0.00% | 98.95% |
| merge-only, target 0.8, weight 0.5 | 0.00% | 100.00% | 0.00% | 99.93% |

This confirms that the penalty removes the argmax bypass exactly where
intended. The negative result is not failure to enforce the structural target:
it is failure to make those two-input choices confident and task useful.
Argmax function diversity is therefore not a sufficient proxy for trained
Boolean computation. A suitable replacement must couple message usefulness
to commitment, for example by margin-aware hard refitting or a staged
distillation objective, rather than rewarding uncertain mass on all complex
functions equally.

`screen_results.csv` contains the six structural-target rows and
`commitment_results.csv` contains the four commitment follow-ups. Each summary
JSON is the unaltered training artifact. Each `gate_function_metrics.json` file
contains the exact 16-function histogram for every named layer plus
independently recomputed deployment and depth-gap diagnostics.
`epochs_to_target=-1` means the 20% validation target was not reached.
