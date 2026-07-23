# Scaled Mind-the-Gap Hard-LGN results

## Scope

This report evaluates the research prototype under matched architecture,
dataset, epoch budget, wiring, and seed conditions. It is a scaled protocol
study, not a reproduction of the full 256K-wide, 12-layer CIFAR experiments in
*Mind the Gap*.

The implementation includes:

- relaxed 16-function DLGN with fixed random wiring and GroupSum;
- temperature-annealed DLGN;
- Gumbel-Softmax with straight-through hard forward;
- deterministic Hard-ST with CAGE-style adaptive backward temperature;
- block-wise relaxed training with final argmax extraction;
- block-wise hard training with per-block truth-table refit;
- task-aware coordinate refit over locally plausible truth tables;
- optional BLIF export, ABC optimization, and post-ABC functional evaluation.

The complete required table is
[required_metrics_table.csv](../repro/mind_gap_scaled_required_table_20260723/required_metrics_table.csv).
Its 34 rows contain every requested field. Per-seed trajectories, block fitting
diagnostics, and layer-prefix mismatch diagnostics are retained with each run.

## Protocol

`--mind-gap-scaled` fixes the paper-aligned optimizer and output protocol:

- Adam, learning rate `0.01`, no weight decay;
- batch size `128`;
- GroupSum scale `1 / 0.01`;
- fixed Gumbel temperature `1.0`;
- fixed random wiring shared by methods for each architecture and seed;
- equal total full-dataset epoch counts for end-to-end and block-wise methods.

The paper-aligned `unused_gate_ratio` is the fraction of gate-logit vectors
whose entropy exceeds the lower edge of the 95% initialization interval. The
deterministic threshold is `1.884315` for 100,000 freshly initialized
16-function logits. The older data-dependent constant-output statistic remains
available as `activation_inactive_gate_ratio`; it is not substituted for the
paper metric.

| run | scale | seeds | role |
| --- | --- | --- | --- |
| Boolean | width 128, depth 4, 120 epochs | 0, 1, 2 | primary matched comparison |
| sklearn digits | width 320, depth 4, 120 epochs | 0 | image-oriented probe |
| CIFAR-10-small | width 1600, depth 3, 20 epochs, 2K/500 split | 0 | execution-path smoke test |
| ABC Boolean | width 128, depth 4, 120 epochs | 0 | synthesis and functional check |

## Primary result

The hypothesis is partially supported on small Boolean tasks and rejected as a
general winner by the current image probes.

Both truth-table refit variants reduce the conventional accuracy gap relative
to DLGN on all three Boolean tasks. They do not consistently match Gumbel's gap,
and neither improves the paper-defined unused-gate ratio. Direct refit often
improves hard accuracy despite a larger gap than Gumbel, showing why gap and
absolute hard accuracy must be read together.

| dataset | method | hard acc | acc gap | gap reduction vs DLGN |
| --- | --- | ---: | ---: | ---: |
| parity8 | DLGN | 52.08% | 15.10 pp | 0.0% |
| parity8 | Gumbel-ST | 54.69% | 8.85 pp | 41.4% |
| parity8 | block hard refit | 48.44% | 5.73 pp | 62.1% |
| parity8 | task-aware refit | 45.83% | 4.17 pp | 72.4% |
| majority9 | DLGN | 70.05% | 29.95 pp | 0.0% |
| majority9 | Gumbel-ST | 52.08% | 4.17 pp | 86.1% |
| majority9 | block hard refit | 78.13% | 28.13 pp | 6.1% |
| majority9 | task-aware refit | 67.71% | 15.63 pp | 47.8% |
| random_sparse10 | DLGN | 61.72% | 38.28 pp | 0.0% |
| random_sparse10 | Gumbel-ST | 55.47% | 6.38 pp | 83.3% |
| random_sparse10 | block hard refit | 69.92% | 19.27 pp | 49.7% |
| random_sparse10 | task-aware refit | 61.85% | 13.54 pp | 64.6% |

Across the three Boolean tasks, the unweighted mean conventional gap is
`27.78 pp` for DLGN, `6.47 pp` for Gumbel-ST, `17.71 pp` for direct block
refit, and `11.11 pp` for task-aware refit. Mean hard accuracy is `61.28%`,
`54.08%`, `65.49%`, and `58.46%`, respectively.

Block-relaxed training reaches an even smaller mean gap (`8.46 pp`) but only
`46.48%` mean hard accuracy. Gumbel-ST similarly has low gaps while remaining
near chance on majority and sparse Boolean tasks. A small gap therefore does
not by itself demonstrate useful discrete learning.

## Depth-wise behavior

For the two hard block methods, the first three frozen prefixes have exactly
zero representation MAE and zero binary-flip ratio in all nine Boolean
dataset/seed combinations. Mismatch is confined to the final trainable block:

| method | frozen prefix flip ratio | final MAE | final flip ratio |
| --- | ---: | ---: | ---: |
| block hard refit | 0.0% | 0.3966 | 0.0% |
| task-aware refit | 0.0% | 0.4137 | 10.28% |

In contrast, DLGN's mean prefix flip ratio grows from `11.73%` after layer 1
to `24.54%`, `30.83%`, and `34.97%` through depth 4. Hard refitting therefore
prevents mismatch accumulation inside the frozen prefix by construction, but
it does not guarantee that the final fitted block preserves the all-relaxed
network.

Local truth-table fitting reduces mean per-gate reconstruction MSE from
`0.1764` for direct argmax to `0.1626`. The task-aware fitter selects its
coordinate-refined candidate for every fitted block and has `0.1860` local
MSE, reflecting its deliberate trade of local fidelity for task loss.

## Image probes

The image-oriented evidence is negative for the proposed method at the tested
budgets.

| dataset | method | hard acc | acc gap | entropy-unused |
| --- | --- | ---: | ---: | ---: |
| digits | DLGN | 42.00% | 50.22 pp | 79.06% |
| digits | annealed DLGN | 69.33% | 20.89 pp | 70.94% |
| digits | Gumbel-ST | 8.22% | 1.33 pp | 99.69% |
| digits | block hard refit | 24.44% | 14.22 pp | 98.91% |
| digits | task-aware refit | 11.78% | 1.78 pp | 98.28% |
| digits | Hard-ST/CAGE | 64.89% | 0 pp native-path gap | 83.98% |
| CIFAR-10-small | DLGN | 9.00% | 13.00 pp | 97.67% |
| CIFAR-10-small | Gumbel-ST | 10.60% | 2.00 pp | 98.40% |
| CIFAR-10-small | block hard refit | 16.40% | 6.40 pp | 98.92% |
| CIFAR-10-small | Hard-ST/CAGE | 19.40% | 0 pp native-path gap | 98.71% |

Annealing is the strongest digits baseline by hard accuracy. The very small
Gumbel and task-refit gaps coincide with collapsed absolute accuracy, so those
rows are not successful gap closure. CIFAR-10-small validates the execution
path only; its 4,800-gate, 20-epoch setting is not comparable to the reported
61M-gate ConvLGN result.

## Persistent-state full CIFAR stress test

A separate scalable prototype keeps every hidden boundary Boolean and uses
XNOR-popcount Top-K routing, majority messages, Boolean state merges, and an
integer GroupSum head. This is a stress test of persistent hard state and
progressive hardening, not a direct scale-up of the per-block truth-table
refitter above. All rows below use the same width-4096 architecture, full
CIFAR-10 split, 30 epochs, seed 0, fixed wiring, and H200 NVL hardware.

| method | hard acc | acc gap | train time | time to 20% | entropy-unused | max layer flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| strong DLGN | 28.13% | 2.10 pp | 1766.6 s | **60.5 s** | 40.24% | 19.11% |
| strong annealing | **28.81%** | **0.28 pp** | **1766.4 s** | 413.3 s | 42.63% | 19.80% |
| progressive scale16 | 26.44% | 0.58 pp | 1994.7 s | 263.9 s | **0.06%** | **6.85%** |
| progressive scale16, width 8192 | 27.88% | **0.05 pp** | 3161.1 s | 521.2 s | **0.03%** | 9.05% |
| strong Gumbel-ST | 15.71% | 0.72 pp | 1849.7 s | not reached | 79.74% | 31.01% |

Progressive hardening reduces the DLGN gap by `72.38%` and greatly limits
depth-wise mismatch, but loses `1.69 pp` hard accuracy to DLGN. Strong
annealing is the seed-0 winner: it improves DLGN hard accuracy by `0.68 pp`
and produces a smaller gap than the proposed hardening schedule at essentially
the same runtime as DLGN. The proposed schedule therefore does not beat the
strongest simple baseline.

The earlier mixed-hardware low-margin seed-0 table is superseded by the
same-H200, four-method, three-seed comparison below. The old rows remain useful
as a hardware-sensitivity replay, not as the primary method ranking.

Width 8192 adds `82.05%` gates and `58.47%` H200 training time for `1.44 pp`
more hard accuracy than width 4096. It also lowers the final gap to `0.05 pp`,
but increases the maximum internal flip ratio from `6.85%` to `9.05%` and
raises literal selection to `93.98%`. Scaling supplies useful routing
redundancy, with visibly diminishing structural efficiency.

The unused-gate result also needs qualification. Progressive scale16 selects
one-input literals for `89.28%` of gates and genuinely two-input functions for
only `7.48%`; its low entropy-unused ratio mostly reflects committed routing,
not broad Boolean computation. Strong annealing retains `56.01%` nontrivial
two-input gates. Conversely, annealing's tiny final accuracy gap coexists with
a `19.80%` peak hidden-state flip ratio, so final accuracy-gap can hide large
internal mismatch through downstream cancellation. The two methods trade
internal bit stability against final classification accuracy rather than one
strictly dominating the other.

The low-margin matched DLGN and annealing rows also select `88.27%` and
`93.06%` literals. Thus this structural collapse is driven mainly by the
identity-biased protocol shared by those ablations, not by progressive
hardening alone. The N(0,1) strong baselines are needed to expose that
confounder.

### Strict shared-protocol three-seed check

All four methods use low-margin targeted initialization, AdamW/cosine, the
same collapse regularization, width 4096, 30 epochs, seeds 0-2, and H200 NVL
hardware. Values are mean plus or minus sample standard deviation.

| method | hard acc | acc gap | loss gap | train time | time to 20% | entropy-unused | max layer flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLGN | 27.14 +/- 0.68% | 0.92 +/- 0.61 pp | 0.0306 +/- 0.0130 | 1966.1 +/- 13.5 s | 285.4 +/- 36.3 s | 10.61 +/- 0.98% | 12.33 +/- 0.62% |
| annealing | 25.91 +/- 1.35% | 0.75 +/- 0.98 pp | 0.0141 +/- 0.0080 | **1955.0 +/- 12.6 s** | 717.0 +/- 126.5 s | 13.59 +/- 0.27% | 20.58 +/- 1.27% |
| Gumbel-ST | 13.06 +/- 1.26% | 3.06 +/- 1.26 pp | 0.0486 +/- 0.0289 | 2041.0 +/- 10.4 s | not reached (0/3) | 94.39 +/- 3.56% | 47.36 +/- 0.74% |
| progressive scale16 | **27.79 +/- 1.17%** | **0.62 +/- 0.27 pp** | **0.0139 +/- 0.0053** | 1987.8 +/- 6.7 s | **284.3 +/- 38.7 s** | **0.042 +/- 0.017%** | **7.23 +/- 0.35%** |

Progressive beats DLGN hard accuracy in all paired seeds by `+0.07`, `+0.98`,
and `+0.90 pp`. Mean gap falls by `0.30 pp` (`32.85%` relative), but seed 1
is a counterexample: its gap rises by `0.35 pp`. The method uses `1.10%` more
total time and has essentially identical time to 20%, so the strict result is
neither a convergence penalty nor a Mind-the-Gap-like speedup.

Progressive also beats annealing hard accuracy in all three seeds by `+0.05`,
`+1.43`, and `+4.17 pp`. Annealing has a smaller final gap in seeds 0 and 1,
while progressive wins seed 2 and has a slightly smaller mean. Progressive
reaches 20% `432.7 s` earlier and reduces the peak internal flip ratio by
`13.35 pp`; annealing retains a `1.24 pp` lower activation-inactive ratio.

Gumbel-ST collapses in this persistent-state architecture: soft accuracy is
exactly `10.00%` in every seed, hard accuracy is `13.06 +/- 1.26%`, and no run
reaches 20%. Relative to Gumbel-ST, progressive raises mean hard accuracy by
`14.73 pp`, lowers accuracy gap by `2.44 pp`, entropy-unused gates by
`94.35 pp`, and peak layer flips by `40.13 pp`. This negative result is
specific to the tested architecture and shared recipe; it is not a direct
reproduction or refutation of the paper's feed-forward LGN result.

The depth traces separate final cancellation from stable hard computation.
Mean DLGN flip ratio grows from `0.27%` at the encoder to `12.33%` at the vote
head. Annealing peaks at `20.58%` after the second global block before falling
to `13.98%` at votes. Gumbel-ST grows from `1.07%` to `47.36%` before ending
at `42.20%`. Progressive grows from `0.81%` to only `7.23%`, with
`1.17-2.96%` mismatch through its local/global blocks. It therefore reduces,
but does not eliminate, depth-wise mismatch accumulation.

The hard function mix remains literal dominated: DLGN and progressive select
`88.14%` and `89.09%` literals, with only `8.30%` and `7.65%` genuinely
two-input gates. Gumbel-ST selects `27.43%` nontrivial functions but leaves
`94.39%` of gates entropy-unused. Progressive's `99.60%` relative reduction
in entropy-unused gates is a commitment result, not evidence for more complex
Boolean functions.

### Three-seed strongest-recipe check

Over seeds 0-2, strong DLGN and progressive scale16 reach `27.70 +/- 0.43%`
and `27.79 +/- 1.17%` mean hard accuracy. The paired proposed-minus-DLGN
differences are `-1.69`, `+1.11`, and `+0.85 pp`; the `+0.09 pp` mean does not
support a consistent accuracy win. Proposed nevertheless reduces the gap in
all three seeds, from `2.48 +/- 0.39 pp` to `0.62 +/- 0.27 pp`, a `75.03%`
relative mean reduction. It also lowers peak hidden-state flips by `12.63 pp`
and entropy-unused gates by `39.53 pp` on average.

The cost is slower optimization: proposed takes `12.65%` more total training
time and reaches 20% hard validation accuracy in `284.3 s` versus `80.0 s` for
DLGN. This comparison uses each method's strongest recipe, so it ranks final
systems but is not a one-factor optimizer/initialization ablation. The strict
comparison above shows that the 3.55x convergence difference is attributable
to the strong DLGN recipe rather than hardening alone.

The complete evidence is in
[bitstate_h200_long_cifar_20260723](../repro/bitstate_h200_long_cifar_20260723/README.md)
and
[bitstate_h200_followup_cifar_20260723](../repro/bitstate_h200_followup_cifar_20260723/README.md).
The one-factor table and complete provenance are in
[bitstate_matched_full_cifar_20260723](../repro/bitstate_matched_full_cifar_20260723/README.md).
Three-seed rows, sample deviations, and paired deltas are in
[bitstate_multiseed_cifar_20260723](../repro/bitstate_multiseed_cifar_20260723/README.md).

## Mind-the-Gap claim audit

| paper target | scaled result | status |
| --- | --- | --- |
| up to 4.5x faster convergence | strict full-CIFAR progressive and DLGN reach 20% in 284.3 and 285.4 s; strongest-recipe progressive is 3.55x slower than strong DLGN. Boolean total time is not convergence because targets are missed | not reproduced |
| 98% lower discretization gap | truth-table block methods reduce Boolean gap by 6.1% to 72.4%; strict full-CIFAR progressive reduces mean DLGN gap by 32.85%, and strongest-recipe progressive by 75.03% | not reproduced |
| 100% lower unused-gate ratio | truth-table refit is worse than DLGN on Boolean tasks; strict progressive reduces full-CIFAR DLGN entropy-unused gates by 99.60% but does not reach zero and remains 89.09% literal | not reproduced; metric nearly matched only by committed routing |

The digits Gumbel row reaches a `97.35%` relative gap reduction and the
task-aware row reaches `96.46%`, but their hard accuracies are `8.22%` and
`11.78%`. These are collapse cases, not evidence for the paper's useful
accuracy-preserving gap reduction.

Most runs never reach their dataset accuracy target, so `epochs_to_target` is
`-1`. Reporting only the lower raw wall time of one-layer-at-a-time training
as a convergence speedup would be misleading.

The compiled DLGN claim of more than one million MNIST images per CPU core per
second is also outside this prototype's PyTorch execution scope. The optional
throughput path measures ordinary PyTorch forwards, not a bit-packed compiled
Boolean kernel.

## ABC synthesis

ABC `strash; dc2` optimization was applied to all 21 seed-0 Boolean networks.
Source BLIF and optimized BLIF were independently evaluated over the exact
Boolean test inputs. Every optimized model preserved discrete accuracy and
loss exactly.

- ABC write/optimization averages `0.0333 s` per network (`0.0286-0.0408 s`).
- All optimized graphs have zero unreachable nodes.
- Direct block refit produces fewer post-ABC nodes than Gumbel-ST on every
  task: `207 vs 240`, `222 vs 266`, and `210 vs 257`.
- Task-aware refit produces the smallest graphs: 151, 170, and 158 nodes.
- AIG levels can increase when arbitrary two-input functions are decomposed;
  post-ABC level is therefore not assumed to improve monotonically.

Synthesis changes structure, not learned behavior. It cannot repair a poor
hard model after fitting.

## Decision

| success criterion | result |
| --- | --- |
| reduce gap relative to DLGN | truth-table refit passes on all Boolean task means; scalable progressive passes on full-CIFAR mean but not every seed |
| avoid depth-wise accumulation | exact frozen prefixes pass by construction; scalable progressive lowers vote flip from 12.33% to 7.23% versus strict DLGN |
| reduce unused gates | truth-table refit fails on Boolean tasks; scalable progressive reduces strict DLGN entropy-unused gates by 99.60% but mostly selects literals |
| keep hard accuracy close to useful soft accuracy | mixed for exact block refit; scalable progressive reaches 27.79% hard from 28.16% soft on full CIFAR |
| work without relying on Gumbel-ST | pass mechanically and in the strict full-CIFAR hard-accuracy comparison |
| remain competitive with strongest baselines | partial: strict progressive beats DLGN/annealing hard accuracy, but strong annealing remains the best seed-0 classifier and absolute CIFAR accuracy is low |

The current prototype establishes a useful controlled ablation: per-block hard
refitting localizes depth-wise mismatch and can improve Boolean hard accuracy.
It does not yet establish a generally superior training method. A bounded
anti-literal screen has now tested both all-layer and role-specific global-merge
penalties. The merge-only penalty removes the argmax bypass almost completely,
but raises entropy-unused gates from `1.03%` to `47.72-50.11%`, peak layer
flips from `27.38%` to about `49%`, and the accuracy gap from `1.25 pp` to
`5.20-6.10 pp`. Raising all-layer entropy pressure reduces that gap to
`1.55-2.05 pp` but still leaves `45.82-46.51%` gates entropy-unused and about
`45-47%` peak internal flips. Restricting entropy pressure to merges reaches
`22.25%` hard accuracy in one bounded row, but has a `3.40 pp` gap,
`55.75%` unused gates, and only `22.63%` merge confidence. None passes the
joint accuracy, gap, and commitment screen, so no variant is promoted to a
full-CIFAR run. The next research step should optimize hard block selection
for downstream hard accuracy while retaining base commitment and coupling
message usefulness to a role-specific margin, then validate over paired seeds.

The complete anti-literal evidence, including named-layer 16-function
histograms, is in
[bitstate_nontrivial_screen_20260723](../repro/bitstate_nontrivial_screen_20260723/README.md).

## References

- [Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](https://arxiv.org/abs/2506.07500)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
- [Official difflogic implementation](https://github.com/Felix-Petersen/difflogic)
