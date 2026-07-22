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

## Mind-the-Gap claim audit

| paper target | scaled result | status |
| --- | --- | --- |
| up to 4.5x faster convergence | direct block refit is 3.63x to 4.13x faster in Boolean total training time, but no compared method reaches the task target; task-aware refit is 0.95x to 1.94x | not reproduced as convergence speed |
| 98% lower discretization gap | block methods reduce Boolean gap by 6.1% to 72.4%; Gumbel reduces it by 41.4% to 86.1% | not reproduced on Boolean tasks |
| 100% lower unused-gate ratio | block refit has 98.6% to 99.0% entropy-unused gates and is worse than DLGN | rejected at this scale |

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
| reduce gap relative to DLGN | pass on all tested datasets |
| avoid depth-wise accumulation | pass within frozen prefixes |
| reduce unused gates | fail |
| keep hard accuracy close to useful soft accuracy | mixed; fails image probes |
| work without relying on Gumbel-ST | pass mechanically |
| remain competitive with strongest baselines | partial on Boolean, fail on digits |

The current prototype establishes a useful controlled ablation: per-block hard
refitting localizes depth-wise mismatch and can improve Boolean hard accuracy.
It does not yet establish a generally superior training method. The next
research step should optimize hard block selection for downstream hard
accuracy while explicitly regularizing gate entropy, then rerun at multiple
depths and seeds before attempting the full CIFAR scale.

## References

- [Mind the Gap: Removing the Discretization Gap in Differentiable Logic Gate Networks](https://arxiv.org/abs/2506.07500)
- [Deep Differentiable Logic Gate Networks](https://arxiv.org/abs/2210.08277)
- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
- [Official difflogic implementation](https://github.com/Felix-Petersen/difflogic)
