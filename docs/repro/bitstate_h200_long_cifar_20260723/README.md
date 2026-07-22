# Width-4096 full-CIFAR bit-state comparison

This artifact records the first full-data persistent Boolean-state comparison
on server 236. It uses 45,000 CIFAR-10 training images, a deterministic 5,000
image validation split, the official 10,000 image test split, 30 epochs, seed
0, width 4096, two local and two global blocks, and identical fixed wiring.
All four methods ran on H200 NVL GPUs with PyTorch 2.9.1/CUDA 12.8. The matched
low-margin runs used training source `7cc6b24`; the strong N(0,1)/Adam runs and
the post-hoc metric/depth analyzer used `5aaee9d`.

The matched Gumbel and progressive rows share the low-margin AdamW protocol.
The strong DLGN and Gumbel rows use the paper-oriented N(0,1) gate
initialization, Adam with constant learning rate 0.01, and no custom collapse
regularization or label smoothing. They retain the same architecture, data,
budget, wiring seed, and evaluation protocol, but are a tuned-baseline
comparison rather than a one-factor optimizer ablation.

## Required metrics

| method | soft acc | hard acc | acc gap | soft loss | hard loss | loss gap | train time | time to 20% | entropy-unused | activation-inactive |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| strong DLGN | **30.23%** | **28.13%** | 2.10 pp | **1.9259** | **1.9704** | 0.0445 | **1766.6 s** | **60.5 s** | 40.24% | 29.41% |
| matched Gumbel-ST | 10.00% | 12.64% | 2.64 pp | 2.3026 | 2.3602 | 0.0575 | 2051.9 s | not reached | 90.29% | 4.91% |
| strong Gumbel-ST | 14.99% | 15.71% | 0.72 pp | 2.2547 | 2.2551 | **0.0003** | 1849.7 s | not reached | 79.74% | 22.01% |
| progressive scale16 | 27.02% | 26.44% | **0.58 pp** | 1.9917 | 2.0050 | 0.0133 | 1994.7 s | 263.9 s | **0.06%** | **3.61%** |

Every row has zero hard-carrier versus integer-bit accuracy/loss difference
and passes the exact per-layer deployment assertion. Against the strongest
DLGN row, progressive hardening reduces the accuracy gap by `72.38%` and the
entropy-unused ratio by `99.85%` relative, but loses `1.69 pp` hard accuracy
and takes `12.91%` longer. It therefore passes the gap/utilization criteria but
does not win the accuracy or convergence-speed criteria.

The strongest Gumbel row has a small final gap but only `15.71%` hard accuracy.
Progressive hardening is `10.73 pp` more accurate, has a slightly smaller
accuracy gap, and uses far more of its gates. The paper's reported Gumbel
advantage does not transfer to this persistent-state, hard Top-K architecture;
this experiment is not evidence against the paper's original CIFAR model.
Validation selects epoch 18 for strong DLGN (`28.46%` hard), epoch 30 for
strong Gumbel (`16.70%`), and epoch 22 for the proposed model (`27.22%`) before
one evaluation on the official test split.

## Depth gap

| boundary | proposed | strong DLGN | strong Gumbel | matched Gumbel |
| --- | ---: | ---: | ---: | ---: |
| encoder | 0.60% | **0.07%** | 0.14% | 1.25% |
| local 1 | **0.99%** | 3.83% | 14.87% | 20.14% |
| local 2 | **1.23%** | 4.66% | 23.71% | 32.55% |
| global 1 | **2.07%** | 11.29% | 31.01% | 45.34% |
| global 2 | **3.02%** | 19.11% | 30.70% | 46.64% |
| vote head | **6.85%** | 10.37% | 25.40% | 42.81% |

The proposed model does not eliminate depth-wise mismatch: its binary flip
ratio rises from 0.60% after encoding to 6.85% at the votes. It does prevent
the much stronger accumulation seen in DLGN and both Gumbel variants. Strong
DLGN rises to 19.11% at the second global block; strong and matched Gumbel peak
at 31.01% and 46.64%, respectively. This distinction is hidden by final
accuracy gap alone.

`long_results.csv` is the four-row machine-readable required table.
`*_summary.json` retains every epoch and complete training arguments.
`*_posthoc.json` retains both unused definitions and every depth boundary.
No checkpoints or dataset payloads are committed. Full matched DLGN and
annealing runs under the low-margin AdamW protocol remain pending on the 4090s;
multi-seed H200 runs are also pending, so this is a seed-0 ranking rather than
a final variance-aware claim.
