# Width-4096 full-CIFAR bit-state comparison

This artifact records the first full-data persistent Boolean-state comparison
on server 236. It uses 45,000 CIFAR-10 training images, a deterministic 5,000
image validation split, the official 10,000 image test split, 30 epochs, seed
0, width 4096, two local and two global blocks, and identical fixed wiring.
Both methods ran on H200 NVL GPUs with PyTorch 2.9.1/CUDA 12.8. Training source
was commit `7cc6b24`; the post-hoc metric schema and depth analyzer are commit
`5aaee9d`.

The two rows here are a strictly matched initialization/optimizer ablation.
The stronger N(0,1)/Adam DLGN, annealing, and Gumbel baselines are separate
runs and must be added before treating this as the final method ranking.

## Required metrics

| method | soft acc | hard acc | acc gap | soft loss | hard loss | loss gap | train time | time to 20% | entropy-unused | gates | depth | max fanout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| matched Gumbel-ST | 10.00% | 12.64% | 2.64 pp | 2.3026 | 2.3602 | 0.0575 | 2051.9 s | not reached | 90.29% | 14,976 | 9 | 228 |
| progressive scale16 | 27.02% | **26.44%** | **0.58 pp** | 1.9917 | 2.0050 | **0.0133** | 1994.7 s | 263.9 s | **0.06%** | 14,976 | 9 | 228 |

Both rows have zero hard-carrier versus integer-bit accuracy/loss difference
and pass the exact per-layer deployment assertion. The proposed row reaches
20% hard validation accuracy at epoch 4; the matched Gumbel row never reaches
it. Validation selects epoch 22 for the proposed model (`27.22%` hard) and
epoch 18 for matched Gumbel (`12.82%` hard), after which the restored
checkpoints are evaluated once on the official test split.

## Depth gap

| boundary | proposed flip | matched Gumbel flip |
| --- | ---: | ---: |
| encoder | 0.60% | 1.25% |
| local 1 | 0.99% | 20.14% |
| local 2 | 1.23% | 32.55% |
| global 1 | 2.07% | 45.34% |
| global 2 | 3.02% | 46.64% |
| vote head | 6.85% | 42.81% |

The proposed model does not eliminate depth-wise mismatch: its binary flip
ratio rises from 0.60% after encoding to 6.85% at the votes. It does prevent
the severe accumulation seen in matched Gumbel, which reaches 46.64% before
the vote head. This distinction is hidden by final accuracy gap alone.

`long_results.csv` is the machine-readable required table.
`*_summary.json` retains every epoch and complete training arguments.
`*_posthoc.json` retains both unused definitions and every depth boundary.
No checkpoints or dataset payloads are committed.
