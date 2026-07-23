# Matched low-margin full-CIFAR comparison

This artifact isolates the training method under the shared low-margin AdamW
protocol. Every row uses the same width-4096 architecture, fixed wiring, full
CIFAR-10 split, 30 epochs, seed 0, gate initialization strength 0.1, collapse
regularization, and evaluation code. DLGN and annealing ran on RTX 4090 GPUs;
the earlier Gumbel-ST and progressive runs used H200 NVL GPUs. Accuracy and gap
are directly comparable, but wall-clock speed is not until the queued H200
DLGN/annealing replays finish.

DLGN/annealing used source `5aaee9d`; the earlier Gumbel/progressive runs used
source `7cc6b24`. Their stored training arguments confirm the same model and
optimization protocol. This table therefore treats accuracy and gap as a
matched experiment while retaining source and hardware provenance explicitly.

## Required metrics

| method | soft acc | hard acc | acc gap | soft loss | hard loss | loss gap | time to 20% | entropy-unused | inactive | max layer flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLGN | **27.83%** | 25.50% | 2.33 pp | 1.9944 | 2.0578 | 0.0634 | 423.2 s | 11.51% | 3.90% | 13.06% |
| annealing | 26.77% | **27.64%** | 0.87 pp | 2.0161 | **2.0131** | **0.0030** | 1684.9 s | 13.74% | **2.08%** | 21.88% |
| Gumbel-ST | 10.00% | 12.64% | 2.64 pp | 2.3026 | 2.3602 | 0.0575 | not reached | 90.29% | 4.91% | 46.64% |
| progressive scale16 | 27.02% | 26.44% | **0.58 pp** | **1.9917** | 2.0050 | 0.0133 | **263.9 s** | **0.06%** | 3.61% | **6.85%** |

Against matched DLGN, progressive hardening improves hard accuracy by
`0.94 pp`, reduces accuracy gap by `1.75 pp` (`75.11%` relative), and halves
the peak layer flip ratio. This is the cleanest evidence that progressive hard
training improves the discrete model rather than merely changing the baseline
configuration. It also decisively beats the matched Gumbel run in this
persistent-state architecture.

Annealing remains the strongest width-4096 hard classifier, exceeding
progressive hardening by `1.20 pp`, while progressive has a `0.29 pp` smaller
accuracy gap and much lower internal mismatch. The proposed method therefore
passes the DLGN gap/accuracy criterion and is competitive with Gumbel, but does
not beat the strongest simple sharpening baseline on final accuracy.

## Structural selection

| method | constant | all literals | nontrivial two-input |
| --- | ---: | ---: | ---: |
| DLGN | 3.66% | 88.27% | 8.07% |
| annealing | **1.80%** | 93.06% | 5.14% |
| Gumbel-ST | 3.86% | **66.55%** | **29.59%** |
| progressive scale16 | 3.24% | 89.28% | 7.48% |

All three deterministic low-margin variants are dominated by one-input
literals. Literal collapse is therefore driven largely by the targeted
identity initialization and shared regularization, not uniquely by progressive
hardening. Gumbel selects more two-input functions but remains near chance,
showing that nontrivial function count alone is not sufficient either. The
next ablation targets only global merge gates, where direct-`a` means bypassing
the cross-token message.

`matched_results.csv` is the machine-readable table. This directory retains
the DLGN/annealing summaries and named-layer gate histograms; the source JSONs
for Gumbel/progressive remain in the adjacent
`bitstate_h200_long_cifar_20260723` artifact. The two
`*_posthoc_diagnostics.json` files independently reload the matched DLGN and
annealing checkpoints and retain exact hard-path checks, named-layer gate
histograms, inactive ratios, and eight-batch depth-gap measurements.
