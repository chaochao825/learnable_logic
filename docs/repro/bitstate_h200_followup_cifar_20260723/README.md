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
| strong Gumbel-ST | 14.99% | 15.71% | 0.72 pp | 2.2547 | 2.2551 | 0.0003 | 1849.7 s | not reached | 79.74% | 22.01% |

Annealing is the strongest final classifier in this seed: it improves hard
accuracy by `0.68 pp` and lowers the gap by `1.82 pp` versus strong DLGN. It is
also `2.37 pp` more accurate and has a `0.30 pp` smaller gap than progressive
scale16. The proposed method therefore cannot claim a seed-0 win over simple
annealing. It still reaches 20% hard accuracy earlier than annealing and has
far fewer entropy-unused and activation-inactive gates, although DLGN reaches
that target fastest.

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
and named-layer 16-function histograms. A full post-hoc depth file will be
added after the queued matched-protocol runs release a GPU; the summary already
contains the final eight-batch depth diagnostics and exact hard-carrier versus
integer-bit verification.

The width-8192 progressive run, strict low-margin AdamW DLGN/annealing rows,
and paired seeds 1-2 remain in progress. Until those finish, the optimizer
comparison above is a strongest-baseline comparison rather than a one-factor
training-method ablation.
