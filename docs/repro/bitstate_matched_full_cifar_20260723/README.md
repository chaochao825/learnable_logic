# Strict matched full-CIFAR H200 comparison

This artifact isolates the training method under one shared low-margin AdamW
protocol. Every primary row uses the same width-4096 persistent Boolean
architecture, full CIFAR-10 split, 30 epochs, fixed wiring generated from the
matched seed, and H200 NVL hardware. The four methods are DLGN argmax,
temperature annealing, Gumbel-ST, and progressive hard-ST at logit scale 16.
All results restore the best validation checkpoint and evaluate the official
10,000-image test split.

The primary comparison uses seeds 0, 1, and 2. A uniform 20% hard-validation
target is recomputed from stored epoch histories for convergence timing. The
older seed-0 summaries used a 50% reporting target, but that argument only
records the first-hit epoch and never stops or changes training.

## Protocol audit

DLGN/annealing and the new Gumbel seeds use source `5aaee9d`; the earlier
Gumbel seed 0 and progressive runs use `7cc6b24`. The model, gate, block,
encoder, and teacher Git blobs are identical across these commits. The later
commit adds diagnostics, optional N(0,1) initialization, stronger baseline
launchers, and the 20% reporting target. Its default targeted training path is
unchanged. The generated validation manifest additionally checks all stored
model and training fields that can affect this experiment.

For every seed, the four methods have exactly the same gate count, depth, and
maximum fanout. The comparison excludes method-defining fields such as
temperature schedule, Gumbel sampling, and progressive hardening from the
common-protocol equality check. Every final model passes exact hard-carrier
versus integer/Boolean execution verification.

## Three-seed result

Values are mean plus or minus sample standard deviation over three seeds.

| method | soft acc | hard acc | acc gap | loss gap | train time | time to 20% | entropy-unused | inactive | max layer flip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DLGN | 28.06 +/- 0.15% | 27.14 +/- 0.68% | 0.92 +/- 0.61 pp | 0.0306 +/- 0.0130 | 1966.1 +/- 13.5 s | 285.4 +/- 36.3 s | 10.61 +/- 0.98% | 3.84 +/- 0.15% | 12.33 +/- 0.62% |
| annealing | 26.41 +/- 0.28% | 25.91 +/- 1.35% | 0.75 +/- 0.98 pp | 0.0141 +/- 0.0080 | **1955.0 +/- 12.6 s** | 717.0 +/- 126.5 s | 13.59 +/- 0.27% | **2.32 +/- 0.23%** | 20.58 +/- 1.27% |
| Gumbel-ST | 10.00 +/- 0.00% | 13.06 +/- 1.26% | 3.06 +/- 1.26 pp | 0.0486 +/- 0.0289 | 2041.0 +/- 10.4 s | not reached (0/3) | 94.39 +/- 3.56% | 4.09 +/- 0.76% | 47.36 +/- 0.74% |
| progressive scale16 | **28.16 +/- 1.14%** | **27.79 +/- 1.17%** | **0.62 +/- 0.27 pp** | **0.0139 +/- 0.0053** | 1987.8 +/- 6.7 s | **284.3 +/- 38.7 s** | **0.042 +/- 0.017%** | 3.56 +/- 0.17% | **7.23 +/- 0.35%** |

### Against DLGN

Progressive hardening improves hard accuracy in every paired seed by `+0.07`,
`+0.98`, and `+0.90 pp`; the mean improvement is `+0.65 pp`. Its accuracy-gap
changes are `-1.05`, `+0.35`, and `-0.21 pp`, so the mean gap falls from
`0.92` to `0.62 pp` (`32.85%` relative) but the direction is not consistent in
all seeds. Mean loss gap falls by `0.0167`.

The strict optimizer comparison is not a convergence penalty: progressive
uses `1.10%` more total wall time and reaches 20% only `1.2 s` earlier on
average. This differs from the strongest-recipe comparison, where N(0,1)+Adam
DLGN reaches 20% much faster than the low-margin AdamW progressive recipe.

Entropy-unused gates fall from `10.61%` to `0.042%`, a `99.60%` relative
reduction rather than the paper's claimed 100%. Activation-inactive gates fall
by only `0.28 pp`, confirming that parameter commitment and data-dependent
activity are distinct metrics.

### Against annealing

Progressive improves hard accuracy in every seed by `+0.05`, `+1.43`, and
`+4.17 pp`, or `+1.88 pp` on average. The gap changes are `+0.42`, `+0.69`, and
`-1.51 pp`: progressive has a slightly lower mean gap, but annealing is better
in two of three seeds. Progressive reaches 20% `432.7 s` earlier on average
and lowers peak hidden-state mismatch by `13.35 pp`, while annealing retains a
`1.24 pp` lower activation-inactive ratio.

### Against Gumbel-ST

Gumbel-ST collapses under this persistent-state protocol: all three soft
models remain at exactly `10.00%`, the hard models reach only
`13.06 +/- 1.26%`, and no seed reaches the 20% hard-validation target.
Progressive improves hard accuracy by `14.73 pp`, reduces mean accuracy gap by
`2.44 pp`, trains `53.2 s` faster, lowers entropy-unused gates by `94.35 pp`,
and lowers peak hidden-state mismatch by `40.13 pp` on average. This is an
architecture-and-recipe-specific failure, not a contradiction of the paper's
result on its original feed-forward LGN configuration.

## Depth-wise mismatch

The table reports the mean binary flip ratio between soft and hard traces at
every stored boundary.

| method | encoder | local 1 | local 2 | global 1 | global 2 | votes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DLGN | 0.27% | 3.03% | 4.40% | 7.54% | 9.24% | 12.33% |
| annealing | 0.31% | 5.63% | 9.70% | 17.39% | 20.58% | 13.98% |
| Gumbel-ST | 1.07% | 20.88% | 33.10% | 45.77% | 47.36% | 42.20% |
| progressive scale16 | 0.81% | **1.17%** | **1.39%** | **2.12%** | **2.96%** | **7.23%** |

DLGN accumulates mismatch almost monotonically. Progressive starts with a
slightly larger encoder mismatch but grows much more slowly through local and
global blocks. Annealing peaks at the second global block and then partially
cancels the disagreement in its vote head. Its small final accuracy gap
therefore does not imply a small internal representation gap. Gumbel-ST
accumulates the largest mismatch, reaching `47.36%` in the second global
block, alongside chance-level soft classification.

## Structural selection

| method | constant | all literals | nontrivial two-input |
| --- | ---: | ---: | ---: |
| DLGN | 3.56 +/- 0.12% | 88.14 +/- 0.16% | 8.30 +/- 0.24% |
| annealing | **2.07 +/- 0.23%** | 92.59 +/- 0.36% | 5.33 +/- 0.19% |
| Gumbel-ST | 3.06 +/- 0.69% | 69.50 +/- 2.56% | **27.43 +/- 1.87%** |
| progressive scale16 | 3.26 +/- 0.09% | 89.09 +/- 0.17% | 7.65 +/- 0.19% |

DLGN and progressive select nearly the same literal-dominated hard function
mix. Progressive's near-zero entropy-unused ratio therefore reflects sharper
commitment to a similar circuit, not increased use of genuinely two-input
logic. Annealing is even more literal-dominated. Gumbel-ST selects more
nontrivial functions, but its `94.39%` entropy-unused ratio and chance-level
soft accuracy show that function-count diversity alone is not useful gate
commitment. The failed anti-literal screens are retained separately because
forcing complex argmax functions likewise did not produce committed or useful
gates.

## Cross-hardware seed-0 replay

The earlier RTX 4090 rows use the same seed and stored arguments but are not
used in H200 means or speed comparisons.

| method | hardware | hard acc | acc gap | train time | time to 20% |
| --- | --- | ---: | ---: | ---: | ---: |
| DLGN | RTX 4090 | 25.50% | 2.33 pp | 3163.4 s | 423.2 s |
| DLGN | H200 NVL | 26.37% | 1.63 pp | 1981.1 s | 266.3 s |
| annealing | RTX 4090 | 27.64% | 0.87 pp | 3159.4 s | 1684.9 s |
| annealing | H200 NVL | 26.39% | 0.16 pp | 1945.5 s | 712.6 s |

Fixed seeds do not make this bfloat16 GPU training bit-deterministic across
architectures. The replay changes DLGN hard accuracy by `+0.87 pp` and
annealing by `-1.25 pp`. Same-hardware three-seed aggregates are therefore the
primary evidence; `matched_results.csv` is retained only as the historical
mixed-hardware seed-0 table.

## Artifact map

- `h200_matched_runs.csv`: all 12 per-seed rows with the requested metrics and
  structural/depth diagnostics.
- `h200_matched_aggregate.csv`: means and sample standard deviations.
- `h200_matched_paired_deltas.csv`: proposed-minus-baseline paired differences
  for DLGN, annealing, and Gumbel-ST.
- `h200_required_metrics_table.csv`: four three-seed means in the exact
  requested 14-column schema.
- `h200_layer_gap_runs.csv` and `h200_layer_gap_aggregate.csv`: boundary-level
  depth accumulation evidence.
- `h200_matched_validation.json`: protocol assertions and SHA-256 hashes for
  every source summary/diagnostic file.
- `aggregate_h200_matched.py`: standard-library-only deterministic generator.

DLGN and annealing raw H200 summaries are stored here. The historical Gumbel
seed 0 and all progressive summaries remain in the adjacent
`bitstate_h200_long_cifar_20260723` and
`bitstate_multiseed_cifar_20260723` artifacts; their hashes are included in
the validation manifest. Gumbel seeds 1-2 and all independent posthoc files
are retained here.
