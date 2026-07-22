# Persistent bit-state H200 screening evidence

This directory preserves the compact results from the first persistent
Boolean-state CIFAR-10 screens. These runs are capacity and optimization
screens, not full-CIFAR conclusions. The matched full-data comparison is run
separately by `vit_lgn/bitstate/run_h200_long_cifar.sh`.

## Protocol

The main screening rows use seed 0, 10,000 training images, a fixed 2,000-image
validation split, a 2,000-image test subset, eight epochs, width 512, two local
and two global blocks, fixed wiring, and the redundant predicate encoder. The
hard path consists only of Boolean gates, XNOR-popcount routing, majority,
threshold predicates, and integer GroupSum. Every retained run verifies that
the floating carrier hard path and the integer/Boolean deployment path agree
exactly.

The source evolved through commits `a08cff7` to `4bcd8ae`; the latter replaced
arithmetic straight-through expressions with a custom autograd operation whose
forward value is exactly hard. The failed pre-fix artifact was quarantined in
remote `trash/` and is not included in the table.

## Selected results

| run | soft acc | hard acc | gap | entropy-unused | activation-inactive | normalized gate entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| targeted DLGN, matched | 16.75% | 17.15% | **0.40 pp** | 100.00% | 0.13% | 0.9439 |
| targeted anneal, matched | 15.65% | 16.15% | 0.50 pp | 100.00% | 0.21% | 0.9397 |
| N(0,1) DLGN, strongest schedule | 23.80% | 23.20% | 0.60 pp | 32.81% | 19.17% | 0.4895 |
| N(0,1) anneal, strongest schedule | 21.90% | **23.25%** | 1.35 pp | 62.20% | 18.03% | 0.6852 |
| Gumbel-ST, init 0.1 | 11.15% | 9.65% | 1.50 pp | 100.00% | 0.72% | 0.9992 |
| Gumbel-ST, N(0,1), Adam, tau 0.1 | 7.60% | 15.65% | 8.05 pp | 96.79% | 23.82% | 0.8452 |
| progressive, scale 2 | 16.00% | **20.95%** | 4.95 pp | 99.37% | 0.21% | 0.8321 |
| progressive, scale 4 | 18.35% | 20.80% | 2.45 pp | 25.30% | 0.17% | 0.5204 |
| progressive, scale 8 | 19.90% | 18.50% | 1.40 pp | 10.26% | 0.13% | 0.1866 |
| progressive, scale 16 | 19.50% | 20.65% | 1.15 pp | **1.94%** | 0.25% | 0.0821 |
| scale 4, entropy ramp 0.1 | 18.65% | 18.25% | **0.40 pp** | 22.04% | 0.21% | 0.4091 |
| scale 4, entropy ramp 0.5 | 19.50% | 20.45% | 0.95 pp | 14.36% | **0.08%** | 0.3277 |

`entropy-unused` follows Mind the Gap: a gate is unused when its categorical
entropy remains above the deterministic 2.5th-percentile boundary for newly
initialized N(0,1) 16-way logits (`1.884315` nats). `activation-inactive` is a
different diagnostic: the gate's hard output is constant on sampled training
batches. Older bit-state summaries stored only the second metric under the
legacy name; `collect_results --infer-legacy-entropy-unused` repairs the paper
metric from each best checkpoint and retains both columns explicitly.

## Interpretation

- Low gap alone is not sufficient: the Gumbel row is near chance while showing
  only a 1.50 pp gap.
- A scaled paper-aligned N(0,1)/Adam/constant-LR screen over tau 0.1, 0.25,
  0.5, and 1.0 improves the strongest Gumbel hard accuracy to 15.65%, but its
  8.05 pp gap and 96.79% entropy-unused ratio do not reproduce the paper's
  behavior on this persistent-state architecture. This is the Gumbel candidate
  selected for the full-data run.
- Scale 16 gives the strongest current accuracy/gap/utilization compromise:
  20.65% hard accuracy, 1.15 pp gap, and 1.94% entropy-unused gates.
- The strongest DLGN and annealing rows reach 23.20% and 23.25% hard accuracy,
  so the proposed method does not win the short screen on absolute accuracy or
  gap. Its clear advantage is gate commitment: 1.94% entropy-unused versus
  32.81% and 62.20%.
- Under the strictly matched targeted initialization, proposed scale 16 gains
  3.50 pp hard accuracy over DLGN and reduces entropy-unused by 98.06 pp, but
  its 1.15 pp gap is larger than DLGN's 0.40 pp.
- Scale 2 has the highest hard accuracy but leaves almost every gate above the
  paper's entropy threshold. Hard argmax accuracy can therefore look useful
  before the relaxed gate distributions have genuinely committed.
- The entropy ramp can reduce the gap further, but the 0.40 pp row also loses
  2.40 pp hard accuracy relative to scale 16.
- All accuracy values in this file use small test subsets and can move by
  several points. They select long-run candidates; they do not establish a
  CIFAR-10 result.

`bitstate_short_results.csv` contains all 56 completed rows, the required loss
and accuracy columns, both unused metrics, structural statistics, anti-collapse
diagnostics, and the complete shape/training identifiers needed to separate
campaigns.
