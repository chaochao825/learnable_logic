# A8 bit-plane Hard-LGN digits scale screen

All 18 registered v2 runs completed. Every result was replayed by the
standalone Boolean/integer executor with zero real-valued tensors and
exact hard-carrier logit equality on all validation and test rows.
Soft metrics are the selected final block before hardening/refit; discrete
metrics are measured after the block is hardened and frozen.

| hardening | state/votes | validation hard mean +/- sd | test hard mean +/- sd | gap | inactive votes | unused gates | gates | payload bits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| argmax | 672/160 | 88.02% +/- 1.40% | 85.31% +/- 0.43% | 1.36% | 0.00% | 2.08% | 320 | 24608 |
| argmax | 832/320 | 91.48% +/- 0.98% | 89.75% +/- 0.93% | 0.74% | 0.00% | 1.98% | 640 | 44608 |
| truth | 672/160 | 86.79% +/- 2.11% | 85.06% +/- 2.11% | 0.86% | 0.00% | 2.19% | 320 | 24608 |
| truth | 832/320 | 89.38% +/- 0.57% | 88.27% +/- 1.19% | 0.99% | 0.00% | 2.08% | 640 | 44608 |
| wiring | 672/160 | 86.79% +/- 2.11% | 85.06% +/- 2.11% | 0.86% | 0.00% | 2.19% | 320 | 24608 |
| wiring | 832/320 | 89.38% +/- 0.57% | 88.27% +/- 1.19% | 0.99% | 0.00% | 2.08% | 640 | 44608 |

## Required metrics (three-seed means)

| method | dataset | state/votes | soft acc | discrete acc | acc gap | soft loss | discrete loss | loss gap | train time (s) | epochs to target | unused | gates | depth | fanout max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bitplane_lut_argmax | sklearn_digits | 672/160 | 86.91% | 88.02% | 1.36% | 1.0314 | 0.9885 | 0.0429 | 16.47 | 7.67 | 2.08% | 320 | 2 | 8.33 |
| bitplane_lut_argmax | sklearn_digits | 832/320 | 91.98% | 91.48% | 0.74% | 0.7207 | 0.6640 | 0.0567 | 18.93 | 6.00 | 1.98% | 640 | 2 | 11.00 |
| bitplane_lut_refit | sklearn_digits | 672/160 | 87.65% | 86.79% | 0.86% | 1.0351 | 0.9897 | 0.0454 | 16.19 | 8.00 | 2.19% | 320 | 2 | 8.33 |
| bitplane_lut_refit | sklearn_digits | 832/320 | 90.37% | 89.38% | 0.99% | 0.7335 | 0.6665 | 0.0670 | 19.05 | 6.00 | 2.08% | 640 | 2 | 11.00 |
| bitplane_lut_wiring_refit | sklearn_digits | 672/160 | 87.65% | 86.79% | 0.86% | 1.0351 | 0.9897 | 0.0454 | 16.32 | 8.00 | 2.19% | 320 | 2 | 8.33 |
| bitplane_lut_wiring_refit | sklearn_digits | 832/320 | 90.37% | 89.38% | 0.99% | 0.7335 | 0.6665 | 0.0670 | 19.23 | 6.00 | 2.08% | 640 | 2 | 11.00 |

## Paired decisions

- `argmax` 320-vote scaling: validation 3.46% and test 4.44% mean delta; 3/3 validation wins.
- `truth` 320-vote scaling: validation 2.59% and test 3.21% mean delta; 3/3 validation wins.
- `wiring` 320-vote scaling: validation 2.59% and test 3.21% mean delta; 3/3 validation wins.
- `truth` versus argmax across both widths: validation -1.67% mean delta; 1/6 wins.
- `wiring` versus argmax across both widths: validation -1.67% mean delta; 1/6 wins.

## Interpretation

The protected bit-plane state prevents depth-wise input erasure and the
learned vote state remains active and jointly diverse. That establishes
that information collapse is avoidable in this bounded setting; it does
not establish CIFAR-scale sufficiency.
For argmax, block-0 to block-1 hard accuracy rises from 80.62% to
88.02% at 160 votes and from 87.65% to 91.48% at 320 votes. Final
vote entropy is about 0.85 bit, inactive vote ratio is zero, and all
270 validation rows retain distinct joint vote states.

Direct hard-ST/argmax is the current winner. Empirical independent LUT
refitting changes truth bits but usually discards task-level vote margin.
Training did learn wiring: 65.27% to
73.67% of exported gate inputs select a candidate
other than the default route. The measured greedy post-refit wiring change
ratio is nevertheless 0.00%, so the wiring-refit
variant is behaviorally identical to truth refit. This rejects the current
coordinate-greedy refitter, not learned discrete wiring itself.

Training times are reported but were collected concurrently across RTX
4090 and H200 devices; they must not be used for a method speed claim.
