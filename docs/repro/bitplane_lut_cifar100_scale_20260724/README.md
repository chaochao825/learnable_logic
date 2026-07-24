# CIFAR-100 A8 bit-plane LUT capacity screen

Architecture selection uses the fixed 5,000-row validation split. The
official test split is report-only. Every row is an argmax-hardened,
per-block-frozen Boolean LUT/wiring payload with fixed integer GroupSum.
Training shadows are real-valued; deployment payloads and audited
execution contain no real-valued tensor or learned numeric matrix.

| variant | soft acc | hard acc | acc gap | soft loss | hard loss | loss gap | train hard | test hard | time | epochs | unused | gates | depth | fanout | vote support | class support | zero-support classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed-v64-d2 | 15.40% | 15.40% | 0.00% | 3.7436 | 3.7271 | 0.0166 | 16.57% | 15.52% | 3407.2s | 19+20 | 7.20% | 12800 | 2 | 12 | 8.4 | 515.0 | 0 |
| spatial-v64-d2 | 16.94% | 17.08% | 0.14% | 3.6245 | 3.6157 | 0.0088 | 18.17% | 16.94% | 4976.4s | 30+30 | 6.65% | 12800 | 2 | 13 | 9.4 | 282.6 | 0 |
| spatial-v128-d2 | 20.18% | 19.82% | 0.36% | 3.4527 | 3.4544 | 0.0016 | 22.21% | 20.37% | 8823.5s | 30+30 | 7.36% | 25600 | 2 | 14 | 9.6 | 597.7 | 0 |
| spatial-v128-d4 | 20.68% | 19.90% | 0.78% | 3.4158 | 3.4327 | 0.0169 | 22.90% | 20.72% | 14492.5s | 30+30+15+26 | 11.43% | 51200 | 4 | 14 | 40.9 | 1421.0 | 0 |
| spatial-v256-d4 | 21.84% | 21.82% | 0.02% | 3.3516 | 3.3324 | 0.0192 | 25.52% | 21.26% | 21303.5s | 30+21+18+17 | 12.44% | 102400 | 4 | 20 | 43.2 | 3208.6 | 0 |

## Registered deltas

| comparison | validation hard delta | training hard delta | test delta (report-only) | gap delta | unused delta | zero-support class delta | gates x | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| routing_spatial_vs_mixed | 1.68% | 1.60% | 1.42% | 0.14% | -0.55% | 0.00% | 1.00 | yes |
| width_128_vs_64_d2 | 2.74% | 4.04% | 3.43% | 0.22% | 0.71% | 0.00% | 2.00 | yes |
| depth_4_vs_2_v128 | 0.08% | 0.70% | 0.35% | 0.42% | 4.07% | 0.00% | 2.00 | yes |
| width_256_vs_128_d4 | 1.92% | 2.62% | 0.54% | -0.76% | 1.01% | 0.00% | 2.00 | yes |

## Screen conclusion

The seed-0 validation winner is `spatial-v256-d4` at 21.82% hard accuracy. Its strict training accuracy is 25.52%, and its soft-hard accuracy gap is only 0.02%. The dominant ceiling is therefore Boolean capacity/routing rather than discretization collapse.
Relative to spatial-v64-d2, the winner uses 8x as many gates for a 4.74 pp validation gain. At fixed v128 width, doubling depth adds only 0.08 pp while unused gates rise by 4.07 pp.
In the winner, each final vote structurally depends on only 43.2 of 24,576 input planes on average (0.18%), and each class aggregates 3208.6 planes (13.06%). Exact truth-table dictionary encoding would cost 114.6% of the raw truth-bit store, so simple table deduplication is not a compression solution.
This is a capacity screen, not a promotion claim; the selected
configuration requires the pre-registered seed-1/2 repeats.
