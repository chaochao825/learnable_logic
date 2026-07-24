# Strict LUT character-model capacity screen

Every row preserves 64 causal A8 character IDs as Boolean planes and
deploys only LUT truth bits, integer source indices, and fixed integer
GroupSum. Perplexity is an offline analysis of integer logits, not a
floating deployment operator.

| variant | soft acc | hard acc | gap | train hard | test hard | top-5 | bpc | time | epochs | unused | inactive | gates | depth | fanout | context support | zero-support classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-mixed-v32-d2 | 23.08% | 25.82% | 2.74% | 28.81% | 25.59% | 60.02% | 3.987 | 122.7s | 4+4 | 2.21% | 19.42% | 4160 | 2 | 18 | 203.7 | 0 |
| smoke-causal-v32-d2 | 25.91% | 28.99% | 3.08% | 31.57% | 28.50% | 61.20% | 3.894 | 123.8s | 4+4 | 1.68% | 17.88% | 4160 | 2 | 100 | 112.8 | 0 |
| causal-v32-d2 | 33.89% | 33.82% | 0.07% | 35.26% | 33.16% | 68.27% | 3.527 | 1271.0s | 20+12 | 2.57% | 18.99% | 4160 | 2 | 127 | 95.7 | 3 |
| causal-v64-d2 | 36.41% | 36.27% | 0.14% | 39.36% | 35.43% | 69.58% | 3.331 | 2728.7s | 20+16 | 4.01% | 7.76% | 8320 | 2 | 182 | 186.2 | 0 |
| causal-v64-d4 | 39.94% | 39.93% | 0.00% | 46.83% | 39.40% | 73.52% | 3.055 | 5309.4s | 20+16+14+20 | 4.60% | 7.21% | 16640 | 4 | 182 | 397.8 | 0 |
| causal-v128-d4 | 40.80% | 40.81% | 0.01% | 51.98% | 39.83% | 73.36% | 3.079 | 8890.2s | 18+18+14+10 | 3.14% | 7.00% | 33280 | 4 | 238 | 490.8 | 0 |

Integer validation references: unigram 15.34%, bigram 26.79%, trigram 38.21%.

## Registered deltas

| comparison | validation hard delta | training hard delta | test delta (report-only) | gap delta | unused delta | zero-support class delta | gates x | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| causal_routing_vs_mixed_smoke | 3.17% | 2.76% | 2.91% | 0.34% | -0.53% | 0.00% | 1.00 | yes |
| width_64_vs_32_d2 | 2.46% | 4.10% | 2.28% | 0.07% | 1.44% | -4.62% | 2.00 | yes |
| depth_4_vs_2_v64 | 3.66% | 7.47% | 3.97% | -0.13% | 0.59% | 0.00% | 2.00 | yes |
| width_128_vs_64_d4 | 0.88% | 5.16% | 0.42% | 0.00% | -1.46% | 0.00% | 2.00 | yes |

## Three-seed depth repeat

The d4 run reuses the exact hardened d2 payload for each seed.
All six deployment payloads pass the zero-real operator audit.

| variant | validation hard mean +/- sd | test hard mean +/- sd | gap mean | unused mean | gates | depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal-v64-d2 | 36.24% +/- 0.05% | 35.49% +/- 0.07% | 0.46% | 3.95% | 8320 | 2 |
| causal-v64-d4 | 40.21% +/- 0.25% | 39.69% +/- 0.33% | 0.10% | 4.68% | 16640 | 4 |

The d4 depth step improves validation hard accuracy by 3.97% on average with 3/3 wins.

## Screen conclusion

The current validation winner is `causal-v128-d4` at 40.81% hard accuracy. The v64 depth effect has a complete three-seed repeat, but the v128 winner remains a seed-0 feasibility result rather than a promotion claim.
