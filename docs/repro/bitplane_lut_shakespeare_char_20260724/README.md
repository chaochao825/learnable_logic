# Strict LUT character-model capacity screen

Every row preserves 64 causal A8 character IDs as Boolean planes and
deploys only LUT truth bits, integer source indices, and fixed integer
GroupSum. Perplexity is an offline analysis of integer logits, not a
floating deployment operator.

| variant | soft acc | hard acc | gap | train hard | test hard | top-5 | bpc | time | epochs | unused | inactive | gates | depth | fanout | context support |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke-mixed-v32-d2 | 23.08% | 25.82% | 2.74% | 28.81% | 25.59% | 60.02% | 3.987 | 122.7s | 4+4 | 2.21% | 19.42% | 4160 | 2 | 18 | 203.7 |
| smoke-causal-v32-d2 | 25.91% | 28.99% | 3.08% | 31.57% | 28.50% | 61.20% | 3.894 | 123.8s | 4+4 | 1.68% | 17.88% | 4160 | 2 | 100 | 112.8 |

Integer validation references: unigram 15.34%, bigram 26.79%, trigram 38.21%.

## Registered deltas

| comparison | validation hard delta | training hard delta | test delta (report-only) | gap delta | unused delta | gates x | success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| causal_routing_vs_mixed_smoke | 3.17% | 2.76% | 2.91% | 0.34% | -0.53% | 1.00 | yes |

## Screen conclusion

The current validation winner is `smoke-causal-v32-d2` at 28.99% hard accuracy. This remains a seed-0 feasibility screen; no language-model method is promoted without the registered seed-1/2 repeats.
