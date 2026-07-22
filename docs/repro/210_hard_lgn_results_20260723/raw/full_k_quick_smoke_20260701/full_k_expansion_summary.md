# LightLogic Full K-Popcount Expansion Report

This experiment keeps K-expanded popcount values as multi-level hidden wires instead of thresholding every layer back to one bit.

Cost interpretation: `threshold_each_layer` is a pure 1-bit LGN replacement; `full_popcount` is a K-bit/popcount or local-LUT network with K+1 activation levels per hidden wire.

| dataset | seed | K | final_readout | teacher_acc | full_acc | threshold_each_layer_acc | delta | gap_vs_teacher | layer_mae | final_mae | bits/wire | gates |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parity6 | 0 | 2 | popcount | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 2 | 192 |
| parity6 | 0 | 2 | final_threshold | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 2 | 192 |
| parity6 | 0 | 4 | popcount | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 3 | 384 |
| parity6 | 0 | 4 | final_threshold | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 3 | 384 |
| parity6 | 0 | 8 | popcount | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 4 | 768 |
| parity6 | 0 | 8 | final_threshold | 0.25 | 0.5625 | 0.5625 | 0 | 0.3125 | 0.0434449 | 0.066779 | 4 | 768 |
| majority7 | 0 | 2 | popcount | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 2 | 192 |
| majority7 | 0 | 2 | final_threshold | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 2 | 192 |
| majority7 | 0 | 4 | popcount | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 3 | 384 |
| majority7 | 0 | 4 | final_threshold | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 3 | 384 |
| majority7 | 0 | 8 | popcount | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 4 | 768 |
| majority7 | 0 | 8 | final_threshold | 0.4375 | 0.4375 | 0.4375 | 0 | 0 | 0.0400635 | 0.0619084 | 4 | 768 |
| random_sparse8 | 0 | 2 | popcount | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 2 | 192 |
| random_sparse8 | 0 | 2 | final_threshold | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 2 | 192 |
| random_sparse8 | 0 | 4 | popcount | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 3 | 384 |
| random_sparse8 | 0 | 4 | final_threshold | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 3 | 384 |
| random_sparse8 | 0 | 8 | popcount | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 4 | 768 |
| random_sparse8 | 0 | 8 | final_threshold | 0.4375 | 0.40625 | 0.40625 | 0 | 0.03125 | 0.0484718 | 0.0744421 | 4 | 768 |

Notes:
- `full_acc` evaluates the quantized multi-level network, not a pure one-bit hard network.
- `final_readout=popcount` sends multi-level final layer outputs directly to GroupSum.
- `final_readout=final_threshold` keeps hidden layers multi-level and thresholds only the last layer before GroupSum.
- `threshold_each_layer_acc` is the old K-expanded path with a 1-bit conversion after every layer.
- A positive `delta` isolates the benefit of preserving multi-level information across depth.
