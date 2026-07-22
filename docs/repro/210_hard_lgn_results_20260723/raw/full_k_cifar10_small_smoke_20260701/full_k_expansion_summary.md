# LightLogic Full K-Popcount Expansion Report

This experiment keeps K-expanded popcount values as multi-level hidden wires instead of thresholding every layer back to one bit.

Cost interpretation: `threshold_each_layer` is a pure 1-bit LGN replacement; `full_popcount` is a K-bit/popcount or local-LUT network with K+1 activation levels per hidden wire.

| dataset | seed | K | final_readout | teacher_acc | full_acc | threshold_each_layer_acc | delta | gap_vs_teacher | layer_mae | final_mae | bits/wire | gates |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cifar10_small | 0 | 8 | popcount | 0.317 | 0.312 | 0.166 | 0.146 | 0.005 | 0.0238121 | 0.0298248 | 4 | 49280 |
| cifar10_small | 0 | 8 | final_threshold | 0.317 | 0.225 | 0.166 | 0.059 | 0.092 | 0.0948251 | 0.313877 | 4 | 49280 |
| cifar10_small | 0 | 16 | popcount | 0.317 | 0.308 | 0.165 | 0.143 | 0.009 | 0.0119701 | 0.0143988 | 5 | 98560 |
| cifar10_small | 0 | 16 | final_threshold | 0.317 | 0.23 | 0.165 | 0.065 | 0.087 | 0.0867597 | 0.313557 | 5 | 98560 |
| cifar10_small | 0 | 32 | popcount | 0.317 | 0.309 | 0.165 | 0.144 | 0.008 | 0.00603678 | 0.0072113 | 6 | 197120 |
| cifar10_small | 0 | 32 | final_threshold | 0.317 | 0.227 | 0.165 | 0.062 | 0.09 | 0.0825973 | 0.313453 | 6 | 197120 |

Notes:
- `full_acc` evaluates the quantized multi-level network, not a pure one-bit hard network.
- `final_readout=popcount` sends multi-level final layer outputs directly to GroupSum.
- `final_readout=final_threshold` keeps hidden layers multi-level and thresholds only the last layer before GroupSum.
- `threshold_each_layer_acc` is the old K-expanded path with a 1-bit conversion after every layer.
- A positive `delta` isolates the benefit of preserving multi-level information across depth.
