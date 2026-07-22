# LightLogic Full K-Popcount Expansion Report

This experiment keeps K-expanded popcount values as multi-level hidden wires instead of thresholding every layer back to one bit.

Cost interpretation: `threshold_each_layer` is a pure 1-bit LGN replacement; `full_popcount` is a K-bit/popcount or local-LUT network with K+1 activation levels per hidden wire.

| dataset | seed | K | final_readout | teacher_acc | full_acc | threshold_each_layer_acc | delta | gap_vs_teacher | layer_mae | final_mae | bits/wire | gates |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| binarized_mnist | 0 | 4 | popcount | 0.877 | 0.869 | 0.782 | 0.087 | 0.008 | 0.0306285 | 0.0453343 | 3 | 6400 |
| binarized_mnist | 0 | 4 | final_threshold | 0.877 | 0.808 | 0.782 | 0.026 | 0.069 | 0.0665264 | 0.188926 | 3 | 6400 |
| binarized_mnist | 0 | 8 | popcount | 0.877 | 0.872 | 0.781 | 0.091 | 0.005 | 0.0162095 | 0.0235959 | 4 | 12800 |
| binarized_mnist | 0 | 8 | final_threshold | 0.877 | 0.819 | 0.781 | 0.038 | 0.058 | 0.0574625 | 0.188608 | 4 | 12800 |
| binarized_mnist | 0 | 16 | popcount | 0.877 | 0.876 | 0.78 | 0.096 | 0.001 | 0.00831828 | 0.0121221 | 5 | 25600 |
| binarized_mnist | 0 | 16 | final_threshold | 0.877 | 0.818 | 0.78 | 0.038 | 0.059 | 0.0524191 | 0.188525 | 5 | 25600 |
| binarized_mnist | 0 | 32 | popcount | 0.877 | 0.874 | 0.78 | 0.094 | 0.003 | 0.00441273 | 0.00601391 | 6 | 51200 |
| binarized_mnist | 0 | 32 | final_threshold | 0.877 | 0.818 | 0.78 | 0.038 | 0.059 | 0.0500372 | 0.188512 | 6 | 51200 |

Notes:
- `full_acc` evaluates the quantized multi-level network, not a pure one-bit hard network.
- `final_readout=popcount` sends multi-level final layer outputs directly to GroupSum.
- `final_readout=final_threshold` keeps hidden layers multi-level and thresholds only the last layer before GroupSum.
- `threshold_each_layer_acc` is the old K-expanded path with a 1-bit conversion after every layer.
- A positive `delta` isolates the benefit of preserving multi-level information across depth.
