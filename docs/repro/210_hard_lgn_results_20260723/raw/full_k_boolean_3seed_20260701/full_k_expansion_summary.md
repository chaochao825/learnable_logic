# LightLogic Full K-Popcount Expansion Report

This experiment keeps K-expanded popcount values as multi-level hidden wires instead of thresholding every layer back to one bit.

Cost interpretation: `threshold_each_layer` is a pure 1-bit LGN replacement; `full_popcount` is a K-bit/popcount or local-LUT network with K+1 activation levels per hidden wire.

| dataset | seed | K | final_readout | teacher_acc | full_acc | threshold_each_layer_acc | delta | gap_vs_teacher | layer_mae | final_mae | bits/wire | gates |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| majority9 | 0 | 2 | popcount | 1 | 0.71875 | 0.734375 | -0.015625 | 0.28125 | 0.104249 | 0.152235 | 2 | 1024 |
| majority9 | 0 | 2 | final_threshold | 1 | 0.757812 | 0.734375 | 0.0234375 | 0.242188 | 0.11982 | 0.214519 | 2 | 1024 |
| majority9 | 0 | 4 | popcount | 1 | 0.984375 | 0.585938 | 0.398438 | 0.015625 | 0.0439243 | 0.0556869 | 3 | 2048 |
| majority9 | 0 | 4 | final_threshold | 1 | 0.835938 | 0.585938 | 0.25 | 0.164062 | 0.0824117 | 0.209636 | 3 | 2048 |
| majority9 | 0 | 8 | popcount | 1 | 1 | 0.515625 | 0.484375 | 0 | 0.0202104 | 0.0251116 | 4 | 4096 |
| majority9 | 0 | 8 | final_threshold | 1 | 0.71875 | 0.515625 | 0.203125 | 0.28125 | 0.0649903 | 0.204231 | 4 | 4096 |
| majority9 | 0 | 16 | popcount | 1 | 1 | 0.515625 | 0.484375 | 0 | 0.0102968 | 0.013049 | 5 | 8192 |
| majority9 | 0 | 16 | final_threshold | 1 | 0.726562 | 0.515625 | 0.210938 | 0.273438 | 0.0580508 | 0.204065 | 5 | 8192 |
| majority9 | 0 | 32 | popcount | 1 | 1 | 0.507812 | 0.492188 | 0 | 0.00493464 | 0.00606364 | 6 | 16384 |
| majority9 | 0 | 32 | final_threshold | 1 | 0.710938 | 0.507812 | 0.203125 | 0.289062 | 0.0544186 | 0.204 | 6 | 16384 |
| random_sparse10 | 0 | 2 | popcount | 1 | 0.929688 | 0.855469 | 0.0742188 | 0.0703125 | 0.0669936 | 0.110229 | 2 | 1024 |
| random_sparse10 | 0 | 2 | final_threshold | 1 | 0.851562 | 0.855469 | -0.00390625 | 0.148438 | 0.0900706 | 0.202537 | 2 | 1024 |
| random_sparse10 | 0 | 4 | popcount | 1 | 0.996094 | 0.820312 | 0.175781 | 0.00390625 | 0.0396974 | 0.0614261 | 3 | 2048 |
| random_sparse10 | 0 | 4 | final_threshold | 1 | 0.859375 | 0.820312 | 0.0390625 | 0.140625 | 0.0735752 | 0.196938 | 3 | 2048 |
| random_sparse10 | 0 | 8 | popcount | 1 | 1 | 0.8125 | 0.1875 | 0 | 0.0212857 | 0.0308038 | 4 | 4096 |
| random_sparse10 | 0 | 8 | final_threshold | 1 | 0.878906 | 0.8125 | 0.0664062 | 0.121094 | 0.062458 | 0.195493 | 4 | 4096 |
| random_sparse10 | 0 | 16 | popcount | 1 | 1 | 0.792969 | 0.207031 | 0 | 0.0100407 | 0.0144936 | 5 | 8192 |
| random_sparse10 | 0 | 16 | final_threshold | 1 | 0.886719 | 0.792969 | 0.09375 | 0.113281 | 0.0552267 | 0.195238 | 5 | 8192 |
| random_sparse10 | 0 | 32 | popcount | 1 | 1 | 0.792969 | 0.207031 | 0 | 0.00559641 | 0.00782859 | 6 | 16384 |
| random_sparse10 | 0 | 32 | final_threshold | 1 | 0.898438 | 0.792969 | 0.105469 | 0.101562 | 0.0524432 | 0.195216 | 6 | 16384 |
| majority9 | 1 | 2 | popcount | 1 | 0.703125 | 0.585938 | 0.117188 | 0.296875 | 0.113717 | 0.16871 | 2 | 1024 |
| majority9 | 1 | 2 | final_threshold | 1 | 0.6875 | 0.585938 | 0.101562 | 0.3125 | 0.129037 | 0.229987 | 2 | 1024 |
| majority9 | 1 | 4 | popcount | 1 | 1 | 0.523438 | 0.476562 | 0 | 0.045704 | 0.0577218 | 3 | 2048 |
| majority9 | 1 | 4 | final_threshold | 1 | 0.859375 | 0.523438 | 0.335938 | 0.140625 | 0.0880312 | 0.227031 | 3 | 2048 |
| majority9 | 1 | 8 | popcount | 1 | 1 | 0.484375 | 0.515625 | 0 | 0.0202166 | 0.0263663 | 4 | 4096 |
| majority9 | 1 | 8 | final_threshold | 1 | 0.726562 | 0.484375 | 0.242188 | 0.273438 | 0.0691449 | 0.222079 | 4 | 4096 |
| majority9 | 1 | 16 | popcount | 1 | 1 | 0.484375 | 0.515625 | 0 | 0.0112952 | 0.0148431 | 5 | 8192 |
| majority9 | 1 | 16 | final_threshold | 1 | 0.703125 | 0.484375 | 0.21875 | 0.296875 | 0.0630009 | 0.221666 | 5 | 8192 |
| majority9 | 1 | 32 | popcount | 1 | 1 | 0.484375 | 0.515625 | 0 | 0.0053626 | 0.00689023 | 6 | 16384 |
| majority9 | 1 | 32 | final_threshold | 1 | 0.710938 | 0.484375 | 0.226562 | 0.289062 | 0.0590479 | 0.221631 | 6 | 16384 |
| random_sparse10 | 1 | 2 | popcount | 1 | 0.875 | 0.792969 | 0.0820312 | 0.125 | 0.0753496 | 0.120516 | 2 | 1024 |
| random_sparse10 | 1 | 2 | final_threshold | 1 | 0.796875 | 0.792969 | 0.00390625 | 0.203125 | 0.093142 | 0.191686 | 2 | 1024 |
| random_sparse10 | 1 | 4 | popcount | 1 | 0.964844 | 0.761719 | 0.203125 | 0.0351562 | 0.0490447 | 0.0748582 | 3 | 2048 |
| random_sparse10 | 1 | 4 | final_threshold | 1 | 0.832031 | 0.761719 | 0.0703125 | 0.167969 | 0.0768694 | 0.186157 | 3 | 2048 |
| random_sparse10 | 1 | 8 | popcount | 1 | 1 | 0.800781 | 0.199219 | 0 | 0.0232735 | 0.0307783 | 4 | 4096 |
| random_sparse10 | 1 | 8 | final_threshold | 1 | 0.832031 | 0.800781 | 0.03125 | 0.167969 | 0.0617746 | 0.184783 | 4 | 4096 |
| random_sparse10 | 1 | 16 | popcount | 1 | 1 | 0.792969 | 0.207031 | 0 | 0.0105453 | 0.0141863 | 5 | 8192 |
| random_sparse10 | 1 | 16 | final_threshold | 1 | 0.832031 | 0.792969 | 0.0390625 | 0.167969 | 0.0531788 | 0.18472 | 5 | 8192 |
| random_sparse10 | 1 | 32 | popcount | 1 | 1 | 0.792969 | 0.207031 | 0 | 0.00543513 | 0.00734636 | 6 | 16384 |
| random_sparse10 | 1 | 32 | final_threshold | 1 | 0.832031 | 0.792969 | 0.0390625 | 0.167969 | 0.0497701 | 0.184686 | 6 | 16384 |
| majority9 | 2 | 2 | popcount | 1 | 0.75 | 0.742188 | 0.0078125 | 0.25 | 0.0958513 | 0.15495 | 2 | 1024 |
| majority9 | 2 | 2 | final_threshold | 1 | 0.75 | 0.742188 | 0.0078125 | 0.25 | 0.109737 | 0.210491 | 2 | 1024 |
| majority9 | 2 | 4 | popcount | 1 | 0.96875 | 0.585938 | 0.382812 | 0.03125 | 0.0501557 | 0.0713673 | 3 | 2048 |
| majority9 | 2 | 4 | final_threshold | 1 | 0.8125 | 0.585938 | 0.226562 | 0.1875 | 0.0825084 | 0.200778 | 3 | 2048 |
| majority9 | 2 | 8 | popcount | 1 | 1 | 0.585938 | 0.414062 | 0 | 0.0195409 | 0.0262317 | 4 | 4096 |
| majority9 | 2 | 8 | final_threshold | 1 | 0.703125 | 0.585938 | 0.117188 | 0.296875 | 0.062297 | 0.197256 | 4 | 4096 |
| majority9 | 2 | 16 | popcount | 1 | 1 | 0.585938 | 0.414062 | 0 | 0.0103058 | 0.0138466 | 5 | 8192 |
| majority9 | 2 | 16 | final_threshold | 1 | 0.703125 | 0.585938 | 0.117188 | 0.296875 | 0.056045 | 0.196804 | 5 | 8192 |
| majority9 | 2 | 32 | popcount | 1 | 1 | 0.585938 | 0.414062 | 0 | 0.00523192 | 0.00678858 | 6 | 16384 |
| majority9 | 2 | 32 | final_threshold | 1 | 0.6875 | 0.585938 | 0.101562 | 0.3125 | 0.0526945 | 0.196639 | 6 | 16384 |
| random_sparse10 | 2 | 2 | popcount | 1 | 0.957031 | 0.933594 | 0.0234375 | 0.0429688 | 0.0656049 | 0.106343 | 2 | 1024 |
| random_sparse10 | 2 | 2 | final_threshold | 1 | 0.925781 | 0.933594 | -0.0078125 | 0.0742188 | 0.087226 | 0.192828 | 2 | 1024 |
| random_sparse10 | 2 | 4 | popcount | 1 | 0.996094 | 0.910156 | 0.0859375 | 0.00390625 | 0.0393163 | 0.059899 | 3 | 2048 |
| random_sparse10 | 2 | 4 | final_threshold | 1 | 0.898438 | 0.910156 | -0.0117188 | 0.101562 | 0.0697863 | 0.181779 | 3 | 2048 |
| random_sparse10 | 2 | 8 | popcount | 1 | 1 | 0.925781 | 0.0742188 | 0 | 0.0219557 | 0.0309966 | 4 | 4096 |
| random_sparse10 | 2 | 8 | final_threshold | 1 | 0.894531 | 0.925781 | -0.03125 | 0.105469 | 0.0594088 | 0.180809 | 4 | 4096 |
| random_sparse10 | 2 | 16 | popcount | 1 | 1 | 0.925781 | 0.0742188 | 0 | 0.0108618 | 0.0150781 | 5 | 8192 |
| random_sparse10 | 2 | 16 | final_threshold | 1 | 0.894531 | 0.925781 | -0.03125 | 0.105469 | 0.0522523 | 0.18064 | 5 | 8192 |
| random_sparse10 | 2 | 32 | popcount | 1 | 1 | 0.925781 | 0.0742188 | 0 | 0.00544486 | 0.0074544 | 6 | 16384 |
| random_sparse10 | 2 | 32 | final_threshold | 1 | 0.894531 | 0.925781 | -0.03125 | 0.105469 | 0.0487148 | 0.180534 | 6 | 16384 |

Notes:
- `full_acc` evaluates the quantized multi-level network, not a pure one-bit hard network.
- `final_readout=popcount` sends multi-level final layer outputs directly to GroupSum.
- `final_readout=final_threshold` keeps hidden layers multi-level and thresholds only the last layer before GroupSum.
- `threshold_each_layer_acc` is the old K-expanded path with a 1-bit conversion after every layer.
- A positive `delta` isolates the benefit of preserving multi-level information across depth.
