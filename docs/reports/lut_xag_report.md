# LUT XAG Backend Report

This report adds a canonical ANF/XAG-style local backend on top of existing LUT artifacts. It complements the AIG/ABC `and/lev` numbers with local `AND/XOR/NOT` estimates.

- combined network rows: 235
- unique truth tables analyzed: 4969

## Summary By Run/Dataset

| run | dataset | rows | xag_and_mean | xag_xor_mean | xag_not_mean | xag_level_mean | aig_and_mean | aig_level_mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 15 | 21.4 | 24.8667 | 12.6667 | 10.8 | 11.1333 | 1.66667 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 15 | 7.8 | 12.3333 | 3.13333 | 8.4 | 3.66667 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 15 | 64.9333 | 95.6667 | 41.6 | 12 | 77 | 5.06667 |
| lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1 | majority9 | 15 | 20.3333 | 17.6 | 5.33333 | 8.73333 | 10.7333 | 1.6 |
| lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1 | parity8 | 15 | 7.6 | 9.2 | 1.66667 | 7.93333 | 2.86667 | 1.53333 |
| lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1 | random_sparse10 | 15 | 62.4 | 84.4667 | 35.8667 | 12 | 73.3333 | 5.2 |
| lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1 | majority9 | 15 | 25.4667 | 33.6 | 7.13333 | 11.4667 | 8.8 | 1.53333 |
| lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1 | parity8 | 15 | 15.7333 | 26.1333 | 1.93333 | 10.9333 | 2.86667 | 1.46667 |
| lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1 | random_sparse10 | 15 | 67.3333 | 94.4 | 37 | 12 | 69.6667 | 5 |
| lightlogic_lut_min_goal8_digits_multiseed_v1 | digits | 15 | 496 | 667.667 | 172.867 | 13.2667 | 556.4 | 6 |
| lightlogic_lut_min_goal8_digits_per_layer_v1 | digits | 15 | 508.8 | 601.4 | 131.933 | 13.6 | 570.2 | 6 |
| lightlogic_lut_min_goal8_digits_per_neuron_v1 | digits | 15 | 520.6 | 664.2 | 138.8 | 13.4667 | 579.2 | 6 |
| lightlogic_lut_min_goal8_mnist_scale_v1 | binarized_mnist | 5 | 1073.2 | 1596.8 | 387.4 | 12 | 1608.8 | 6 |
| lightlogic_lut_min_goal8_mnist_per_layer_v1 | binarized_mnist | 5 | 1079 | 1450 | 290.6 | 12 | 1567.6 | 6 |
| lightlogic_lut_min_goal8_cifar10_scale_v1 | cifar10_small | 5 | 1594.4 | 2082.4 | 542 | 9.8 | 1771 | 5 |
| lightlogic_lut_min_goal8_cifar10_per_layer_v1 | cifar10_small | 5 | 1686.2 | 1509 | 268.8 | 10 | 1796.4 | 5.2 |
| lightlogic_b_lut_min_goal7_bool_multiseed_v1 | majority9 | 6 | 1169.33 | 1080 | 157.167 | 26 | 652.5 | 11.8333 |
| lightlogic_b_lut_min_goal7_bool_multiseed_v1 | parity8 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| lightlogic_b_lut_min_goal7_bool_multiseed_v1 | random_sparse10 | 6 | 1264.67 | 1209 | 101.333 | 25.8333 | 725.167 | 12.3333 |
| lightlogic_b_lut_min_goal7_digits_multiseed_v1 | digits | 6 | 2745.33 | 2468.33 | 150.5 | 26 | 1691 | 12.6667 |
| lightlogic_b_lut_min_goal7_mnist_seed0_v1 | binarized_mnist | 2 | 3795.5 | 3612.5 | 299.5 | 26 | 2773 | 15 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_v1 | binarized_mnist | 1 | 2639 | 2811 | 425 | 24 | 2337 | 12 |
| lightlogic_b_lut_min_goal7_mnist_b4_w360_v1 | binarized_mnist | 1 | 8422 | 7729 | 510 | 28 | 6296 | 19 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_e150_v1 | binarized_mnist | 1 | 2680 | 2824 | 513 | 24 | 2485 | 11 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_warmup_v1 | binarized_mnist | 1 | 2700 | 2908 | 382 | 24 | 2391 | 11 |
| lightlogic_b_lut_min_goal7_mnist_b4_w360_warmup_v1 | binarized_mnist | 1 | 8601 | 7834 | 473 | 28 | 6374 | 18 |
| lightlogic_b_lut_min_goal7_mnist_matched_b3_v1 | binarized_mnist | 1 | 4526 | 4802 | 840 | 18 | 4125 | 10 |
| lightlogic_b_lut_min_goal7_mnist_matched_b4_v1 | binarized_mnist | 1 | 14081 | 13036 | 849 | 21 | 9977 | 14 |
| lightlogic_b_lut_min_goal7_mnist_matched_b3_multiseed_v1 | binarized_mnist | 2 | 4477.5 | 4692 | 801 | 18 | 4102 | 10 |

## Example Network Rows

| run | dataset | seed | key | xag_local_and_count | xag_local_xor_count | xag_local_not_count | xag_local_level_estimate | abc_and_count | abc_level |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 0 | k=2 | 23 | 40 | 6 | 9 | 5 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 0 | k=4 | 8 | 12 | 2 | 9 | 7 | 4 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 0 | k=8 | 5 | 6 | 0 | 5 | 5 | 4 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 0 | k=16 | 3 | 2 | 0 | 3 | 3 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 0 | k=32 | 3 | 2 | 0 | 3 | 3 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 0 | k=2 | 66 | 86 | 44 | 12 | 41 | 3 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 0 | k=4 | 20 | 20 | 12 | 12 | 11 | 3 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 0 | k=8 | 13 | 12 | 6 | 7 | 6 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 0 | k=16 | 11 | 9 | 2 | 9 | 2 | 1 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 0 | k=32 | 9 | 6 | 1 | 6 | 0 | 0 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 0 | k=2 | 86 | 141 | 52 | 12 | 92 | 6 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 0 | k=4 | 67 | 105 | 47 | 12 | 90 | 6 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 0 | k=8 | 57 | 88 | 38 | 12 | 84 | 6 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 0 | k=16 | 53 | 78 | 35 | 12 | 73 | 5 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | random_sparse10 | 0 | k=32 | 53 | 78 | 35 | 12 | 73 | 5 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 1 | k=2 | 20 | 30 | 10 | 12 | 11 | 3 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 1 | k=4 | 7 | 10 | 5 | 12 | 4 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 1 | k=8 | 4 | 6 | 3 | 9 | 2 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 1 | k=16 | 5 | 6 | 3 | 8 | 3 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | parity8 | 1 | k=32 | 5 | 6 | 3 | 8 | 3 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 1 | k=2 | 66 | 85 | 45 | 12 | 35 | 3 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 1 | k=4 | 20 | 24 | 12 | 12 | 7 | 2 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 1 | k=8 | 7 | 5 | 6 | 12 | 2 | 1 |
| lightlogic_lut_min_goal8_bool_multiseed_fixed_v1 | majority9 | 1 | k=16 | 9 | 6 | 5 | 11 | 2 | 1 |

Notes:
- `anf_xag` is an exact canonical ANF decomposition of each local LUT truth table, then a shared-product XAG estimate inside that LUT.
- `xag_local_*` numbers are local per-LUT sums; they do not claim global cross-LUT XAG optimization.
- `abc_and_count` / `abc_level` remain the global AIG/ABC numbers from the existing Goal 8 flow.
- `xag_not_count` uses output inversion when the ANF has a constant-one bias term.
