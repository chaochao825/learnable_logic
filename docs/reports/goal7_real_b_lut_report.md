# Real b-Input LUT Network Report

This report aggregates the direct b-input LUT training runs and compares them against the earlier Goal 8 b=2 LightLogic/K-expansion baseline.

- combined rows: 36
- dataset/b groups: 10
- direct run configs summarized: 12
- baseline datasets imported from `reports/goal8_large_scale_v1/best_by_dataset.csv`: 5

## Comparison Protocol

- Direct b-input candidates are loaded from the run directories listed in this report's `config.json`.
- For each `(dataset, b)` pair, the report keeps the row with the highest `discrete_acc` from `network_minimization_results.csv`.
- For each dataset, the final comparison uses the best direct `b>2` row against the dataset-specific Goal 8 `b=2` baseline imported from `reports/goal8_large_scale_v1/best_by_dataset.csv`.
- The baseline rows are reused from the prior Goal 8 selection and are not retrained inside this report; the baseline section below documents the exact source run and config behind each dataset.

## Experiment Configurations

| run | datasets | b_values | seeds | width | layers | epochs | train/test | train_mode | warmup | distill | device | rows |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---:|
| lightlogic_b_lut_min_goal7_bool_multiseed_v1 | parity8,majority9,random_sparse10 | 3,4 | 0,1,2 | 160 | 4 | 60 | 4000/1000 | st | off | off | cpu | 18 |
| lightlogic_b_lut_min_goal7_digits_multiseed_v1 | digits | 3,4 | 0,1,2 | 240 | 4 | 60 | 4000/1000 | st | off | off | cuda | 6 |
| lightlogic_b_lut_min_goal7_mnist_seed0_v1 | binarized_mnist | 3,4 | 0 | 240 | 4 | 60 | 4000/1000 | st | off | off | cuda | 2 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_v1 | binarized_mnist | 3 | 0 | 360 | 4 | 90 | 4000/1000 | st | off | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_b4_w360_v1 | binarized_mnist | 4 | 0 | 360 | 4 | 90 | 4000/1000 | st | off | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_e150_v1 | binarized_mnist | 3 | 0 | 360 | 4 | 150 | 4000/1000 | st | off | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_b3_w360_warmup_v1 | binarized_mnist | 3 | 0 | 360 | 4 | 90 | 4000/1000 | st | 30ep:continuous | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_b4_w360_warmup_v1 | binarized_mnist | 4 | 0 | 360 | 4 | 90 | 4000/1000 | st | 30ep:continuous | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_b3_distill_v1 | binarized_mnist | 3 | 0 | 360 | 4 | 90 | 4000/1000 | st | off | w=0.5,tau=2,teacher=128x0/60 | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_matched_b3_v1 | binarized_mnist | 3 | 0 | 800 | 3 | 80 | 12000/3000 | st | off | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_matched_b4_v1 | binarized_mnist | 4 | 0 | 800 | 3 | 80 | 12000/3000 | st | off | off | cuda | 1 |
| lightlogic_b_lut_min_goal7_mnist_matched_b3_multiseed_v1 | binarized_mnist | 3 | 1,2 | 800 | 3 | 80 | 12000/3000 | st | off | off | cuda | 2 |

## Baseline Definition

| dataset | baseline_source_run | baseline_run_dir | calib | seed | k | width | layers | epochs | train/test | baseline_raw_lut_acc | baseline_abc_and_count | note |
|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| binarized_mnist | mnist_fixed | `runs/lightlogic_lut_min_goal8_mnist_scale_v1` | fixed | 0 | 4 | 800 | 3 | 80 | 12000/3000 | 0.786667 | 1613 | matched on width/layers/epochs/train/test |
| digits | digits_fixed | `runs/lightlogic_lut_min_goal8_digits_multiseed_v1` | fixed | 1 | 4 | 320 | 4 | 150 | 4000/1000 | 0.922222 | 547 | best Goal 8 row; differs on width:320->240, epochs:150->60 |
| majority9 | bool_per_neuron | `runs/lightlogic_lut_min_goal8_bool_multiseed_per_neuron_v1` | per_neuron | 2 | 2 | 128 | 4 | 120 | 4000/1000 | 0.742188 | 37 | best Goal 8 row; differs on width:128->160, epochs:120->60 |
| parity8 | bool_fixed | `runs/lightlogic_lut_min_goal8_bool_multiseed_fixed_v1` | fixed | 1 | 8 | 128 | 4 | 120 | 4000/1000 | 0.53125 | 2 | best Goal 8 row; differs on width:128->160, epochs:120->60 |
| random_sparse10 | bool_per_layer | `runs/lightlogic_lut_min_goal8_bool_multiseed_per_layer_v1` | per_layer | 2 | 2 | 128 | 4 | 120 | 4000/1000 | 0.9375 | 100 | best Goal 8 row; differs on width:128->160, epochs:120->60 |

## Best Direct b-Input Results

| dataset | b | run | seed | width | layers | epochs | train/test | best_discrete_acc | best_minimized_blif_acc | best_abc_and_count | best_abc_level | best_raw_lut_gate_estimate |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| binarized_mnist | 3 | lightlogic_b_lut_min_goal7_mnist_matched_b3_v1 | 0 | 800 | 3 | 80 | 12000/3000 | 0.835667 | 0.835667 | 4125 | 10 | 16800 |
| binarized_mnist | 4 | lightlogic_b_lut_min_goal7_mnist_matched_b4_v1 | 0 | 800 | 3 | 80 | 12000/3000 | 0.814333 | 0.814333 | 9977 | 14 | 36000 |
| digits | 3 | lightlogic_b_lut_min_goal7_digits_multiseed_v1 | 1 | 240 | 4 | 60 | 4000/1000 | 0.895556 | 0.895556 | 1216 | 10 | 6720 |
| digits | 4 | lightlogic_b_lut_min_goal7_digits_multiseed_v1 | 1 | 240 | 4 | 60 | 4000/1000 | 0.917778 | 0.917778 | 2224 | 14 | 14400 |
| majority9 | 3 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 1 | 160 | 4 | 60 | 4000/1000 | 0.828125 | 0.828125 | 413 | 10 | 4480 |
| majority9 | 4 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 0 | 160 | 4 | 60 | 4000/1000 | 0.90625 | 0.90625 | 784 | 14 | 9600 |
| parity8 | 3 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 1 | 160 | 4 | 60 | 4000/1000 | 0.53125 | 0.53125 | 0 | 0 | 4480 |
| parity8 | 4 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 2 | 160 | 4 | 60 | 4000/1000 | 0.53125 | 0.53125 | 0 | 0 | 9600 |
| random_sparse10 | 3 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 0 | 160 | 4 | 60 | 4000/1000 | 0.972656 | 0.972656 | 468 | 10 | 4480 |
| random_sparse10 | 4 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 2 | 160 | 4 | 60 | 4000/1000 | 0.957031 | 0.957031 | 915 | 15 | 9600 |

## Comparison Against Prior b=2 Baseline

| dataset | best_new_b | best_new_run | best_new_discrete_acc | baseline_source_run | baseline_raw_lut_acc | delta_acc | best_new_abc_and_count | baseline_abc_and_count | delta_abc_and | best_new_raw_lut_gate_estimate | baseline_raw_lut_gate_estimate |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| binarized_mnist | 3 | lightlogic_b_lut_min_goal7_mnist_matched_b3_v1 | 0.835667 | mnist_fixed | 0.786667 | 0.049 | 4125 | 1613 | 2512 | 16800 | 7200 |
| digits | 4 | lightlogic_b_lut_min_goal7_digits_multiseed_v1 | 0.917778 | digits_fixed | 0.922222 | -0.00444444 | 2224 | 547 | 1677 | 14400 | 3840 |
| majority9 | 4 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 0.90625 | bool_per_neuron | 0.742188 | 0.164062 | 784 | 37 | 747 | 9600 | 1536 |
| parity8 | 3 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 0.53125 | bool_fixed | 0.53125 | 0 | 0 | 2 | -2 | 4480 | 1536 |
| random_sparse10 | 3 | lightlogic_b_lut_min_goal7_bool_multiseed_v1 | 0.972656 | bool_per_layer | 0.9375 | 0.0351562 | 468 | 100 | 368 | 4480 | 1536 |

Interpretation:
- Accuracy preservation through BLIF export and ABC minimization holds across the aggregated direct b-input runs in this report.
- The strongest closure of the original evidence gap is matched-setting binarized MNIST: direct `b=3` at `width=800`, `layers=3`, `train/test=12000/3000`, `epochs=80` beats the earlier matched Goal 8 `b=2` baseline by `+0.049` absolute accuracy.
- Digits direct `b=4` is slightly below the imported Goal 8 baseline, but that baseline uses a stronger `width=320`, `epochs=150` configuration while the direct b-input run here uses `width=240`, `epochs=60`; this should be read as feasibility evidence rather than a clean negative result.
- Boolean `majority9` and `random_sparse10` improve accuracy over their imported Goal 8 baselines, while `parity8` remains flat at `0.53125` and therefore does not show a direct-b accuracy gain yet.
- The cost tradeoff remains real: the direct b-input winners generally require substantially larger raw LUT gate estimates and larger final ABC AIG counts than the best Goal 8 b=2 baselines.
- This report stays AIG/ABC-focused; the separate `reports/lut_xag_backend_v1` report provides local AND/XOR/NOT XAG estimates for the LUT functions.
