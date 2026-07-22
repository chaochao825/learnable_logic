# Hard-LGN Gap Prototype Summary

Elapsed seconds: 98.21

Metric notes:

- `soft_acc`/`soft_loss` evaluate all trained relaxed blocks in soft mode.
- `discrete_acc`/`discrete_loss` evaluate the corresponding hard network: argmax gates for DLGN-style methods and fitted gates for block-hard methods.
- `path_*` metrics evaluate the method-native training path. Hard-ST uses the deterministic hard forward; block-hard methods use hard frozen prefixes plus the final relaxed block.
- `unused_gate_ratio` follows Mind the Gap: logit entropy above the deterministic N(0,1) initialization 2.5% threshold (1.884315).
- `activation_inactive_gate_ratio` retains the older data-dependent statistic: hard gate outputs that are constant on the training set.
- `layer_diagnostics.csv` tracks relaxed-path versus hard-path representation mismatch after each layer prefix for depth-wise gap accumulation checks.
- `*_inference_samples_per_sec` are optional PyTorch forward-pass throughput measurements from `--inference-bench`; they are not bit-packed Boolean inference kernels.
- For block-wise methods, `epochs_to_target` is conservative: total block epochs if the final hard model reaches the dataset target, otherwise `-1`.
- `time_to_target` is wall-clock seconds to first discrete target hit for end-to-end methods; for block-wise methods it is conservative final train time if the final hard model reaches target, otherwise `-1`.

Configuration:

```json
{
  "abc_max_gates": 10000,
  "abc_path": "/home/spco/boolean_sat/abc/abc",
  "abc_stats": true,
  "batch_size": 128,
  "block_epochs": 40,
  "block_total_epochs": 120,
  "cage_beta": 0.99,
  "cage_tau_max": 3.0,
  "cage_tau_min": 0.5,
  "compat_only": false,
  "data_dir": "/home/spco/data",
  "datasets": [
    "parity8",
    "majority9",
    "random_sparse10"
  ],
  "device": "cpu",
  "difflogic_compat_check": false,
  "download_data": false,
  "entropy_coef": 0.001,
  "epochs": 120,
  "eval_batch_size": 2048,
  "exact_truth_max": 12,
  "group_tau": 0.01,
  "gumbel_temp_end": 1.0,
  "gumbel_temp_start": 1.0,
  "hard_st_temp": 1.0,
  "image_max_test": 1000,
  "image_max_train": 4000,
  "inference_bench": false,
  "inference_bench_device": "cpu",
  "inference_bench_repeats": 20,
  "inference_bench_warmup": 1,
  "layers": 4,
  "lr": 0.01,
  "methods": [
    "dlgn",
    "dlgn_anneal",
    "gumbel_st",
    "hard_st_cage",
    "block_relaxed",
    "block_hard_refit",
    "block_hard_task_refit"
  ],
  "mind_gap_scaled": true,
  "optimizer": "adam",
  "out_dir": "runs/mind_gap_scaled_bool_abc_w128_d4_e120_s0_20260723",
  "quick": false,
  "refit_candidate_topk": 16,
  "refit_coordinate_passes": 2,
  "refit_distill_weight": 0.25,
  "refit_inactive_weight": 0.01,
  "refit_local_weight": 0.05,
  "refit_samples": 4096,
  "refit_seed": 12345,
  "refit_selection_mode": "balanced",
  "refit_validation_fraction": 0.2,
  "seeds": [
    0
  ],
  "target_acc_override": null,
  "temp_end": 0.2,
  "temp_start": 2.0,
  "threshold_levels": 1,
  "weight_decay": 0.0,
  "width": 128
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats | activation_inactive_gate_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | parity8 | 0 | 0.59375 | 0.453125 | 0.140625 | 1.37558 | 175.152 | 173.776 | 0.59375 | 0.453125 | 0.140625 | 1.37558 | 175.152 | 173.776 | 3.00143 | -1 | -1 | 0.972656 | 512 | 4 | 32 | nan | nan | 0 | 0.314453 |
| dlgn_anneal | parity8 | 0 | 0.28125 | 0.390625 | 0.109375 | 18.2034 | 150.065 | 131.862 | 0.28125 | 0.390625 | 0.109375 | 18.2034 | 150.065 | 131.862 | 2.88928 | -1 | -1 | 0.982422 | 512 | 4 | 32 | nan | nan | 0 | 0.300781 |
| gumbel_st | parity8 | 0 | 0.40625 | 0.59375 | 0.1875 | 1.43203 | 112.522 | 111.09 | 0.40625 | 0.59375 | 0.1875 | 1.43203 | 112.522 | 111.09 | 2.8655 | -1 | -1 | 0.992188 | 512 | 4 | 32 | nan | nan | 0 | 0.322266 |
| hard_st_cage | parity8 | 0 | 0.40625 | 0.328125 | 0.078125 | 4.20978 | 103.32 | 99.1102 | 0.328125 | 0.328125 | 0 | 103.32 | 103.32 | 0 | 2.82583 | -1 | -1 | 0.992188 | 512 | 4 | 32 | nan | nan | 0 | 0.40625 |
| block_relaxed | parity8 | 0 | 0.59375 | 0.484375 | 0.109375 | 0.678053 | 154.763 | 154.085 | 0.59375 | 0.484375 | 0.109375 | 0.678053 | 154.763 | 154.085 | 0.710527 | -1 | -1 | 0.980469 | 512 | 4 | 32 | nan | nan | 0 | 0.324219 |
| block_hard_refit | parity8 | 0 | 0.59375 | 0.484375 | 0.109375 | 3.24229 | 98.5458 | 95.3035 | 0.375 | 0.484375 | 0.109375 | 17.7364 | 98.5458 | 80.8094 | 0.75645 | -1 | -1 | 0.982422 | 512 | 4 | 32 | nan | nan | 0 | 0.359375 |
| block_hard_task_refit | parity8 | 0 | 0.40625 | 0.34375 | 0.0625 | 27.8715 | 42.5449 | 14.6734 | 0.34375 | 0.34375 | 0 | 16.4628 | 42.5449 | 26.0821 | 2.87125 | -1 | -1 | 0.986328 | 512 | 4 | 32 | nan | nan | 0 | 0.423828 |
| dlgn | majority9 | 0 | 1 | 0.695312 | 0.304688 | 0.016097 | 97.6996 | 97.6835 | 1 | 0.695312 | 0.304688 | 0.016097 | 97.6996 | 97.6835 | 5.16636 | -1 | -1 | 0.943359 | 512 | 4 | 29 | nan | nan | 0 | 0.261719 |
| dlgn_anneal | majority9 | 0 | 0.976562 | 0.820312 | 0.15625 | 0.469442 | 50.8029 | 50.3335 | 0.976562 | 0.820312 | 0.15625 | 0.469442 | 50.8029 | 50.3335 | 5.79911 | -1 | -1 | 0.947266 | 512 | 4 | 29 | nan | nan | 0 | 0.220703 |
| gumbel_st | majority9 | 0 | 0.460938 | 0.484375 | 0.0234375 | 37.3057 | 159.478 | 122.172 | 0.460938 | 0.484375 | 0.0234375 | 37.3057 | 159.478 | 122.172 | 5.35402 | -1 | -1 | 0.990234 | 512 | 4 | 29 | nan | nan | 0 | 0.300781 |
| hard_st_cage | majority9 | 0 | 0.539062 | 0.8125 | 0.273438 | 3.57899 | 22.7104 | 19.1314 | 0.8125 | 0.8125 | 0 | 22.7104 | 22.7104 | 0 | 5.70496 | -1 | -1 | 0.96875 | 512 | 4 | 29 | nan | nan | 0 | 0.349609 |
| block_relaxed | majority9 | 0 | 0.640625 | 0.5 | 0.140625 | 0.624828 | 138.39 | 137.765 | 0.640625 | 0.5 | 0.140625 | 0.624828 | 138.39 | 137.765 | 1.25191 | -1 | -1 | 0.986328 | 512 | 4 | 29 | nan | nan | 0 | 0.314453 |
| block_hard_refit | majority9 | 0 | 0.539062 | 0.828125 | 0.289062 | 36.9718 | 22.7375 | 14.2343 | 0.882812 | 0.828125 | 0.0546875 | 2.72728 | 22.7375 | 20.0102 | 1.3062 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.304688 |
| block_hard_task_refit | majority9 | 0 | 0.539062 | 0.71875 | 0.179688 | 36.1386 | 32.172 | 3.96659 | 0.859375 | 0.71875 | 0.140625 | 4.22471 | 32.172 | 27.9473 | 3.59146 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.378906 |
| dlgn | random_sparse10 | 0 | 1 | 0.585938 | 0.414062 | 0.00141117 | 132.471 | 132.469 | 1 | 0.585938 | 0.414062 | 0.00141117 | 132.471 | 132.469 | 10.1882 | -1 | -1 | 0.890625 | 512 | 4 | 26 | nan | nan | 0 | 0.1875 |
| dlgn_anneal | random_sparse10 | 0 | 1 | 0.800781 | 0.199219 | 9.83128e-06 | 47.2954 | 47.2954 | 1 | 0.800781 | 0.199219 | 9.83128e-06 | 47.2954 | 47.2954 | 11.3903 | -1 | -1 | 0.873047 | 512 | 4 | 26 | nan | nan | 0 | 0.175781 |
| gumbel_st | random_sparse10 | 0 | 0.515625 | 0.613281 | 0.0976562 | 39.6432 | 112.939 | 73.2961 | 0.515625 | 0.613281 | 0.0976562 | 39.6432 | 112.939 | 73.2961 | 10.2261 | -1 | -1 | 0.994141 | 512 | 4 | 26 | nan | nan | 0 | 0.314453 |
| hard_st_cage | random_sparse10 | 0 | 0.515625 | 0.917969 | 0.402344 | 30.4624 | 9.42645 | 21.0359 | 0.917969 | 0.917969 | 0 | 9.42645 | 9.42645 | 0 | 10.2761 | -1 | -1 | 0.882812 | 512 | 4 | 26 | nan | nan | 0 | 0.353516 |
| block_relaxed | random_sparse10 | 0 | 0.527344 | 0.476562 | 0.0507812 | 0.76412 | 187.188 | 186.424 | 0.527344 | 0.476562 | 0.0507812 | 0.76412 | 187.188 | 186.424 | 2.37573 | -1 | -1 | 0.984375 | 512 | 4 | 26 | nan | nan | 0 | 0.285156 |
| block_hard_refit | random_sparse10 | 0 | 0.484375 | 0.484375 | 0 | 26.7067 | 220.717 | 194.01 | 0.824219 | 0.484375 | 0.339844 | 1.82246 | 220.717 | 218.894 | 2.43987 | -1 | -1 | 0.992188 | 512 | 4 | 26 | nan | nan | 0 | 0.375 |
| block_hard_task_refit | random_sparse10 | 0 | 0.484375 | 0.421875 | 0.0625 | 49.0547 | 47.4741 | 1.58061 | 0.699219 | 0.421875 | 0.277344 | 2.45888 | 47.4741 | 45.0152 | 5.14178 | -1 | -1 | 0.994141 | 512 | 4 | 26 | nan | nan | 0 | 0.492188 |
