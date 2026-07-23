# Hard-LGN Gap Prototype Summary

Elapsed seconds: 250.85

Metric notes:

- `soft_acc`/`soft_loss` evaluate all trained relaxed blocks in soft mode.
- `discrete_acc` uses the strict Boolean/integer executor: argmax gates for DLGN-style methods and fitted gates for block-hard methods. `discrete_loss` materializes integer class counts only after audited inference for external cross-entropy measurement.
- `path_*` metrics evaluate the method-native training path. Hard-ST uses the deterministic hard forward; block-hard methods use hard frozen prefixes plus the final relaxed block.
- `unused_gate_ratio` follows Mind the Gap: logit entropy above the deterministic N(0,1) initialization 2.5% threshold (1.884315).
- `activation_inactive_gate_ratio` retains the older data-dependent statistic: hard gate outputs that are constant on the training set.
- `layer_diagnostics.csv` tracks relaxed-path versus hard-path representation mismatch after each layer prefix for depth-wise gap accumulation checks.
- Hard inference throughput uses Boolean gate tensors and integer class counts. It remains an unpacked PyTorch transaction reference, not a bit-packed kernel.
- `hard_runtime_floating_tensor_count=0` is enforced dynamically; a BooleanRuntimeAudit violation raises instead of producing a row.
- For block-wise methods, `epochs_to_target` is conservative: total block epochs if the final hard model reaches the dataset target, otherwise `-1`.
- `time_to_target` is wall-clock seconds to first discrete target hit for end-to-end methods; for block-wise methods it is conservative final train time if the final hard model reaches target, otherwise `-1`.

Configuration:

```json
{
  "abc_max_gates": 10000,
  "abc_path": "/home/spco/boolean_sat/abc/abc",
  "abc_stats": false,
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
  "out_dir": "/home/wangmeiqi/learnable_logic_scaling_20260723/remote_runs/strict_bool_scaled_w128_d4_e120_s012_20260723",
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
    0,
    1,
    2
  ],
  "target_acc_override": null,
  "temp_end": 0.2,
  "temp_start": 2.0,
  "threshold_levels": 1,
  "weight_decay": 0.0,
  "width": 128
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats | activation_inactive_gate_ratio | hard_runtime_domain | hard_runtime_audit_operations | hard_runtime_floating_tensor_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | parity8 | 0 | 0.59375 | 0.453125 | 0.140625 | 1.37558 | 175.152 | 173.776 | 0.59375 | 0.453125 | 0.140625 | 1.37558 | 175.152 | 173.776 | 4.77688 | -1 | -1 | 0.972656 | 512 | 4 | 32 | nan | nan | 0 | 0.314453 | bool_int | 39 | 0 |
| dlgn_anneal | parity8 | 0 | 0.28125 | 0.390625 | 0.109375 | 18.2034 | 150.065 | 131.862 | 0.28125 | 0.390625 | 0.109375 | 18.2034 | 150.065 | 131.862 | 4.32166 | -1 | -1 | 0.982422 | 512 | 4 | 32 | nan | nan | 0 | 0.300781 | bool_int | 39 | 0 |
| gumbel_st | parity8 | 0 | 0.40625 | 0.59375 | 0.1875 | 1.43203 | 112.522 | 111.09 | 0.40625 | 0.59375 | 0.1875 | 1.43203 | 112.522 | 111.09 | 3.82952 | -1 | -1 | 0.992188 | 512 | 4 | 32 | nan | nan | 0 | 0.322266 | bool_int | 39 | 0 |
| hard_st_cage | parity8 | 0 | 0.40625 | 0.328125 | 0.078125 | 4.20978 | 103.32 | 99.1102 | 0.328125 | 0.328125 | 0 | 103.32 | 103.32 | 0 | 2.7949 | -1 | -1 | 0.992188 | 512 | 4 | 32 | nan | nan | 0 | 0.40625 | bool_int | 39 | 0 |
| block_relaxed | parity8 | 0 | 0.59375 | 0.484375 | 0.109375 | 0.678053 | 154.763 | 154.085 | 0.59375 | 0.484375 | 0.109375 | 0.678053 | 154.763 | 154.085 | 0.630958 | -1 | -1 | 0.980469 | 512 | 4 | 32 | nan | nan | 0 | 0.324219 | bool_int | 39 | 0 |
| block_hard_refit | parity8 | 0 | 0.59375 | 0.484375 | 0.109375 | 3.24229 | 98.5458 | 95.3035 | 0.375 | 0.484375 | 0.109375 | 17.7364 | 98.5458 | 80.8094 | 0.67964 | -1 | -1 | 0.982422 | 512 | 4 | 32 | nan | nan | 0 | 0.359375 | bool_int | 39 | 0 |
| block_hard_task_refit | parity8 | 0 | 0.40625 | 0.34375 | 0.0625 | 27.8715 | 42.5449 | 14.6734 | 0.34375 | 0.34375 | 0 | 16.4628 | 42.5449 | 26.0821 | 2.93478 | -1 | -1 | 0.986328 | 512 | 4 | 32 | nan | nan | 0 | 0.423828 | bool_int | 39 | 0 |
| dlgn | majority9 | 0 | 1 | 0.695312 | 0.304688 | 0.016097 | 97.6996 | 97.6835 | 1 | 0.695312 | 0.304688 | 0.016097 | 97.6996 | 97.6835 | 4.31915 | -1 | -1 | 0.943359 | 512 | 4 | 29 | nan | nan | 0 | 0.261719 | bool_int | 39 | 0 |
| dlgn_anneal | majority9 | 0 | 0.976562 | 0.820312 | 0.15625 | 0.469442 | 50.8029 | 50.3335 | 0.976562 | 0.820312 | 0.15625 | 0.469442 | 50.8029 | 50.3335 | 5.2925 | -1 | -1 | 0.947266 | 512 | 4 | 29 | nan | nan | 0 | 0.220703 | bool_int | 39 | 0 |
| gumbel_st | majority9 | 0 | 0.460938 | 0.484375 | 0.0234375 | 37.3057 | 159.478 | 122.172 | 0.460938 | 0.484375 | 0.0234375 | 37.3057 | 159.478 | 122.172 | 5.21126 | -1 | -1 | 0.990234 | 512 | 4 | 29 | nan | nan | 0 | 0.300781 | bool_int | 39 | 0 |
| hard_st_cage | majority9 | 0 | 0.539062 | 0.8125 | 0.273438 | 3.57899 | 22.7104 | 19.1314 | 0.8125 | 0.8125 | 0 | 22.7104 | 22.7104 | 0 | 4.82335 | -1 | -1 | 0.96875 | 512 | 4 | 29 | nan | nan | 0 | 0.349609 | bool_int | 39 | 0 |
| block_relaxed | majority9 | 0 | 0.640625 | 0.5 | 0.140625 | 0.624828 | 138.39 | 137.765 | 0.640625 | 0.5 | 0.140625 | 0.624828 | 138.39 | 137.765 | 1.18362 | -1 | -1 | 0.986328 | 512 | 4 | 29 | nan | nan | 0 | 0.314453 | bool_int | 39 | 0 |
| block_hard_refit | majority9 | 0 | 0.539062 | 0.828125 | 0.289062 | 36.9718 | 22.7375 | 14.2343 | 0.882812 | 0.828125 | 0.0546875 | 2.72728 | 22.7375 | 20.0102 | 1.24336 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.304688 | bool_int | 39 | 0 |
| block_hard_task_refit | majority9 | 0 | 0.539062 | 0.71875 | 0.179688 | 36.1386 | 32.172 | 3.96659 | 0.859375 | 0.71875 | 0.140625 | 4.22471 | 32.172 | 27.9473 | 3.46058 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.378906 | bool_int | 39 | 0 |
| dlgn | random_sparse10 | 0 | 1 | 0.585938 | 0.414062 | 0.00141117 | 132.471 | 132.469 | 1 | 0.585938 | 0.414062 | 0.00141117 | 132.471 | 132.469 | 7.10359 | -1 | -1 | 0.890625 | 512 | 4 | 26 | nan | nan | 0 | 0.1875 | bool_int | 39 | 0 |
| dlgn_anneal | random_sparse10 | 0 | 1 | 0.800781 | 0.199219 | 9.83128e-06 | 47.2954 | 47.2954 | 1 | 0.800781 | 0.199219 | 9.83128e-06 | 47.2954 | 47.2954 | 8.52369 | -1 | -1 | 0.873047 | 512 | 4 | 26 | nan | nan | 0 | 0.175781 | bool_int | 39 | 0 |
| gumbel_st | random_sparse10 | 0 | 0.515625 | 0.613281 | 0.0976562 | 39.6432 | 112.939 | 73.2961 | 0.515625 | 0.613281 | 0.0976562 | 39.6432 | 112.939 | 73.2961 | 8.32372 | -1 | -1 | 0.994141 | 512 | 4 | 26 | nan | nan | 0 | 0.314453 | bool_int | 39 | 0 |
| hard_st_cage | random_sparse10 | 0 | 0.515625 | 0.917969 | 0.402344 | 30.4624 | 9.42645 | 21.0359 | 0.917969 | 0.917969 | 0 | 9.42645 | 9.42645 | 0 | 8.17049 | -1 | -1 | 0.882812 | 512 | 4 | 26 | nan | nan | 0 | 0.353516 | bool_int | 39 | 0 |
| block_relaxed | random_sparse10 | 0 | 0.527344 | 0.476562 | 0.0507812 | 0.76412 | 187.188 | 186.424 | 0.527344 | 0.476562 | 0.0507812 | 0.76412 | 187.188 | 186.424 | 2.28806 | -1 | -1 | 0.984375 | 512 | 4 | 26 | nan | nan | 0 | 0.285156 | bool_int | 39 | 0 |
| block_hard_refit | random_sparse10 | 0 | 0.484375 | 0.484375 | 0 | 26.7067 | 220.717 | 194.01 | 0.824219 | 0.484375 | 0.339844 | 1.82246 | 220.717 | 218.894 | 2.33869 | -1 | -1 | 0.992188 | 512 | 4 | 26 | nan | nan | 0 | 0.375 | bool_int | 39 | 0 |
| block_hard_task_refit | random_sparse10 | 0 | 0.484375 | 0.421875 | 0.0625 | 49.0547 | 47.4741 | 1.58061 | 0.699219 | 0.421875 | 0.277344 | 2.45888 | 47.4741 | 45.0152 | 4.59336 | -1 | -1 | 0.994141 | 512 | 4 | 26 | nan | nan | 0 | 0.492188 | bool_int | 39 | 0 |
| dlgn | parity8 | 1 | 0.328125 | 0.5 | 0.171875 | 0.992841 | 176.649 | 175.656 | 0.328125 | 0.5 | 0.171875 | 0.992841 | 176.649 | 175.656 | 2.4545 | -1 | -1 | 0.966797 | 512 | 4 | 32 | nan | nan | 0 | 0.28125 | bool_int | 39 | 0 |
| dlgn_anneal | parity8 | 1 | 0.21875 | 0.46875 | 0.25 | 11.948 | 175 | 163.052 | 0.21875 | 0.46875 | 0.25 | 11.948 | 175 | 163.052 | 2.61853 | -1 | -1 | 0.972656 | 512 | 4 | 32 | nan | nan | 0 | 0.248047 | bool_int | 39 | 0 |
| gumbel_st | parity8 | 1 | 0.515625 | 0.5 | 0.015625 | 14.1981 | 150.043 | 135.845 | 0.515625 | 0.5 | 0.015625 | 14.1981 | 150.043 | 135.845 | 2.58132 | -1 | -1 | 0.984375 | 512 | 4 | 32 | nan | nan | 0 | 0.353516 | bool_int | 39 | 0 |
| hard_st_cage | parity8 | 1 | 0.484375 | 0.546875 | 0.0625 | 8.60504 | 89.1925 | 80.5874 | 0.546875 | 0.546875 | 0 | 89.1925 | 89.1925 | 0 | 2.53187 | -1 | -1 | 0.994141 | 512 | 4 | 32 | nan | nan | 0 | 0.412109 | bool_int | 39 | 0 |
| block_relaxed | parity8 | 1 | 0.40625 | 0.46875 | 0.0625 | 0.78362 | 220.323 | 219.54 | 0.40625 | 0.46875 | 0.0625 | 0.78362 | 220.323 | 219.54 | 0.719117 | -1 | -1 | 0.978516 | 512 | 4 | 32 | nan | nan | 0 | 0.294922 | bool_int | 39 | 0 |
| block_hard_refit | parity8 | 1 | 0.515625 | 0.46875 | 0.046875 | 11.8183 | 118.837 | 107.018 | 0.453125 | 0.46875 | 0.015625 | 14.1238 | 118.837 | 104.713 | 0.769745 | -1 | -1 | 0.986328 | 512 | 4 | 32 | nan | nan | 0 | 0.373047 | bool_int | 39 | 0 |
| block_hard_task_refit | parity8 | 1 | 0.484375 | 0.53125 | 0.046875 | 45.6733 | 42.4258 | 3.24749 | 0.421875 | 0.53125 | 0.109375 | 8.90571 | 42.4258 | 33.5201 | 2.86338 | -1 | -1 | 0.982422 | 512 | 4 | 32 | nan | nan | 0 | 0.478516 | bool_int | 39 | 0 |
| dlgn | majority9 | 1 | 1 | 0.65625 | 0.34375 | 0.00898349 | 113.335 | 113.326 | 1 | 0.65625 | 0.34375 | 0.00898349 | 113.335 | 113.326 | 4.07897 | -1 | -1 | 0.914062 | 512 | 4 | 29 | nan | nan | 0 | 0.230469 | bool_int | 39 | 0 |
| dlgn_anneal | majority9 | 1 | 0.976562 | 0.773438 | 0.203125 | 0.144006 | 57.8504 | 57.7064 | 0.976562 | 0.773438 | 0.203125 | 0.144006 | 57.8504 | 57.7064 | 4.93186 | -1 | -1 | 0.9375 | 512 | 4 | 29 | nan | nan | 0 | 0.199219 | bool_int | 39 | 0 |
| gumbel_st | majority9 | 1 | 0.46875 | 0.515625 | 0.046875 | 27.3935 | 133.68 | 106.287 | 0.46875 | 0.515625 | 0.046875 | 27.3935 | 133.68 | 106.287 | 4.29433 | -1 | -1 | 0.988281 | 512 | 4 | 29 | nan | nan | 0 | 0.316406 | bool_int | 39 | 0 |
| hard_st_cage | majority9 | 1 | 0.53125 | 0.851562 | 0.320312 | 12.8612 | 26.595 | 13.7338 | 0.851562 | 0.851562 | 0 | 26.595 | 26.595 | 0 | 4.21629 | -1 | -1 | 0.974609 | 512 | 4 | 29 | nan | nan | 0 | 0.320312 | bool_int | 39 | 0 |
| block_relaxed | majority9 | 1 | 0.703125 | 0.453125 | 0.25 | 0.588374 | 224.24 | 223.652 | 0.703125 | 0.453125 | 0.25 | 0.588374 | 224.24 | 223.652 | 1.18884 | -1 | -1 | 0.978516 | 512 | 4 | 29 | nan | nan | 0 | 0.279297 | bool_int | 39 | 0 |
| block_hard_refit | majority9 | 1 | 0.46875 | 0.765625 | 0.296875 | 14.6946 | 55.5337 | 40.8392 | 0.898438 | 0.765625 | 0.132812 | 2.47748 | 55.5337 | 53.0563 | 1.24241 | -1 | -1 | 0.990234 | 512 | 4 | 29 | nan | nan | 0 | 0.326172 | bool_int | 39 | 0 |
| block_hard_task_refit | majority9 | 1 | 0.53125 | 0.703125 | 0.171875 | 1.90353 | 19.6991 | 17.7956 | 0.882812 | 0.703125 | 0.179688 | 2.3524 | 19.6991 | 17.3467 | 3.4216 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.417969 | bool_int | 39 | 0 |
| dlgn | random_sparse10 | 1 | 1 | 0.550781 | 0.449219 | 0.00179707 | 132.875 | 132.873 | 1 | 0.550781 | 0.449219 | 0.00179707 | 132.875 | 132.873 | 7.77233 | -1 | -1 | 0.884766 | 512 | 4 | 26 | nan | nan | 0 | 0.228516 | bool_int | 39 | 0 |
| dlgn_anneal | random_sparse10 | 1 | 1 | 0.761719 | 0.238281 | 0.00180023 | 60.9727 | 60.9709 | 1 | 0.761719 | 0.238281 | 0.00180023 | 60.9727 | 60.9709 | 9.7355 | -1 | -1 | 0.861328 | 512 | 4 | 26 | nan | nan | 0 | 0.154297 | bool_int | 39 | 0 |
| gumbel_st | random_sparse10 | 1 | 0.5625 | 0.523438 | 0.0390625 | 2.7141 | 206.291 | 203.577 | 0.5625 | 0.523438 | 0.0390625 | 2.7141 | 206.291 | 203.577 | 8.27854 | -1 | -1 | 0.988281 | 512 | 4 | 26 | nan | nan | 0 | 0.347656 | bool_int | 39 | 0 |
| hard_st_cage | random_sparse10 | 1 | 0.5625 | 0.980469 | 0.417969 | 15.2717 | 0.406871 | 14.8648 | 0.980469 | 0.980469 | 0 | 0.406871 | 0.406871 | 0 | 8.13034 | 84 | 5.68592 | 0.867188 | 512 | 4 | 26 | nan | nan | 0 | 0.253906 | bool_int | 39 | 0 |
| block_relaxed | random_sparse10 | 1 | 0.445312 | 0.398438 | 0.046875 | 0.829824 | 200.485 | 199.656 | 0.445312 | 0.398438 | 0.046875 | 0.829824 | 200.485 | 199.656 | 2.22982 | -1 | -1 | 0.976562 | 512 | 4 | 26 | nan | nan | 0 | 0.251953 | bool_int | 39 | 0 |
| block_hard_refit | random_sparse10 | 1 | 0.5625 | 0.859375 | 0.296875 | 16.0349 | 19.9869 | 3.952 | 0.917969 | 0.859375 | 0.0585938 | 0.892434 | 19.9869 | 19.0944 | 2.29897 | -1 | -1 | 0.986328 | 512 | 4 | 26 | nan | nan | 0 | 0.271484 | bool_int | 39 | 0 |
| block_hard_task_refit | random_sparse10 | 1 | 0.5625 | 0.738281 | 0.175781 | 47.3121 | 11.5474 | 35.7647 | 0.957031 | 0.738281 | 0.21875 | 0.325435 | 11.5474 | 11.222 | 4.50261 | -1 | -1 | 0.980469 | 512 | 4 | 26 | nan | nan | 0 | 0.378906 | bool_int | 39 | 0 |
| dlgn | parity8 | 2 | 0.46875 | 0.609375 | 0.140625 | 1.24473 | 103.179 | 101.934 | 0.46875 | 0.609375 | 0.140625 | 1.24473 | 103.179 | 101.934 | 2.4008 | -1 | -1 | 0.980469 | 512 | 4 | 32 | nan | nan | 0 | 0.341797 | bool_int | 39 | 0 |
| dlgn_anneal | parity8 | 2 | 0.234375 | 0.453125 | 0.21875 | 16.7735 | 182.888 | 166.115 | 0.234375 | 0.453125 | 0.21875 | 16.7735 | 182.888 | 166.115 | 2.57635 | -1 | -1 | 0.980469 | 512 | 4 | 32 | nan | nan | 0 | 0.3125 | bool_int | 39 | 0 |
| gumbel_st | parity8 | 2 | 0.484375 | 0.546875 | 0.0625 | 10.452 | 229.752 | 219.3 | 0.484375 | 0.546875 | 0.0625 | 10.452 | 229.752 | 219.3 | 2.54241 | -1 | -1 | 0.992188 | 512 | 4 | 32 | nan | nan | 0 | 0.316406 | bool_int | 39 | 0 |
| hard_st_cage | parity8 | 2 | 0.515625 | 0.4375 | 0.078125 | 7.83379 | 95.4208 | 87.587 | 0.4375 | 0.4375 | 0 | 95.4208 | 95.4208 | 0 | 2.53245 | -1 | -1 | 0.990234 | 512 | 4 | 32 | nan | nan | 0 | 0.386719 | bool_int | 39 | 0 |
| block_relaxed | parity8 | 2 | 0.5 | 0.5 | 0 | 0.752834 | 134.462 | 133.709 | 0.5 | 0.5 | 0 | 0.752834 | 134.462 | 133.709 | 0.714543 | -1 | -1 | 0.980469 | 512 | 4 | 32 | nan | nan | 0 | 0.34375 | bool_int | 39 | 0 |
| block_hard_refit | parity8 | 2 | 0.515625 | 0.5 | 0.015625 | 6.18988 | 84.505 | 78.3151 | 0.40625 | 0.5 | 0.09375 | 14.2902 | 84.505 | 70.2147 | 0.756455 | -1 | -1 | 0.990234 | 512 | 4 | 32 | nan | nan | 0 | 0.367188 | bool_int | 39 | 0 |
| block_hard_task_refit | parity8 | 2 | 0.515625 | 0.5 | 0.015625 | 27.463 | 47.1349 | 19.6719 | 0.453125 | 0.5 | 0.046875 | 15.9035 | 47.1349 | 31.2314 | 2.82087 | -1 | -1 | 0.984375 | 512 | 4 | 32 | nan | nan | 0 | 0.423828 | bool_int | 39 | 0 |
| dlgn | majority9 | 2 | 1 | 0.75 | 0.25 | 0.0134075 | 93.0283 | 93.0149 | 1 | 0.75 | 0.25 | 0.0134075 | 93.0283 | 93.0149 | 4.01229 | -1 | -1 | 0.929688 | 512 | 4 | 29 | nan | nan | 0 | 0.230469 | bool_int | 39 | 0 |
| dlgn_anneal | majority9 | 2 | 1 | 0.882812 | 0.117188 | 0 | 23.4592 | 23.4592 | 1 | 0.882812 | 0.117188 | 0 | 23.4592 | 23.4592 | 4.81754 | -1 | -1 | 0.919922 | 512 | 4 | 29 | nan | nan | 0 | 0.191406 | bool_int | 39 | 0 |
| gumbel_st | majority9 | 2 | 0.507812 | 0.5625 | 0.0546875 | 13.6671 | 196.891 | 183.224 | 0.507812 | 0.5625 | 0.0546875 | 13.6671 | 196.891 | 183.224 | 4.13044 | -1 | -1 | 0.992188 | 512 | 4 | 29 | nan | nan | 0 | 0.335938 | bool_int | 39 | 0 |
| hard_st_cage | majority9 | 2 | 0.492188 | 0.75 | 0.257812 | 14.644 | 20.3991 | 5.75511 | 0.75 | 0.75 | 0 | 20.3991 | 20.3991 | 0 | 4.16832 | -1 | -1 | 0.96875 | 512 | 4 | 29 | nan | nan | 0 | 0.470703 | bool_int | 39 | 0 |
| block_relaxed | majority9 | 2 | 0.53125 | 0.507812 | 0.0234375 | 0.841381 | 139.16 | 138.319 | 0.53125 | 0.507812 | 0.0234375 | 0.841381 | 139.16 | 138.319 | 1.17042 | -1 | -1 | 0.978516 | 512 | 4 | 29 | nan | nan | 0 | 0.371094 | bool_int | 39 | 0 |
| block_hard_refit | majority9 | 2 | 0.492188 | 0.75 | 0.257812 | 26.4025 | 47.6942 | 21.2916 | 0.898438 | 0.75 | 0.148438 | 2.63402 | 47.6942 | 45.0601 | 1.41859 | -1 | -1 | 0.984375 | 512 | 4 | 29 | nan | nan | 0 | 0.300781 | bool_int | 39 | 0 |
| block_hard_task_refit | majority9 | 2 | 0.492188 | 0.609375 | 0.117188 | 34.8452 | 22.8729 | 11.9723 | 0.914062 | 0.609375 | 0.304688 | 1.883 | 22.8729 | 20.9899 | 3.35891 | -1 | -1 | 0.980469 | 512 | 4 | 29 | nan | nan | 0 | 0.392578 | bool_int | 39 | 0 |
| dlgn | random_sparse10 | 2 | 1 | 0.714844 | 0.285156 | 0.00168896 | 63.7369 | 63.7352 | 1 | 0.714844 | 0.285156 | 0.00168896 | 63.7369 | 63.7352 | 7.73296 | -1 | -1 | 0.900391 | 512 | 4 | 26 | nan | nan | 0 | 0.230469 | bool_int | 39 | 0 |
| dlgn_anneal | random_sparse10 | 2 | 0.996094 | 0.78125 | 0.214844 | 0.00444736 | 50.046 | 50.0416 | 0.996094 | 0.78125 | 0.214844 | 0.00444736 | 50.046 | 50.0416 | 9.39015 | -1 | -1 | 0.876953 | 512 | 4 | 26 | nan | nan | 0 | 0.224609 | bool_int | 39 | 0 |
| gumbel_st | random_sparse10 | 2 | 0.472656 | 0.527344 | 0.0546875 | 14.5064 | 155.143 | 140.637 | 0.472656 | 0.527344 | 0.0546875 | 14.5064 | 155.143 | 140.637 | 8.24866 | -1 | -1 | 0.994141 | 512 | 4 | 26 | nan | nan | 0 | 0.365234 | bool_int | 39 | 0 |
| hard_st_cage | random_sparse10 | 2 | 0.527344 | 0.980469 | 0.453125 | 6.11867 | 1.97479 | 4.14388 | 0.980469 | 0.980469 | 0 | 1.97479 | 1.97479 | 0 | 7.95272 | 59 | 3.92008 | 0.900391 | 512 | 4 | 26 | nan | nan | 0 | 0.322266 | bool_int | 39 | 0 |
| block_relaxed | random_sparse10 | 2 | 0.472656 | 0.394531 | 0.078125 | 0.893405 | 229.327 | 228.433 | 0.472656 | 0.394531 | 0.078125 | 0.893405 | 229.327 | 228.433 | 2.22139 | -1 | -1 | 0.986328 | 512 | 4 | 26 | nan | nan | 0 | 0.318359 | bool_int | 39 | 0 |
| block_hard_refit | random_sparse10 | 2 | 0.472656 | 0.753906 | 0.28125 | 2.5414 | 48.516 | 45.9746 | 0.875 | 0.753906 | 0.121094 | 2.84556 | 48.516 | 45.6705 | 2.27628 | -1 | -1 | 0.992188 | 512 | 4 | 26 | nan | nan | 0 | 0.302734 | bool_int | 39 | 0 |
| block_hard_task_refit | random_sparse10 | 2 | 0.527344 | 0.695312 | 0.167969 | 27.4886 | 14.3549 | 13.1337 | 0.878906 | 0.695312 | 0.183594 | 0.968776 | 14.3549 | 13.3861 | 4.36362 | -1 | -1 | 0.988281 | 512 | 4 | 26 | nan | nan | 0 | 0.466797 | bool_int | 39 | 0 |
