# Hard-LGN Gap Prototype Summary

Elapsed seconds: 17.77

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
  "abc_stats": false,
  "batch_size": 128,
  "block_epochs": 40,
  "block_total_epochs": 20,
  "cage_beta": 0.99,
  "cage_tau_max": 3.0,
  "cage_tau_min": 0.5,
  "compat_only": false,
  "data_dir": "/home/spco/data",
  "datasets": [
    "cifar10_small"
  ],
  "device": "cuda",
  "difflogic_compat_check": false,
  "download_data": false,
  "entropy_coef": 0.001,
  "epochs": 20,
  "eval_batch_size": 512,
  "exact_truth_max": 12,
  "group_tau": 0.01,
  "gumbel_temp_end": 1.0,
  "gumbel_temp_start": 1.0,
  "hard_st_temp": 1.0,
  "image_max_test": 500,
  "image_max_train": 2000,
  "inference_bench": false,
  "inference_bench_device": "cpu",
  "inference_bench_repeats": 20,
  "inference_bench_warmup": 1,
  "layers": 3,
  "lr": 0.01,
  "methods": [
    "dlgn",
    "dlgn_anneal",
    "gumbel_st",
    "hard_st_cage",
    "block_relaxed",
    "block_hard_refit"
  ],
  "mind_gap_scaled": true,
  "optimizer": "adam",
  "out_dir": "runs/mind_gap_scaled_cifar10small_w1600_d3_e20_s0_20260723",
  "quick": false,
  "refit_candidate_topk": 8,
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
  "width": 1600
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats | activation_inactive_gate_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | cifar10_small | 0 | 0.22 | 0.09 | 0.13 | 8.96938 | 893.681 | 884.711 | 0.22 | 0.09 | 0.13 | 8.96938 | 893.681 | 884.711 | 2.46993 | -1 | -1 | 0.976667 | 4800 | 3 | 2 | nan | nan | 0 | 0.2075 |
| dlgn_anneal | cifar10_small | 0 | 0.226 | 0.15 | 0.076 | 108.179 | 655.724 | 547.545 | 0.226 | 0.15 | 0.076 | 108.179 | 655.724 | 547.545 | 2.00864 | -1 | -1 | 0.979375 | 4800 | 3 | 2 | nan | nan | 0 | 0.211042 |
| gumbel_st | cifar10_small | 0 | 0.126 | 0.106 | 0.02 | 131.837 | 992.884 | 861.048 | 0.126 | 0.106 | 0.02 | 131.837 | 992.884 | 861.048 | 2.01352 | -1 | -1 | 0.983958 | 4800 | 3 | 2 | nan | nan | 0 | 0.208333 |
| hard_st_cage | cifar10_small | 0 | 0.1 | 0.194 | 0.094 | 39.3455 | 540.134 | 500.789 | 0.194 | 0.194 | 0 | 540.134 | 540.134 | 0 | 1.89375 | -1 | -1 | 0.987083 | 4800 | 3 | 2 | nan | nan | 0 | 0.258333 |
| block_relaxed | cifar10_small | 0 | 0.116 | 0.112 | 0.004 | 6.81237 | 1033.26 | 1026.45 | 0.116 | 0.112 | 0.004 | 6.81237 | 1033.26 | 1026.45 | 0.719207 | -1 | -1 | 0.982917 | 4800 | 3 | 2 | nan | nan | 0 | 0.221667 |
| block_hard_refit | cifar10_small | 0 | 0.1 | 0.164 | 0.064 | 86.253 | 516.117 | 429.864 | 0.154 | 0.164 | 0.01 | 90.8393 | 516.117 | 425.277 | 2.15804 | -1 | -1 | 0.989167 | 4800 | 3 | 2 | nan | nan | 0 | 0.296875 |
