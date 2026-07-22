# Hard-LGN Gap Prototype Summary

Elapsed seconds: 61.44

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
  "block_total_epochs": 120,
  "cage_beta": 0.99,
  "cage_tau_max": 3.0,
  "cage_tau_min": 0.5,
  "compat_only": false,
  "data_dir": "/home/spco/data",
  "datasets": [
    "digits"
  ],
  "device": "cuda",
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
  "out_dir": "runs/mind_gap_scaled_digits_w320_d4_e120_s0_20260723",
  "quick": false,
  "refit_candidate_topk": 8,
  "refit_coordinate_passes": 1,
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
  "width": 320
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats | activation_inactive_gate_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | digits | 0 | 0.922222 | 0.42 | 0.502222 | 0.418131 | 142.435 | 142.017 | 0.922222 | 0.42 | 0.502222 | 0.418131 | 142.435 | 142.017 | 12.5802 | -1 | -1 | 0.790625 | 1280 | 4 | 10 | nan | nan | 0 | 0.178125 |
| dlgn_anneal | digits | 0 | 0.902222 | 0.693333 | 0.208889 | 9.51755 | 54.5978 | 45.0803 | 0.902222 | 0.693333 | 0.208889 | 9.51755 | 54.5978 | 45.0803 | 13.629 | -1 | -1 | 0.709375 | 1280 | 4 | 10 | nan | nan | 0 | 0.132812 |
| gumbel_st | digits | 0 | 0.0955556 | 0.0822222 | 0.0133333 | 34.375 | 409.114 | 374.739 | 0.0955556 | 0.0822222 | 0.0133333 | 34.375 | 409.114 | 374.739 | 12.0429 | -1 | -1 | 0.996875 | 1280 | 4 | 10 | nan | nan | 0 | 0.39375 |
| hard_st_cage | digits | 0 | 0.366667 | 0.648889 | 0.282222 | 9.49008 | 49.2919 | 39.8018 | 0.648889 | 0.648889 | 0 | 49.2919 | 49.2919 | 0 | 11.8545 | -1 | -1 | 0.839844 | 1280 | 4 | 10 | nan | nan | 0 | 0.380469 |
| block_relaxed | digits | 0 | 0.135556 | 0.104444 | 0.0311111 | 2.36402 | 406.427 | 404.063 | 0.135556 | 0.104444 | 0.0311111 | 2.36402 | 406.427 | 404.063 | 2.68405 | -1 | -1 | 0.978125 | 1280 | 4 | 10 | nan | nan | 0 | 0.364063 |
| block_hard_refit | digits | 0 | 0.102222 | 0.244444 | 0.142222 | 36.8284 | 186.351 | 149.523 | 0.704444 | 0.244444 | 0.46 | 2.54753 | 186.351 | 183.803 | 2.69671 | -1 | -1 | 0.989062 | 1280 | 4 | 10 | nan | nan | 0 | 0.501563 |
| block_hard_task_refit | digits | 0 | 0.1 | 0.117778 | 0.0177778 | 48.271 | 69.6792 | 21.4082 | 0.486667 | 0.117778 | 0.368889 | 2.51751 | 69.6792 | 67.1617 | 4.54435 | -1 | -1 | 0.982812 | 1280 | 4 | 10 | nan | nan | 0 | 0.735938 |
