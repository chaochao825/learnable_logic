# Hard-LGN Gap Prototype Summary

Elapsed seconds: 3.49

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
  "batch_size": 256,
  "block_epochs": 15,
  "block_total_epochs": 8,
  "cage_beta": 0.99,
  "cage_tau_max": 3.0,
  "cage_tau_min": 0.5,
  "compat_only": false,
  "data_dir": "/home/spco/data",
  "datasets": [
    "parity6",
    "majority7",
    "random_sparse8"
  ],
  "device": "auto",
  "difflogic_compat_check": false,
  "download_data": false,
  "entropy_coef": 0.001,
  "epochs": 20,
  "eval_batch_size": 2048,
  "exact_truth_max": 12,
  "group_tau": 1.0,
  "gumbel_temp_end": 0.3,
  "gumbel_temp_start": 1.5,
  "hard_st_temp": 1.0,
  "image_max_test": 1000,
  "image_max_train": 4000,
  "inference_bench": false,
  "inference_bench_device": "cpu",
  "inference_bench_repeats": 20,
  "inference_bench_warmup": 1,
  "layers": 3,
  "lr": 0.02,
  "methods": [
    "dlgn",
    "gumbel_st",
    "block_relaxed",
    "block_hard_refit"
  ],
  "mind_gap_scaled": false,
  "optimizer": "adamw",
  "out_dir": "remote_runs/strict_bool_smoke_20260723",
  "quick": true,
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
  "width": 32
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats | activation_inactive_gate_ratio | hard_runtime_domain | hard_runtime_audit_operations | hard_runtime_floating_tensor_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | parity6 | 0 | 0.25 | 0.5 | 0.25 | 0.842596 | 1.32139 | 0.478796 | 0.25 | 0.5 | 0.25 | 0.842596 | 1.32139 | 0.478796 | 1.03299 | -1 | -1 | 0.927083 | 96 | 3 | 11 | nan | nan | 0 | 0.197917 | bool_int | 30 | 0 |
| gumbel_st | parity6 | 0 | 0.25 | 0.3125 | 0.0625 | 1.25894 | 2.18787 | 0.92893 | 0.25 | 0.3125 | 0.0625 | 1.25894 | 2.18787 | 0.92893 | 0.332116 | -1 | -1 | 0.958333 | 96 | 3 | 11 | nan | nan | 0 | 0.239583 | bool_int | 30 | 0 |
| block_relaxed | parity6 | 0 | 0.25 | 0.4375 | 0.1875 | 0.785043 | 1.27663 | 0.491592 | 0.25 | 0.4375 | 0.1875 | 0.785043 | 1.27663 | 0.491592 | 0.049913 | -1 | -1 | 0.947917 | 96 | 3 | 11 | nan | nan | 0 | 0.208333 | bool_int | 30 | 0 |
| block_hard_refit | parity6 | 0 | 0.25 | 0.6875 | 0.4375 | 0.775439 | 1.06178 | 0.286336 | 0.6875 | 0.6875 | 0 | 0.691275 | 1.06178 | 0.370501 | 0.0840311 | -1 | -1 | 0.947917 | 96 | 3 | 11 | nan | nan | 0 | 0.1875 | bool_int | 30 | 0 |
| dlgn | majority7 | 0 | 0.65625 | 0.53125 | 0.125 | 0.66239 | 0.989629 | 0.327238 | 0.65625 | 0.53125 | 0.125 | 0.66239 | 0.989629 | 0.327238 | 0.219568 | -1 | -1 | 0.958333 | 96 | 3 | 10 | nan | nan | 0 | 0.354167 | bool_int | 30 | 0 |
| gumbel_st | majority7 | 0 | 0.5625 | 0.46875 | 0.09375 | 0.717026 | 1.25324 | 0.53621 | 0.5625 | 0.46875 | 0.09375 | 0.717026 | 1.25324 | 0.53621 | 0.302358 | -1 | -1 | 0.979167 | 96 | 3 | 10 | nan | nan | 0 | 0.302083 | bool_int | 30 | 0 |
| block_relaxed | majority7 | 0 | 0.5 | 0.46875 | 0.03125 | 0.698256 | 1.38869 | 0.690438 | 0.5 | 0.46875 | 0.03125 | 0.698256 | 1.38869 | 0.690438 | 0.0570754 | -1 | -1 | 0.958333 | 96 | 3 | 10 | nan | nan | 0 | 0.3125 | bool_int | 30 | 0 |
| block_hard_refit | majority7 | 0 | 0.5 | 0.40625 | 0.09375 | 0.710328 | 1.34517 | 0.634845 | 0.46875 | 0.40625 | 0.0625 | 0.676581 | 1.34517 | 0.668591 | 0.0678588 | -1 | -1 | 0.958333 | 96 | 3 | 10 | nan | nan | 0 | 0.354167 | bool_int | 30 | 0 |
| dlgn | random_sparse8 | 0 | 0.734375 | 0.5625 | 0.171875 | 0.642089 | 1.26755 | 0.625463 | 0.734375 | 0.5625 | 0.171875 | 0.642089 | 1.26755 | 0.625463 | 0.327172 | -1 | -1 | 0.947917 | 96 | 3 | 8 | nan | nan | 0 | 0.09375 | bool_int | 30 | 0 |
| gumbel_st | random_sparse8 | 0 | 0.328125 | 0.3125 | 0.015625 | 1.09963 | 2.46992 | 1.37029 | 0.328125 | 0.3125 | 0.015625 | 1.09963 | 2.46992 | 1.37029 | 0.296745 | -1 | -1 | 0.96875 | 96 | 3 | 8 | nan | nan | 0 | 0.104167 | bool_int | 30 | 0 |
| block_relaxed | random_sparse8 | 0 | 0.671875 | 0.34375 | 0.328125 | 0.683277 | 1.55912 | 0.875843 | 0.671875 | 0.34375 | 0.328125 | 0.683277 | 1.55912 | 0.875843 | 0.0769981 | -1 | -1 | 0.96875 | 96 | 3 | 8 | nan | nan | 0 | 0.1875 | bool_int | 30 | 0 |
| block_hard_refit | random_sparse8 | 0 | 0.421875 | 0.703125 | 0.28125 | 0.694679 | 0.772765 | 0.0780862 | 0.703125 | 0.703125 | 0 | 0.651879 | 0.772765 | 0.120887 | 0.0812034 | -1 | -1 | 0.96875 | 96 | 3 | 8 | nan | nan | 0 | 0.15625 | bool_int | 30 | 0 |
