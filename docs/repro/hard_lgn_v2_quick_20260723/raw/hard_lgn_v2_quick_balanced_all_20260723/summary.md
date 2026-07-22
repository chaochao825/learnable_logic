# Hard-LGN Gap Prototype Summary

Elapsed seconds: 4.66

Metric notes:

- `soft_acc`/`soft_loss` evaluate all trained relaxed blocks in soft mode.
- `discrete_acc`/`discrete_loss` evaluate the corresponding hard network: argmax gates for DLGN-style methods and fitted gates for block-hard methods.
- `path_*` metrics evaluate the method-native training path. Hard-ST uses the deterministic hard forward; block-hard methods use hard frozen prefixes plus the final relaxed block.
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
  "batch_size": 256,
  "block_epochs": 15,
  "block_total_epochs": 20,
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
  "device": "cpu",
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
    "dlgn_anneal",
    "gumbel_st",
    "gumbel_st_cage",
    "hard_st",
    "hard_st_cage",
    "block_relaxed",
    "block_hard_refit",
    "block_hard_task_refit"
  ],
  "out_dir": "/home/spco/sow_linear/learnable_logic_integration_runs/hard_lgn_v2_quick_balanced_all_20260723",
  "quick": true,
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
  "width": 32
}
```

| method | dataset | seed | soft_acc | discrete_acc | acc_gap | soft_loss | discrete_loss | loss_gap | path_soft_acc | path_discrete_acc | path_acc_gap | path_soft_loss | path_discrete_loss | path_loss_gap | train_time | epochs_to_target | time_to_target | unused_gate_ratio | gate_count | depth | fanout_max | soft_inference_samples_per_sec | discrete_inference_samples_per_sec | inference_bench_repeats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dlgn | parity6 | 0 | 0.25 | 0.5 | 0.25 | 0.842597 | 1.32139 | 0.478796 | 0.25 | 0.5 | 0.25 | 0.842597 | 1.32139 | 0.478796 | 0.170024 | -1 | -1 | 0.197917 | 96 | 3 | 11 | nan | nan | 0 |
| dlgn_anneal | parity6 | 0 | 0.1875 | 0.375 | 0.1875 | 0.944189 | 1.23023 | 0.286037 | 0.1875 | 0.375 | 0.1875 | 0.944189 | 1.23023 | 0.286037 | 0.0967451 | -1 | -1 | 0.197917 | 96 | 3 | 11 | nan | nan | 0 |
| gumbel_st | parity6 | 0 | 0.25 | 0.375 | 0.125 | 1.32042 | 2.26855 | 0.948125 | 0.25 | 0.375 | 0.125 | 1.32042 | 2.26855 | 0.948125 | 0.0927424 | -1 | -1 | 0.229167 | 96 | 3 | 11 | nan | nan | 0 |
| gumbel_st_cage | parity6 | 0 | 0.75 | 0.75 | 0 | 0.671472 | 0.692245 | 0.0207729 | 0.75 | 0.75 | 0 | 0.671472 | 0.692245 | 0.0207729 | 0.09278 | -1 | -1 | 0.239583 | 96 | 3 | 11 | nan | nan | 0 |
| hard_st | parity6 | 0 | 0.25 | 0.3125 | 0.0625 | 0.794598 | 1.56197 | 0.767377 | 0.3125 | 0.3125 | 0 | 1.56197 | 1.56197 | 0 | 0.0900499 | -1 | -1 | 0.260417 | 96 | 3 | 11 | nan | nan | 0 |
| hard_st_cage | parity6 | 0 | 0.25 | 0.4375 | 0.1875 | 0.703243 | 1.22394 | 0.520697 | 0.4375 | 0.4375 | 0 | 1.22394 | 1.22394 | 0 | 0.0901812 | -1 | -1 | 0.291667 | 96 | 3 | 11 | nan | nan | 0 |
| block_relaxed | parity6 | 0 | 0.25 | 0.4375 | 0.1875 | 0.838276 | 1.27511 | 0.43683 | 0.25 | 0.4375 | 0.1875 | 0.838276 | 1.27511 | 0.43683 | 0.0442577 | -1 | -1 | 0.208333 | 96 | 3 | 11 | nan | nan | 0 |
| block_hard_refit | parity6 | 0 | 0.25 | 0.4375 | 0.1875 | 0.881922 | 1.36657 | 0.484652 | 0.375 | 0.4375 | 0.0625 | 0.878282 | 1.36657 | 0.488292 | 0.052259 | -1 | -1 | 0.197917 | 96 | 3 | 11 | nan | nan | 0 |
| block_hard_task_refit | parity6 | 0 | 0.25 | 0.375 | 0.125 | 0.79113 | 0.994476 | 0.203346 | 0.25 | 0.375 | 0.125 | 0.871924 | 0.994476 | 0.122552 | 0.438461 | -1 | -1 | 0.375 | 96 | 3 | 11 | nan | nan | 0 |
| dlgn | majority7 | 0 | 0.65625 | 0.53125 | 0.125 | 0.66239 | 0.989629 | 0.327238 | 0.65625 | 0.53125 | 0.125 | 0.66239 | 0.989629 | 0.327238 | 0.103295 | -1 | -1 | 0.354167 | 96 | 3 | 10 | nan | nan | 0 |
| dlgn_anneal | majority7 | 0 | 0.875 | 0.625 | 0.25 | 0.397795 | 0.84377 | 0.445974 | 0.875 | 0.625 | 0.25 | 0.397795 | 0.84377 | 0.445974 | 0.112263 | -1 | -1 | 0.364583 | 96 | 3 | 10 | nan | nan | 0 |
| gumbel_st | majority7 | 0 | 0.5625 | 0.4375 | 0.125 | 0.819857 | 1.17176 | 0.351907 | 0.5625 | 0.4375 | 0.125 | 0.819857 | 1.17176 | 0.351907 | 0.108142 | -1 | -1 | 0.322917 | 96 | 3 | 10 | nan | nan | 0 |
| gumbel_st_cage | majority7 | 0 | 0.5 | 0.40625 | 0.09375 | 0.700995 | 1.50251 | 0.801517 | 0.5 | 0.40625 | 0.09375 | 0.700995 | 1.50251 | 0.801517 | 0.108607 | -1 | -1 | 0.322917 | 96 | 3 | 10 | nan | nan | 0 |
| hard_st | majority7 | 0 | 0.6875 | 0.71875 | 0.03125 | 0.680194 | 0.774177 | 0.0939826 | 0.71875 | 0.71875 | 0 | 0.774177 | 0.774177 | 0 | 0.105709 | -1 | -1 | 0.28125 | 96 | 3 | 10 | nan | nan | 0 |
| hard_st_cage | majority7 | 0 | 0.5 | 0.59375 | 0.09375 | 0.692786 | 0.938673 | 0.245888 | 0.59375 | 0.59375 | 0 | 0.938673 | 0.938673 | 0 | 0.10644 | -1 | -1 | 0.28125 | 96 | 3 | 10 | nan | nan | 0 |
| block_relaxed | majority7 | 0 | 0.5 | 0.53125 | 0.03125 | 0.68967 | 1.46544 | 0.775774 | 0.5 | 0.53125 | 0.03125 | 0.68967 | 1.46544 | 0.775774 | 0.0524715 | -1 | -1 | 0.34375 | 96 | 3 | 10 | nan | nan | 0 |
| block_hard_refit | majority7 | 0 | 0.5 | 0.75 | 0.25 | 0.692415 | 0.609252 | 0.0831623 | 0.75 | 0.75 | 0 | 0.614706 | 0.609252 | 0.00545317 | 0.0611681 | -1 | -1 | 0.322917 | 96 | 3 | 10 | nan | nan | 0 |
| block_hard_task_refit | majority7 | 0 | 0.5 | 0.75 | 0.25 | 0.723962 | 0.440861 | 0.283101 | 0.8125 | 0.75 | 0.0625 | 0.544421 | 0.440861 | 0.10356 | 0.501131 | -1 | -1 | 0.3125 | 96 | 3 | 10 | nan | nan | 0 |
| dlgn | random_sparse8 | 0 | 0.734375 | 0.5625 | 0.171875 | 0.642089 | 1.26755 | 0.625463 | 0.734375 | 0.5625 | 0.171875 | 0.642089 | 1.26755 | 0.625463 | 0.165644 | -1 | -1 | 0.09375 | 96 | 3 | 8 | nan | nan | 0 |
| dlgn_anneal | random_sparse8 | 0 | 0.765625 | 0.703125 | 0.0625 | 0.433494 | 0.734755 | 0.301261 | 0.765625 | 0.703125 | 0.0625 | 0.433494 | 0.734755 | 0.301261 | 0.17829 | -1 | -1 | 0.0833333 | 96 | 3 | 8 | nan | nan | 0 |
| gumbel_st | random_sparse8 | 0 | 0.328125 | 0.421875 | 0.09375 | 1.17658 | 1.82618 | 0.649606 | 0.328125 | 0.421875 | 0.09375 | 1.17658 | 1.82618 | 0.649606 | 0.163646 | -1 | -1 | 0.166667 | 96 | 3 | 8 | nan | nan | 0 |
| gumbel_st_cage | random_sparse8 | 0 | 0.671875 | 0.28125 | 0.390625 | 0.690981 | 1.8491 | 1.15812 | 0.671875 | 0.28125 | 0.390625 | 0.690981 | 1.8491 | 1.15812 | 0.133847 | -1 | -1 | 0.166667 | 96 | 3 | 8 | nan | nan | 0 |
| hard_st | random_sparse8 | 0 | 0.671875 | 0.734375 | 0.0625 | 0.633846 | 0.558813 | 0.0750323 | 0.734375 | 0.734375 | 0 | 0.558813 | 0.558813 | 0 | 0.165509 | -1 | -1 | 0.197917 | 96 | 3 | 8 | nan | nan | 0 |
| hard_st_cage | random_sparse8 | 0 | 0.671875 | 0.78125 | 0.109375 | 0.651245 | 0.41668 | 0.234565 | 0.78125 | 0.78125 | 0 | 0.41668 | 0.41668 | 0 | 0.169301 | -1 | -1 | 0.197917 | 96 | 3 | 8 | nan | nan | 0 |
| block_relaxed | random_sparse8 | 0 | 0.671875 | 0.328125 | 0.34375 | 0.672276 | 2.59353 | 1.92125 | 0.671875 | 0.328125 | 0.34375 | 0.672276 | 2.59353 | 1.92125 | 0.252826 | -1 | -1 | 0.21875 | 96 | 3 | 8 | nan | nan | 0 |
| block_hard_refit | random_sparse8 | 0 | 0.671875 | 0.640625 | 0.03125 | 0.676172 | 0.846853 | 0.170681 | 0.59375 | 0.640625 | 0.046875 | 0.672654 | 0.846853 | 0.174199 | 0.097809 | -1 | -1 | 0.25 | 96 | 3 | 8 | nan | nan | 0 |
| block_hard_task_refit | random_sparse8 | 0 | 0.671875 | 0.640625 | 0.03125 | 0.640614 | 0.553367 | 0.0872467 | 0.515625 | 0.640625 | 0.125 | 0.724613 | 0.553367 | 0.171246 | 0.588395 | -1 | -1 | 0.34375 | 96 | 3 | 8 | nan | nan | 0 |
