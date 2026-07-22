# Soft-Loss Precision Sweep

This diagnostic prioritizes continuous loss/accuracy and intentionally ignores synthesis cost.

## Configuration

```json
{
  "anneal_train": false,
  "b_values": [
    3
  ],
  "batch_size": 256,
  "cosine_lr": false,
  "data_dir": "/home/spco/data",
  "datasets": [
    "digits"
  ],
  "device": "cuda",
  "download_data": false,
  "entropy_weight": 0.0,
  "epochs": 5,
  "estimator": "sinusoidal",
  "eval_batch_size": 2048,
  "eval_every": 0,
  "grad_clip": 0.0,
  "group_tau": 1.0,
  "gumbel_temp_end": 0.3,
  "gumbel_temp_start": 1.5,
  "head_norm": false,
  "head_types": [
    "groupsum",
    "affine_groupsum",
    "final_linear"
  ],
  "image_max_test": 500,
  "image_max_train": 1000,
  "include_input_in_concat": false,
  "init": "residual",
  "init_strength": 0.98,
  "label_smoothing": 0.0,
  "layers": 3,
  "lr": 0.003,
  "lr_floor": 0.05,
  "mlp_hidden": 1024,
  "out_dir": "runs/soft_loss_precision_quick_v1",
  "quick": true,
  "seeds": [
    0
  ],
  "temp_end": 0.2,
  "temp_eval": 1.0,
  "temp_start": 2.0,
  "threshold_levels": 1,
  "threshold_levels_list": [
    1
  ],
  "train_forward_mode": "continuous",
  "warmup_epochs": 0,
  "warmup_mode": "continuous",
  "weight_decay": 1e-05,
  "width": 160
}
```

## Best Rows By Dataset

| dataset | best_head | seed | threshold_levels | b | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| digits | final_linear | 0 | 1 | 3 | 160 | 0.815556 | 1.20761 | 0.811111 | 1.16836 | 0.00444444 | 5450 |

## Component Deltas Vs GroupSum

| dataset | seed | threshold_levels | b | width | head | soft_loss_delta | soft_acc_delta |
|---|---:|---:|---:|---:|---|---:|---:|
| digits | 0 | 1 | 3 | 160 | affine_groupsum | -0.250879 | -0.0177778 |
| digits | 0 | 1 | 3 | 160 | final_linear | -1.98972 | 0.728889 |

## All Rows

| dataset | seed | threshold_levels | b | head | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | train_time | params | unused_gate_ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| digits | 0 | 1 | 3 | final_linear | 160 | 0.815556 | 1.20761 | 0.811111 | 1.16836 | 0.00444444 | 1.58619 | 5450 | 0.15625 |
| digits | 0 | 1 | 3 | affine_groupsum | 160 | 0.0688889 | 2.94645 | 0.0622222 | 3.57301 | 0.00666667 | 0.123353 | 3860 | 0.15625 |
| digits | 0 | 1 | 3 | groupsum | 160 | 0.0866667 | 3.19733 | 0.0711111 | 3.93503 | 0.0155556 | 0.346468 | 3840 | 0.15625 |
