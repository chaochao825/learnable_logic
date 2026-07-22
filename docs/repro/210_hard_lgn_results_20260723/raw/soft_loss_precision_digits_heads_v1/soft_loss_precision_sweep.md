# Soft-Loss Precision Sweep

This diagnostic prioritizes continuous loss/accuracy and intentionally ignores synthesis cost.

## Configuration

```json
{
  "anneal_train": false,
  "b_values": [
    3,
    4,
    6
  ],
  "batch_size": 512,
  "cosine_lr": true,
  "data_dir": "/home/spco/data",
  "datasets": [
    "digits"
  ],
  "device": "cuda",
  "download_data": false,
  "entropy_weight": 0.0,
  "epochs": 80,
  "estimator": "sinusoidal",
  "eval_batch_size": 4096,
  "eval_every": 0,
  "grad_clip": 0.0,
  "group_tau": 1.0,
  "gumbel_temp_end": 0.3,
  "gumbel_temp_start": 1.5,
  "head_norm": true,
  "head_types": [
    "groupsum",
    "affine_groupsum",
    "final_linear",
    "concat_linear",
    "concat_mlp",
    "input_linear"
  ],
  "image_max_test": 3000,
  "image_max_train": 12000,
  "include_input_in_concat": true,
  "init": "residual",
  "init_strength": 0.98,
  "label_smoothing": 0.0,
  "layers": 4,
  "lr": 0.003,
  "lr_floor": 0.05,
  "mlp_hidden": 2048,
  "out_dir": "runs/soft_loss_precision_digits_heads_v1",
  "quick": false,
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
  "width": 800
}
```

## Best Rows By Dataset

| dataset | best_head | seed | threshold_levels | b | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| digits | final_linear | 0 | 1 | 6 | 800 | 0.924444 | 0.259466 | 0.911111 | 0.296302 | 0.0133333 | 214410 |

## Component Deltas Vs GroupSum

| dataset | seed | threshold_levels | b | width | head | soft_loss_delta | soft_acc_delta |
|---|---:|---:|---:|---:|---|---:|---:|
| digits | 0 | 1 | 3 | 800 | affine_groupsum | -0.00903714 | 0.0111111 |
| digits | 0 | 1 | 3 | 800 | concat_linear | -0.374432 | 0.166667 |
| digits | 0 | 1 | 3 | 800 | concat_mlp | -0.51624 | 0.204444 |
| digits | 0 | 1 | 3 | 800 | final_linear | -0.535744 | 0.184444 |
| digits | 0 | 1 | 3 | 800 | input_linear | -0.407608 | 0.171111 |
| digits | 0 | 1 | 4 | 800 | affine_groupsum | -0.0136323 | -0.00222222 |
| digits | 0 | 1 | 4 | 800 | concat_linear | -0.236681 | 0.104444 |
| digits | 0 | 1 | 4 | 800 | concat_mlp | -0.295081 | 0.133333 |
| digits | 0 | 1 | 4 | 800 | final_linear | -0.382352 | 0.12 |
| digits | 0 | 1 | 4 | 800 | input_linear | -0.248767 | 0.113333 |
| digits | 0 | 1 | 6 | 800 | affine_groupsum | -0.0101069 | 0.00666667 |
| digits | 0 | 1 | 6 | 800 | concat_linear | -0.0327933 | 0.0266667 |
| digits | 0 | 1 | 6 | 800 | concat_mlp | -0.154073 | 0.0622222 |
| digits | 0 | 1 | 6 | 800 | final_linear | -0.194086 | 0.0488889 |
| digits | 0 | 1 | 6 | 800 | input_linear | -0.0453754 | 0.0244444 |

## All Rows

| dataset | seed | threshold_levels | b | head | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | train_time | params | unused_gate_ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| digits | 0 | 1 | 6 | final_linear | 800 | 0.924444 | 0.259466 | 0.911111 | 0.296302 | 0.0133333 | 74.173 | 214410 | 0.226875 |
| digits | 0 | 1 | 3 | final_linear | 800 | 0.917778 | 0.28292 | 0.915556 | 0.299264 | 0.00222222 | 9.37184 | 35210 | 0.23625 |
| digits | 0 | 1 | 4 | final_linear | 800 | 0.917778 | 0.289193 | 0.9 | 0.323219 | 0.0177778 | 16.6757 | 60810 | 0.184375 |
| digits | 0 | 1 | 6 | concat_mlp | 800 | 0.937778 | 0.299479 | 0.94 | 0.285351 | 0.00222222 | 8.61142 | 6918538 | 0.226875 |
| digits | 0 | 1 | 3 | concat_mlp | 800 | 0.937778 | 0.302423 | 0.935556 | 0.291644 | 0.00222222 | 2.31443 | 6739338 | 0.23625 |
| digits | 0 | 1 | 4 | concat_mlp | 800 | 0.931111 | 0.376463 | 0.931111 | 0.362886 | 0 | 3.142 | 6764938 | 0.184375 |
| digits | 0 | 1 | 6 | input_linear | 800 | 0.9 | 0.408176 | 0.9 | 0.408176 | 0 | 2.30238 | 205578 | 0.226875 |
| digits | 0 | 1 | 3 | input_linear | 800 | 0.904444 | 0.411055 | 0.904444 | 0.411055 | 0 | 0.988106 | 26378 | 0.23625 |
| digits | 0 | 1 | 6 | concat_linear | 800 | 0.902222 | 0.420758 | 0.9 | 0.439946 | 0.00222222 | 8.09089 | 243978 | 0.226875 |
| digits | 0 | 1 | 4 | input_linear | 800 | 0.911111 | 0.422778 | 0.911111 | 0.422778 | 0 | 1.02227 | 51978 | 0.184375 |
| digits | 0 | 1 | 4 | concat_linear | 800 | 0.902222 | 0.434864 | 0.9 | 0.450451 | 0.00222222 | 2.6639 | 90378 | 0.184375 |
| digits | 0 | 1 | 6 | affine_groupsum | 800 | 0.882222 | 0.443444 | 0.131111 | 5.47092 | 0.751111 | 77.9114 | 204820 | 0.226875 |
| digits | 0 | 1 | 3 | concat_linear | 800 | 0.9 | 0.444232 | 0.9 | 0.450033 | 0 | 2.11396 | 64778 | 0.23625 |
| digits | 0 | 1 | 6 | groupsum | 800 | 0.875556 | 0.453551 | 0.131111 | 6.57786 | 0.744444 | 82.2228 | 204800 | 0.226875 |
| digits | 0 | 1 | 4 | affine_groupsum | 800 | 0.795556 | 0.657912 | 0.122222 | 5.62802 | 0.673333 | 27.8327 | 51220 | 0.184375 |
| digits | 0 | 1 | 4 | groupsum | 800 | 0.797778 | 0.671545 | 0.124444 | 6.55192 | 0.673333 | 33.2327 | 51200 | 0.184375 |
| digits | 0 | 1 | 3 | affine_groupsum | 800 | 0.744444 | 0.809626 | 0.0844444 | 5.13736 | 0.66 | 19.7185 | 25620 | 0.23625 |
| digits | 0 | 1 | 3 | groupsum | 800 | 0.733333 | 0.818664 | 0.0844444 | 6.06744 | 0.648889 | 24.468 | 25600 | 0.23625 |
