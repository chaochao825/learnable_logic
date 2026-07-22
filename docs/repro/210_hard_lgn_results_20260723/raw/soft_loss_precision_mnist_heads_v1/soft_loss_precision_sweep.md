# Soft-Loss Precision Sweep

This diagnostic prioritizes continuous loss/accuracy and intentionally ignores synthesis cost.

## Configuration

```json
{
  "anneal_train": false,
  "b_values": [
    4,
    6
  ],
  "batch_size": 512,
  "cosine_lr": true,
  "data_dir": "/home/spco/data",
  "datasets": [
    "binarized_mnist"
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
  "out_dir": "runs/soft_loss_precision_mnist_heads_v1",
  "quick": false,
  "seeds": [
    0
  ],
  "temp_end": 0.2,
  "temp_eval": 1.0,
  "temp_start": 2.0,
  "threshold_levels": 1,
  "threshold_levels_list": [
    1,
    3
  ],
  "train_forward_mode": "continuous",
  "warmup_epochs": 0,
  "warmup_mode": "continuous",
  "weight_decay": 1e-05,
  "width": 1000
}
```

## Best Rows By Dataset

| dataset | best_head | seed | threshold_levels | b | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | params |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binarized_mnist | concat_mlp | 0 | 3 | 4 | 1000 | 0.965 | 0.176081 | 0.964333 | 0.183085 | 0.000666667 | 13108138 |

## Component Deltas Vs GroupSum

| dataset | seed | threshold_levels | b | width | head | soft_loss_delta | soft_acc_delta |
|---|---:|---:|---:|---:|---|---:|---:|
| binarized_mnist | 0 | 1 | 4 | 1000 | concat_linear | 0.168847 | -0.00533333 |
| binarized_mnist | 0 | 1 | 4 | 1000 | concat_mlp | -0.0567772 | 0.0343333 |
| binarized_mnist | 0 | 1 | 4 | 1000 | final_linear | 0.0624177 | 0.01 |
| binarized_mnist | 0 | 1 | 4 | 1000 | input_linear | 0.213996 | -0.0343333 |
| binarized_mnist | 0 | 1 | 6 | 1000 | concat_linear | 0.166179 | -0.0113333 |
| binarized_mnist | 0 | 1 | 6 | 1000 | concat_mlp | 0.00183925 | 0.0163333 |
| binarized_mnist | 0 | 1 | 6 | 1000 | final_linear | 0.070474 | 0 |
| binarized_mnist | 0 | 1 | 6 | 1000 | input_linear | 0.248593 | -0.0493333 |
| binarized_mnist | 0 | 3 | 4 | 1000 | concat_linear | 0.235627 | -0.012 |
| binarized_mnist | 0 | 3 | 4 | 1000 | concat_mlp | -0.0658827 | 0.035 |
| binarized_mnist | 0 | 3 | 4 | 1000 | final_linear | 0.10149 | 0 |
| binarized_mnist | 0 | 3 | 4 | 1000 | input_linear | 0.256907 | -0.0313333 |
| binarized_mnist | 0 | 3 | 6 | 1000 | concat_linear | 0.235945 | -0.0106667 |
| binarized_mnist | 0 | 3 | 6 | 1000 | concat_mlp | 0.00352173 | 0.027 |
| binarized_mnist | 0 | 3 | 6 | 1000 | final_linear | 0.0749641 | 0.00766667 |
| binarized_mnist | 0 | 3 | 6 | 1000 | input_linear | 0.302325 | -0.037 |

## All Rows

| dataset | seed | threshold_levels | b | head | width | soft_acc | soft_loss | hard_acc | hard_loss | acc_gap | train_time | params | unused_gate_ratio |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| binarized_mnist | 0 | 3 | 4 | concat_mlp | 1000 | 0.965 | 0.176081 | 0.964333 | 0.183085 | 0.000666667 | 37.8192 | 13108138 | 0.1645 |
| binarized_mnist | 0 | 1 | 4 | concat_mlp | 1000 | 0.963 | 0.176121 | 0.961 | 0.173047 | 0.002 | 35.6866 | 9893738 | 0.177 |
| binarized_mnist | 0 | 1 | 6 | groupsum | 1000 | 0.943 | 0.19542 | 0.354333 | 3.04589 | 0.588667 | 3587.07 | 256000 | 0.0945 |
| binarized_mnist | 0 | 1 | 6 | concat_mlp | 1000 | 0.959333 | 0.197259 | 0.959 | 0.200168 | 0.000333333 | 91.6095 | 10085738 | 0.18825 |
| binarized_mnist | 0 | 3 | 6 | groupsum | 1000 | 0.935 | 0.200956 | 0.333333 | 3.28568 | 0.601667 | 3586.04 | 256000 | 0.08475 |
| binarized_mnist | 0 | 3 | 6 | concat_mlp | 1000 | 0.962 | 0.204478 | 0.957333 | 0.209698 | 0.00466667 | 94.3064 | 13300138 | 0.1805 |
| binarized_mnist | 0 | 1 | 4 | groupsum | 1000 | 0.928667 | 0.232898 | 0.279333 | 3.30729 | 0.649333 | 900.629 | 64000 | 0.1075 |
| binarized_mnist | 0 | 3 | 4 | groupsum | 1000 | 0.93 | 0.241964 | 0.339333 | 3.44492 | 0.590667 | 900.776 | 64000 | 0.09475 |
| binarized_mnist | 0 | 1 | 6 | final_linear | 1000 | 0.943 | 0.265894 | 0.668 | 1.42204 | 0.275 | 3418.24 | 268010 | 0.141 |
| binarized_mnist | 0 | 3 | 6 | final_linear | 1000 | 0.942667 | 0.27592 | 0.674 | 1.26843 | 0.268667 | 3291.75 | 268010 | 0.127 |
| binarized_mnist | 0 | 1 | 4 | final_linear | 1000 | 0.938667 | 0.295315 | 0.648667 | 1.73595 | 0.29 | 861.58 | 76010 | 0.13475 |
| binarized_mnist | 0 | 3 | 4 | final_linear | 1000 | 0.93 | 0.343453 | 0.659333 | 1.51224 | 0.270667 | 884.848 | 76010 | 0.1185 |
| binarized_mnist | 0 | 1 | 6 | concat_linear | 1000 | 0.931667 | 0.361599 | 0.658667 | 2.35833 | 0.273 | 1118.84 | 313418 | 0.18575 |
| binarized_mnist | 0 | 1 | 4 | concat_linear | 1000 | 0.923333 | 0.401745 | 0.702667 | 1.52403 | 0.220667 | 277.436 | 121418 | 0.17125 |
| binarized_mnist | 0 | 3 | 6 | concat_linear | 1000 | 0.924333 | 0.436901 | 0.635667 | 2.64295 | 0.288667 | 1130 | 332234 | 0.1805 |
| binarized_mnist | 0 | 1 | 6 | input_linear | 1000 | 0.893667 | 0.444013 | 0.893667 | 0.444013 | 0 | 25.2485 | 265418 | 0.18825 |
| binarized_mnist | 0 | 1 | 4 | input_linear | 1000 | 0.894333 | 0.446894 | 0.894333 | 0.446894 | 0 | 11.0202 | 73418 | 0.177 |
| binarized_mnist | 0 | 3 | 4 | concat_linear | 1000 | 0.918 | 0.47759 | 0.718 | 1.71 | 0.2 | 264.003 | 140234 | 0.1645 |
| binarized_mnist | 0 | 3 | 4 | input_linear | 1000 | 0.898667 | 0.498871 | 0.898667 | 0.498871 | 0 | 12.3362 | 92234 | 0.1645 |
| binarized_mnist | 0 | 3 | 6 | input_linear | 1000 | 0.898 | 0.503282 | 0.898 | 0.503282 | 0 | 26.6698 | 284234 | 0.1805 |
