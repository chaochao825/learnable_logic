# ViT-LGN Token Mixer And Attention-K Summary

## Scope

This document summarizes the ViT-LGN work imported from the 210 server into
`chaochao825/learnable_logic` on 2026-07-10. It covers two related trees:

- `/home/spco/sow_linear/ViT-LGN_goal6plus`: original ViT-LGN + LogicFFN
  experiments, including the CLS-only token-mixer ablation.
- `/home/spco/sow_linear/ViT-LGN_attention_clean_20260629`: a clean
  attention-k implementation and longer 200k training runs.

The repository import intentionally excludes raw `data/`, `logs/`, `runs/`,
compiled `.so` files, `build/`, and `__pycache__/` directories. Only source
code, launch scripts, config JSONs, and compact result summaries are included.

## Model Families

### 1. Goal6plus ViT-LGN baseline

The current baseline is a CIFAR-10 ViT-like model with a logic FFN path:

| item | value |
|---|---|
| dataset | CIFAR-10 |
| image / patch | `32x32`, patch `4` |
| depth / width / heads | depth `6`, embed dim `192`, heads `6` |
| FFN | `ffn_type=logic`, `logic_ffn_layers=2` |
| logic encoding | thermometer, `logic_n_thresholds=7`, `logic_act_fn=SIN01` |
| connection mode | fixed random logic connections |
| augmentation | disabled for the best d6/h6 8k baseline |
| training length | `teacher_iters=8000` |
| eval split | full CIFAR-10 test set |

Best full-attention baseline from
`vit_tiny_scale_full_d6_h6_noaug_8000_20260628`:

| soft acc | hard acc | gate count | peak memory MB |
|---:|---:|---:|---:|
| `0.6249` | `0.6243` | `16128` | `1690` |

### 2. CLS-only top-k token mixers

The CLS-only ablation replaces only selected attention blocks. It does not
change LogicFFN, light logic gates, GroupSum, training loss, or the classifier.
Patch-token outputs from the replacement mixer stay unchanged except for the
outer residual path; only the CLS update is replaced.

Implemented mixers:

- `CLSStaticTopKMixer`: learned static patch-position score per head; top-k
  selected patch value tokens are aggregated by mean or learned scalar weights.
- `CLSDynamicTopKRouter`: CLS-to-patch q/k score using projection layers; no
  full `N x N` attention matrix is formed.

Important 8k results against the d6/h6 full-attention baseline:

| setting | top-k | variant | hard acc | delta vs full attention |
|---|---:|---|---:|---:|
| full attention | - | baseline | `0.6243` | `0.0000` |
| replace `b4,b5` | `8` | `static_weighted` | `0.6360` | `+0.0117` |
| replace `b4,b5` | `4` | `static_mean` | `0.6358` | `+0.0115` |
| replace `b5` | `4` | `static_weighted` | `0.6308` | `+0.0065` |
| replace `b5` | `8` | `static_weighted` | `0.6308` | `+0.0065` |

Interpretation:

- Late-block CLS-only static routing is stable and can slightly exceed the
  8k full-attention baseline in this run.
- Dynamic q/k routing is generally weaker in the first sweep, especially for
  larger top-k with mean aggregation.
- Mean and weighted aggregation are close enough that both remain useful
  ablation points, but the best single 8k row is `b4,b5/static_weighted/K=8`.

### 3. Attention-clean top-k attention model

The attention-clean branch moves the token selection into the attention path
itself with a top-k attention setting (`attention_k`). This branch uses a
separate training script and is the source of the 200k runs.

Common 200k configuration:

| item | value |
|---|---|
| dataset | CIFAR-10 |
| image / patch | `32x32`, patch `4` |
| depth / width / heads | depth `6`, embed dim `192`, heads `6` |
| batch size | `512` |
| iterations | `200000` |
| optimizer LR | `5e-4` constant unless noted |
| eval cadence | every `5000` steps for K=8, every `10000` for K=4 |
| top-k implementation | `torch-topk` |
| majority path | `--fast-vote`, count surrogate |
| checkpoints | disabled (`--no-save-checkpoints`) |

Final attention-clean results:

| run | best valid/eval | test/eval | final fulltrain/eval |
|---|---:|---:|---:|
| `h6_k8_defaulttemp_d6e192_aug_const_200k_20260630` | `0.7913` | `0.7893` | `0.9284` |
| `h6_k8_defaulttemp_d6e192_noaug_warmstay_200k_20260630` | `0.7219` | `0.7193` | `0.9415` |
| `h6_k8_defaulttemp_d6e192_noaug_const_200k_20260630` | `0.7208` | `0.7154` | `0.8765` |
| `h6_k4_defaulttemp_d6e192_noaug_const_200k_20260630` | `0.7051` | `0.6893` | `0.9834` |
| historical `h6_k8_defaulttemp_d6e192_noaug_20k_20260629` | `0.7172` | `0.7158` | `0.9313` |

Interpretation:

- Extending to 200k iterations substantially improves the augmented K=8 run:
  test/eval reaches `0.7893`.
- `attention_k=8` is consistently stronger than `attention_k=4` in this
  configuration.
- The no-augmentation K=8 200k constant-LR run does not materially beat the
  historical no-augmentation 20k run on test/eval (`0.7154` vs `0.7158`).
- The K=4 200k run shows high full-train eval (`0.9834`) but weaker test/eval
  (`0.6893`), indicating overfit or insufficient token fan-in.

## Imported Repository Layout

```text
vit_lgn/
  README.md
  goal6plus/
    *.py, launch/run scripts
    local_difflogic/
    artifacts/
      vit_tiny_scale_full_d6_h6_noaug_8000_{config,summary}.json
      cls_topk_b4b5_8000_{config,summary,results,route_diagnostics}.*
      cls_topk_b5_8000_{config,summary,results,route_diagnostics}.*
  attention_clean/
    train_logic_vit_tiny.py
    vit_tiny_attention_logic.py
    src/
    tools/
    local_difflogic/
    artifacts/
      accuracy_summary_final_20260710.json
      accuracy_summary_20260630.json
      accuracy_summary_200k_live_20260630.json

docs/
  vit_lgn_method_results_20260710.md
  tables/
    vit_lgn_cls_topk_results_20260710.csv
    vit_lgn_attention_clean_accuracy_20260710.csv
```

## Checkpoints And Saved Weights

No reusable model weight files were found in either source tree at import time.
The scan covered these extensions:

- `.pt`
- `.pth`
- `.ckpt`
- `.bin`
- `.safetensors`

The attention-clean 200k commands explicitly used `--no-save-checkpoints`, and
therefore only logs and metric summaries are available for those runs. The
large raw logs remain on 210 under:

- `/home/spco/sow_linear/ViT-LGN_attention_clean_20260629/runs_attention_clean`
- `/home/spco/sow_linear/ViT-LGN_goal6plus/runs`

## Reproduction Pointers

For the goal6plus CLS-only ablation, start from:

```bash
cd /home/spco/sow_linear/ViT-LGN_goal6plus
export PYTHONPATH=/home/wangmeiqi/pyf
/home/wangmeiqi/anaconda3/envs/convlogic/bin/python cls_topk_token_mixer_ablation.py \
  --config-json runs/vit_tiny_scale_full_d6_h6_noaug_8000_20260628/20260628_034705/config.json \
  --teacher-iters 8000 \
  --eval-split test \
  --eval-max-batches -1 \
  --route-stats-batches 20 \
  --block-settings b4,b5 \
  --topk-list 4,8,16 \
  --mixer-variants static_mean,static_weighted,dynamic_mean,dynamic_weighted
```

For the attention-clean 200k family, the representative best command is:

```bash
cd /home/spco/sow_linear/ViT-LGN_attention_clean_20260629
/home/wangmeiqi/anaconda3/envs/vit310/bin/python train_logic_vit_tiny.py \
  --num-iterations 200000 \
  --batch-size 512 \
  --depth 6 \
  --embed-dim 192 \
  --num-heads 6 \
  --attention-k 8 \
  --eval-freq 5000 \
  --ext-eval-freq 999999 \
  --print-freq 1000 \
  --num-workers 4 \
  --no-save-checkpoints \
  --topk-impl torch-topk \
  --fast-vote
```

Omit `--no-augment` for the best augmented run. Add `--no-augment` for the
no-augmentation controls.
