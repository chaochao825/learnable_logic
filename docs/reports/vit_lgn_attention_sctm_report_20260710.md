# ViT-LGN Attention And SCTM Integration Report (2026-07-10)

## Scope

This report merges the ViT-LGN token-mixer work into the clean `learnable_logic` publish tree. It covers two related tracks run on server 210:

- `attention-clean`: a Boolean-attention branch using packed XNOR/popcount Top-K routing and selector-mask majority value aggregation.
- `goal6plus_sctm_scale`: a Sparse CLS Patch Token Mixer (SCTM) branch using hard Top-K sparse CLS reads, low-bit weighted value aggregation, local patch mixing, and auxiliary accumulator/value-discretization probes.

The raw experiment trees stay on 210. This repository includes source snapshots, concise tables, and method/result documentation, but not datasets, log streams, diagnostic histograms, or large `.pt` checkpoints.

## Repository Merge Layout

- `vit_lgn/attention_clean/`: source snapshot for the packed-XNOR Top-K selector-majority attention implementation.
- `vit_lgn/sctm_scale/`: source snapshot for SCTM, auxiliary accumulator, and value-discretization experiments.
- `docs/tables/vit_lgn_attention_clean_200k.csv`: parsed 200k attention-clean results.
- `docs/tables/vit_lgn_sctm_results.csv`: selected SCTM/auxiliary run summaries parsed from `summary.json`.
- `docs/tables/vit_lgn_weight_manifest_210.csv`: checkpoint inventory and 210 paths.

## Method 1: Attention-Clean Boolean Attention

The attention-clean branch replaces dense softmax attention with a hard-routing Boolean-style attention module:

1. Standard ViT attention wrappers still form continuous Q/K/V projections.
2. Q/K/V are thermometer encoded with straight-through estimation.
3. Q/K similarity is computed with packed XNOR + popcount.
4. A hard Top-K mask is applied during training and evaluation.
5. Selected V columns are aggregated by selector-mask majority; `--fast-vote` uses a matmul implementation for the hard majority forward path.

This is not softmax quantization. The inductive bias changes from dense continuous weighted averaging to discrete content-addressed selection plus Boolean voting. Evaluation and inference use the same hard path, so the valid eval/inference gap is zero in the parsed runs.

### Main 200k Configuration

All 200k attention-clean runs used CIFAR-10 with `img_size=32`, `patch_size=4`, `tokens=65`, `depth=6`, `embed_dim=192`, `num_heads=6`, `mlp_ratio=4`, `batch_size=512`, label smoothing `0.1`, and hard Top-K active throughout training/evaluation. The strongest run was:

```bash
train_logic_vit_tiny.py \
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

`K8 aug const` means `attention_k=8`, default augmentation enabled, and constant LR (`5e-4`). The augmentation recipe is random horizontal flip, random crop with padding 4, mild color jitter, random grayscale, and random autocontrast.

### 200k Result Summary

| run | best valid eval/infer | final test eval/infer | interpretation |
| --- | ---: | ---: | --- |
| K8 aug const | 0.7913 @150k | 0.7893 | strongest; large gain over prior 20k baseline |
| K8 noaug const | 0.7208 @30k | 0.7154 | no meaningful gain over prior 20k test |
| K4 noaug const | 0.7051 @110k | 0.6893 | weaker K setting |
| K8 noaug warm-stay-cosine | 0.7219 @50k | 0.7193 | small gain only |

The prior 20k baseline was `best_valid=0.7172`, `test_eval=0.7158`. The strongest 200k attention-clean test result is `0.7893`, a `+0.0735` absolute improvement over the prior 20k test. However, the `K8 aug const` curve peaks at `150k=0.7913`; after that it fluctuates below the peak (`155k=0.7838`, `160k=0.7898`, `175k=0.7906`, `195k=0.7771`). The correct conclusion is: longer training is highly useful up to roughly 150k, but the 150k-200k segment did not create a new best.

## Method 2: SCTM Sparse CLS Patch Mixer

SCTM was designed after probing showed the bottleneck was the token mixer/attention path rather than simply LogicFFN width. Expanding LogicFFN ratio to 2/4 hurt stability, and immediately hardening V to sign/majority was too lossy. SCTM therefore preserves hard sparse routing while keeping a less destructive low-bit value path.

Block structure remains:

```text
x = x + SCTM(LN(x))
x = x + LogicFFN(LN(x))
```

SCTM has two paths:

- CLS global sparse read: CLS queries patch tokens, applies hard Top-K, and aggregates selected values.
- Patch local path: patch tokens are preserved by residual flow and optionally mixed with a local 3x3 patch mixer.

The promoted stable variant uses continuous score routing, hard Top-K, low-bit weighted selected V aggregation, `K=8`, and local3x3 patch mixing. Auxiliary accumulator variants add a signed saturating state as an auxiliary CLS delta while the next layer still consumes the normal low-bit token stream.

## SCTM And Auxiliary Result Highlights

Selected parsed results are in `docs/tables/vit_lgn_sctm_results.csv`. Main takeaways:

- Large SCTM scale run `d16/e1024/h32/K8/local3x3/aug/16k` reached `teacher_hard_acc=teacher_soft_acc=0.7192`, with zero hard/soft gap, peak memory about `22510 MB`, and train time about `8182 s`.
- Wider/deeper alternatives at the same 16k scale were weaker: `d12/e1536/h48=0.7154`, `d20/e1024/h32=0.7086`.
- Auxiliary accumulator is useful only when it preserves the SCTM low-bit weighted V aggregation. Replacing the feature stream with a hard saturating/sign state was too lossy in short probes.
- Value discretization was more promising than sign(V): the 8000-step d6/e192/h6 no-augmentation auxiliary baseline was around `0.633`, VQ4 reached `0.6448`, VQ3 stayed near `0.6303`, and bitplane matched the low-bit aggregation numerically with zero bitplane match error in the parsed summaries.

## Checkpoints And Weights

Attention-clean 200k checkpoints were not saved. The workdir `/home/spco/sow_linear/ViT-LGN_attention_clean_20260629` has zero `.pt/.pth/.ckpt/.safetensors` files; these runs used `--no-save-checkpoints`.

SCTM and auxiliary runs saved final checkpoints under:

```text
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/<run_name>/<timestamp>/final_model.pt
```

The full inventory is in `docs/tables/vit_lgn_weight_manifest_210.csv`. Important checkpoints include:

```text
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/sctm_long16k_d16_e1024_h32_k8_local3x3_aug_20260629/20260629_034723/final_model.pt
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/aux_value_disc_long16k_vq4_d16_e1024_h32_k8_local3x3_aug_20260629/20260629_125113/final_model.pt
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/aux_value_disc_long16k_vq3_d16_e1024_h32_k8_local3x3_aug_20260629/20260629_125114/final_model.pt
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628/runs/aux_value_disc_long16k_bitplane_d16_e1024_h32_k8_local3x3_aug_20260629/20260629_125114/final_model.pt
```

The `.pt` binaries are intentionally not committed to GitHub because they are large binary artifacts. The manifest records their 210 paths and sizes.

## Interpretation

- The strongest pure hard-route attention result is currently attention-clean `K8 aug const`, not SCTM: `0.7893` final test and `0.7913` best valid.
- SCTM is still useful as a hardware-oriented design because it separates hard sparse routing from less destructive low-bit V aggregation; this directly addresses the information loss seen in all-token Boolean V majority designs.
- The main information loss in pure majority V aggregation is value amplitude/order loss: selected V features are collapsed to binary votes, so weak but consistent evidence and magnitude differences disappear before residual accumulation. SCTM/aux variants keep more information in the selected V path and only discretize progressively.
- For future fully logic-gate conversion, the best sequence is: maintain hard Top-K routing, preserve low-bit weighted/bitplane V aggregation, stabilize route entropy and patch-token norms, then progressively replace continuous projections and accumulators with packed XNOR/popcount/comparator/shift-add implementations.
