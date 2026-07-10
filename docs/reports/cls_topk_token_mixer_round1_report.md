# CLS Top-K Token Mixer Round 1

Date: 2026-06-28

## Scope

- Added an independent ablation script: `cls_topk_token_mixer_ablation.py`.
- Added launch helpers:
  - `run_cls_topk_token_mixer_ablation_20260628.sh`
  - `launch_cls_topk_token_mixer_ablation_20260628.sh`
  - `run_cls_topk_token_mixer_b123_20260628.sh`
  - `launch_cls_topk_token_mixer_b123_20260628.sh`
- No changes were made to `LogicFFN`, light logic gates, `GroupSum`, training loss, or the classifier.
- Mixers are patch-only CLS mixers: selected patch values update CLS; patch-token residual outputs are zero.

## Artifacts

- b3 and b2+b3 run:
  - `runs/cls_topk_token_mixer_round1_topkmasked/20260628_002845_783062_pid448746/results.csv`
  - `runs/cls_topk_token_mixer_round1_topkmasked/20260628_002845_783062_pid448746/route_diagnostics.csv`
  - `logs/20260628_cls_topk_token_mixer_ablation/run_20260628_002840.log`
  - Note: this log was produced before the run script was updated to append `_pid$$` to future log filenames.
- b1+b2+b3 conditional run:
  - `runs/cls_topk_token_mixer_b123_topkmasked/20260628_011343_059184_pid851020/results.csv`
  - `runs/cls_topk_token_mixer_b123_topkmasked/20260628_011343_059184_pid851020/route_diagnostics.csv`
  - `logs/20260628_cls_topk_token_mixer_b123/run_20260628_011337_pid851011.log`

Both runs used the same full-attention baseline config:
`runs/attn_best_depth_expand_pilot_l4_fixed/20260627_232055/config.json`

Baseline full attention:

| soft acc | hard acc | eval split | teacher iters |
|---:|---:|---|---:|
| 0.4562 | 0.4562 | test | 1000 |

## Best Results

| block setting | K | mixer | hard acc | hard drop vs full | CLS cosine | attention top-k overlap | route entropy |
|---|---:|---|---:|---:|---:|---:|---:|
| b3 | 4 | dynamic_weighted | 0.4743 | -0.0181 | 0.598 | 0.089 | 0.319 |
| b3 | 4 | static_mean | 0.4724 | -0.0162 | 0.593 | 0.092 | 0.333 |
| b2,b3 | 8 | static_weighted | 0.4709 | -0.0147 | 0.579 | 0.185 | 0.499 |
| b2,b3 | 8 | dynamic_weighted | 0.4648 | -0.0086 | 0.568 | 0.317 | 0.464 |
| b1,b2,b3 | 4 | dynamic_weighted | 0.4825 | -0.0263 | 0.580 | 0.147 | 0.316 |
| b1,b2,b3 | 8 | dynamic_weighted | 0.4731 | -0.0169 | 0.573 | 0.194 | 0.466 |
| b1,b2,b3 | 16 | dynamic_weighted | 0.4712 | -0.0150 | 0.575 | 0.367 | 0.588 |

Negative drop means the mixer run exceeded the full-attention baseline in this 1000-iteration ablation.

## Observations

- Replacing b3 only is stable for several variants. Best b3 result is K=4 dynamic weighted.
- Replacing b2+b3 is also stable. Best b2+b3 result is K=8 static weighted.
- The conditional b1+b2+b3 run has a strong dynamic weighted path, but mean/static variants are much less stable. This matches the prior probe that earlier attention blocks are more important.
- Mean top-k is competitive in some cases, especially b3/static K=4, but dynamic weighted is consistently stronger for b1+b2+b3 and often stronger for dynamic routing.
- Dynamic routers select from almost all patch positions over the eval sample (`unique_selected_patch_ratio` near 1.0), while static routers select exactly K fixed positions per head (`K/64` unique ratio).
- Overlap with original attention top-k increases with K but remains modest; the mixer can work without copying the original attention top-k exactly.
- Corrected top-k-masked runs have `soft_route_topk_active=true` for mixer rows and `hard_extra_drop_vs_soft=0.0`; hard top-k routing does not add extra loss beyond the selected top-k training/eval path in this ablation.

## Review

Mandatory review found and fixed one high-severity issue before final runs: the first draft trained the soft path with dense patch softmax and only used top-k in hard eval. The corrected script applies top-k masking in training and soft eval as well.

Final review status: PASS WITH LOW RISK. The only remaining low-risk note was log filename uniqueness, fixed by adding the process id to timestamped log names.
