# Learnable Connection Round 1 Report

This report summarizes the strict-scope learnable-connection ablation for the current ViT_LGN / LightLGN branch.

## Scope

The current implementation keeps the Light Logic gate implementation and classifier surface out of scope:

- No hard-ST gate training path is present in the active files.
- No optional signed/nonnegative classifier heads are present in the active files.
- The ViT classifier remains the original `nn.Linear` head.
- Changes are limited to candidate-pool connectivity, connection scheduling, optimizer parameter groups, post-discretization gate-only finetune plumbing, and metrics.

Checks run on the current remote worktree:

- `python -m py_compile logic_iwp.py logic_vit_tiny.py train_logic_vit_tiny.py goal6plus_vit_experiment.py`
- grep for `set_train_hard_st`, `_train_hard_st`, `head_type`, `signed_head`, `NonNegative`, and `SignedSparse` returns no matches in the active files.
- Fixed and learnable smoke runs completed.

## Final Runs

All runs use CIFAR-10 valid split, seed 0, full validation evaluation, `embed_dim=48`, ViT depth 2, `logic_ffn_layers=4`, thermometer thresholds 5, teacher iterations 400, and `K=64`.

| method | soft_acc | discrete_acc | acc_gap | train_time_s | peak_mem_mb | conn_eff_depth | avg_skip | input | l1 | l2 | older | dead_gate | fanout_max | fanout_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.3048 | 0.3106 | 0.0058 | 18.07 | 230.9 | 4.0 | 1.00 | 0.250 | 0.750 | 0.000 | 0.000 | 0.000 | 2.0 | 2.0 |
| learnable_nobias | 0.3240 | 0.3052 | 0.0188 | 25.35 | 377.8 | 4.0 | 1.69 | 0.598 | 0.284 | 0.098 | 0.021 | 0.000 | 31.5 | 10.9 |
| learnable_bias | 0.3240 | 0.3222 | 0.0018 | 25.31 | 377.8 | 4.0 | 1.00 | 0.250 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 | 52.0 | 10.07 |
| learnable_bias_postft | 0.3248 | 0.3222 | 0.0026 | 25.31 + 3.61 | 377.8 | 4.0 | 1.00 | 0.250 | 0.750 | 0.000 | 0.000 | 0.000 | 52.0 | 10.07 |

Result directories:

- `runs/learnable_conn_round1_globalrng_dense_fixed/20260627_191238`
- `runs/learnable_conn_round1_globalrng_dense_nobias/20260627_191310`
- `runs/learnable_conn_round1_globalrng_dense_bias/20260627_191729`
- `runs/learnable_conn_round1_globalrng_dense_bias_postft/20260627_191833`

## Interpretation

The skip-biased learnable connection variant is the only positive setting in this round. It improves discrete accuracy over fixed random connectivity (`0.3222` vs `0.3106`) and reduces the soft-to-discrete gap (`0.0018` vs `0.0058`) with moderate overhead (`25.31s`, `377.8MB` vs `18.07s`, `230.9MB`).

The no-bias variant is stable but worse for discretization. It selects many input or early-layer sources, has a larger average skip distance, concentrates fanout, and increases the gap to `0.0188`.

Post-discretization gate-only finetuning does not improve hard accuracy in this setup. It raises relaxed accuracy slightly but increases the post-finetune gap.

Effective depth does not exceed the fixed baseline. With `logic_ffn_layers=4` and fixed connectivity already using layer-by-layer paths, the current connection-depth metric has a hard ceiling of `4.0` inside each LogicFFN. Skip candidates can preserve or shorten this path, but cannot make it deeper than the number of logic layers. The useful finding is therefore that skip bias preserves depth while no-bias routing tends to collapse toward input/early sources.

The main remaining negative result is source utilization. Learnable bias improves discrete behavior, but it concentrates routing: fanout max is `52`, compared with `2` for fixed connectivity.

## Status Against Criteria

- Stable training: pass.
- Memory/time close to baseline: pass with moderate overhead.
- Discrete accuracy drop not worse than baseline: pass for learnable bias, fail for no-bias.
- Effective depth higher than baseline: not achieved; under the current backward-looking candidate topology, this criterion is structurally unreachable when fixed connectivity already reaches all `logic_ffn_layers`.
- Dead/constant gate ratio not increased: pass.
- Fixed-connection gate-only finetune helps: not supported.

## Review Status

The required subagent review was attempted, but the reviewer timed out twice and was closed while still running. A local fallback review was completed: scope grep, compile, result-path audit, and metric-definition audit. The fallback did not find a code-scope violation, but it did identify the effective-depth criterion as unmet under the current topology.
