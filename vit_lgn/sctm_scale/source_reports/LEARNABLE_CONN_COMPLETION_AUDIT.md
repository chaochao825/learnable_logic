# Learnable Connection Completion Audit

This audit maps the active objective to the current implementation and first-round evidence. It is intentionally stricter than the result report: an item is marked complete only when current evidence proves it.

## Current Evidence

Code state:

- `logic_iwp.py`, `logic_vit_tiny.py`, `train_logic_vit_tiny.py`, and `goal6plus_vit_experiment.py` compile.
- Grep over those files for `set_train_hard_st`, `_train_hard_st`, `head_type`, `signed_head`, `NonNegative`, and `SignedSparse` returns no matches.
- The active ViT classifier is `nn.Linear`; no classifier or GroupSum path was changed for this goal.
- The active Light Logic gate training forward is unchanged in behavior: training uses soft gate weights, hard gate weights are only used through existing eval/harden paths.

Final result directories:

- `runs/learnable_conn_round1_globalrng_dense_fixed/20260627_191238`
- `runs/learnable_conn_round1_globalrng_dense_nobias/20260627_191310`
- `runs/learnable_conn_round1_globalrng_dense_bias/20260627_191729`
- `runs/learnable_conn_round1_globalrng_dense_bias_postft/20260627_191833`

Review gate:

- A subagent review was spawned, but it timed out twice and was closed while still running.
- Local fallback review completed: full scope grep, compile, result-path audit, and metric-definition audit.

## Requirement Audit

| Requirement | Evidence | Status |
|---|---|---|
| Add `LearnableConnLightLogicLayer` as a drop-in alternative to fixed `LogicLayer` connectivity | Implemented in `logic_iwp.py`; `LogicFFN_IWP(connectivity=\"fixed\"|\"learnable\")`; fixed and learnable smoke runs passed. | Proven |
| Candidate pool per gate input port, default `K=64` | Final run configs use `--learnable-conn-k 64`; code maintains `candidate_layers0/1`, `candidate_nodes0/1`. | Proven |
| Candidate mix mostly previous layer with limited skip/input candidates | Candidate metadata exists and final metrics record `conn_pct_l1/l2/older/input`. Availability fallbacks reassign unavailable early-layer buckets to previous layer. | Proven |
| Per-port trainable logits `[out_features, K]` | `conn_logits0` and `conn_logits1` are parameters shaped by `out_dim` and `candidate_k`; optimizer separates `conn_logits`. | Proven |
| Training soft routing uses `softmax((conn_logits - beta_skip * distance_penalty) / tau_conn)` | `_scores()` applies distance penalty; `_soft_port()` applies softmax and weighted source-layer routing. | Proven |
| Gate input is existing Light Logic gate `u0,u1` path | Learnable layer wraps `LogicLayerIWP` and writes `u0/u1` into strict pair layout; no gate forward/param/init/discretization changes remain. | Proven |
| Schedule: first 5% frozen, 5-80% `tau 2.0 -> 0.3`, 80-100% `tau 0.3 -> 0.1` | `get_connection_schedule()` implements the schedule. | Proven |
| `conn_lr = 0.2 * gate_lr` | Optimizer has separate `conn_logits` group using `logic_conn_lr_multiplier`; final runs use default `0.2`. | Proven |
| No-bias uses `beta_skip = 0`; bias uses fixed `0.5` | Schedule reads `learnable_conn_use_skip_bias`; final no-bias/bias configs differ by that flag. | Proven |
| Discretization uses argmax of `conn_logits - beta_skip * distance_penalty` | `selected_sources()` uses `_scores(...).argmax`; `discretize_connections()` stores selected source metadata and freezes connection logits. | Proven |
| Post-discretization fixed-connection gate-only finetune option | Implemented and run in `bias_postft`; `post_disc_ft_trainable_params=7680`, matching gate-weight-only finetune. | Proven |
| Record required metrics | Final summaries include soft/hard accuracy, gap/drop, selected skip distance, source percentages, entropy, dead gate ratio, effective depth, fanout max/p95, training time, and peak memory. | Proven |

## Success Criteria Audit

| Criterion | Final evidence | Status |
|---|---|---|
| Training stable | All four final runs completed. | Met |
| No obvious memory/time explosion | Fixed: `18.07s`, `230.9MB`; bias: `25.31s`, `377.8MB`. | Met |
| Discrete accuracy drop not worse than baseline | Fixed gap `0.0058`; bias gap `0.0018`; no-bias gap `0.0188`. | Met for bias, contradicted for no-bias |
| Effective depth higher than baseline | Fixed connection effective depth is `4.0`; bias is also `4.0`. | Not met |
| Dead gate ratio does not increase significantly | Dead ratio remains `0.0`. | Met |
| Soft accuracy not required if discrete behavior improves | Bias hard accuracy improves `0.3106 -> 0.3222`; gap shrinks. | Met for bias |
| Post-discretization finetune improves hard model | Bias hard stays `0.3222`; post-ft only raises soft acc from `0.3240` to `0.3248`. | Contradicted |

## Effective-Depth Finding

The objective asks whether learnable connectivity can increase effective depth. In the current architecture, fixed connectivity already reaches the maximum possible depth inside each LogicFFN:

- `logic_ffn_layers = 4`
- fixed `conn_effective_depth = 4.0`
- fixed source pattern is `input` for the first logic layer and `l-1` for later logic layers

The learnable candidate pool is backward-looking: candidates come from the previous layer, earlier hidden layers, or input. Such candidates can preserve depth or skip layers, but cannot create a path deeper than the number of logic layers. Therefore, the strict success criterion "effective depth higher than baseline" is not achievable under this exact topology and metric when the fixed baseline already reaches all layers.

What the experiment does prove:

- no-bias routing uses longer skips and input sources, increasing average skip distance to `1.69`, worsening gap to `0.0188`, and concentrating fanout.
- skip-biased routing preserves the fixed depth pattern and improves hard accuracy/gap.

The valid claim is that skip bias prevents depth collapse and improves discretization behavior, not that it increases effective depth.

## Final Status

The implementation and first-round comparison are sufficient to support the strict connectivity ablation, but the full objective should not be marked complete because one explicit success criterion is unmet:

- effective depth is not higher than baseline.

The strongest supported claim is:

> Under the strict seed-0 CIFAR-10 valid ablation, skip-biased learnable candidate-pool connectivity improves hard accuracy (`0.3106 -> 0.3222`) and reduces the soft-to-hard gap (`0.0058 -> 0.0018`) with moderate overhead, while preserving but not increasing effective depth. It also worsens fanout concentration relative to the fixed baseline.
