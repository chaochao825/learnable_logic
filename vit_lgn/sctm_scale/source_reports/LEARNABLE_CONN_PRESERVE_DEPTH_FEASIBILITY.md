# Learnable Connection Preserve-Depth Feasibility

This note extends the first-round learnable-connection ablation after changing the depth success criterion from "higher than fixed baseline" to "preserve depth / avoid depth collapse".

## Feasibility Summary

The revised criterion is feasible and better matched to the current candidate topology.

The current learnable connection pool is backward-looking: sources come from the previous logic layer, earlier hidden layers, or input. This topology cannot create a path deeper than the number of LogicFFN layers. Therefore, `conn_effective_depth > fixed_baseline` is not a meaningful success criterion when fixed random connectivity already uses the full layer chain.

Recommended depth metrics:

- `depth_preservation = conn_avg_output_depth / logic_ffn_layers`
- `depth_collapse = 1 - depth_preservation`
- `p95_depth_preservation = conn_output_depth_p95 / logic_ffn_layers`
- Keep `conn_effective_depth` only as a max-depth sanity check, not the primary criterion.

## Depth-4 Baseline Evidence

Original first-round runs use `logic_ffn_layers=4`.

| method | soft_acc | hard_acc | gap | avg_out_depth | preservation | fanout_max |
|---|---:|---:|---:|---:|---:|---:|
| fixed | 0.3048 | 0.3106 | 0.0058 | 4.00 | 1.000 | 2.0 |
| learnable_nobias | 0.3240 | 0.3052 | 0.0188 | 2.99 | 0.748 | 31.5 |
| learnable_bias | 0.3240 | 0.3222 | 0.0018 | 4.00 | 1.000 | 52.0 |
| learnable_bias_postft | 0.3248 | 0.3222 | 0.0026 | 4.00 | 1.000 | 52.0 |

Result directories:

- `runs/learnable_conn_round1_globalrng_dense_fixed/20260627_191238`
- `runs/learnable_conn_round1_globalrng_dense_nobias/20260627_191310`
- `runs/learnable_conn_round1_globalrng_dense_bias/20260627_191729`
- `runs/learnable_conn_round1_globalrng_dense_bias_postft/20260627_191833`

## Depth-8 Probe

The same setup was rerun with `logic_ffn_layers=8`, seed 0, CIFAR-10 valid split, full validation evaluation, `embed_dim=48`, ViT depth 2, thresholds 5, teacher iterations 400, and `K=64`.

| method | soft_acc | hard_acc | gap | avg_out_depth | preservation | avg_skip | input | l1 | l2 | older | train_time_s | peak_mem_mb | fanout_max | fanout_p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed | 0.3326 | 0.3282 | 0.0044 | 8.00 | 1.000 | 1.00 | 0.125 | 0.875 | 0.000 | 0.000 | 18.30 | 311.2 | 2.0 | 2.0 |
| learnable_nobias | 0.3356 | 0.3024 | 0.0332 | 3.98 | 0.497 | 2.88 | 0.635 | 0.249 | 0.088 | 0.028 | 56.23 | 651.3 | 80.5 | 18.0 |
| learnable_bias | 0.3472 | 0.3408 | 0.0064 | 8.00 | 1.000 | 1.00 | 0.125 | 0.875 | 0.000 | 0.000 | 59.31 | 651.3 | 75.0 | 18.3 |
| learnable_bias_postft | 0.3430 | 0.3408 | 0.0022 | 8.00 | 1.000 | 1.00 | 0.125 | 0.875 | 0.000 | 0.000 | 59.07 + 7.39 | 651.3 | 75.0 | 18.3 |

Result directories:

- `runs/learnable_conn_deeper8_preserve_fixed/20260627_213306`
- `runs/learnable_conn_deeper8_preserve_nobias/20260627_213400`
- `runs/learnable_conn_deeper8_preserve_bias/20260627_213551`
- `runs/learnable_conn_deeper8_preserve_bias_postft/20260627_213744`

## Interpretation

The depth-8 probe supports the revised success criterion:

- no-bias learnable routing collapses depth: preservation falls to `0.497`, and the soft-to-hard gap increases to `0.0332`.
- skip-biased routing preserves full depth: preservation stays at `1.000`.
- skip-biased routing improves hard accuracy over fixed: `0.3408` vs `0.3282`.
- post-discretization gate-only finetune reduces the gap from `0.0064` to `0.0022`, but does not improve hard accuracy.

The tradeoff remains fanout concentration. Both learnable variants have much higher fanout than fixed connectivity.

## Design Recommendation

Use the following hierarchy for the next iteration:

1. Keep the current minimal-intrusion learnable layer and change the main depth success metric to depth preservation.
2. Report both depth-4 and depth-8 probes; depth-8 is the clearer stress test because no-bias collapse becomes obvious.
3. Add fanout control only as a second-stage ablation. Do not add it to the first clean comparison, because it changes the mechanism being tested.
4. Treat cross-block learnable connections as a separate method. They require carrying source histories across Transformer blocks and are no longer a drop-in replacement for a single LogicLayer.

## Cross-Block Connectivity Feasibility

Cross-block connections are feasible but not minimal-intrusion:

- `LogicViTTiny.forward()` would need to retain logic outputs from earlier Transformer blocks.
- Candidate metadata would need `(block_id, logic_layer_id, node_id)` instead of just local layer ids.
- Effective-depth accounting must span blocks and token features.
- Memory grows because previous block histories must remain available for routing.
- The method would no longer isolate "replace fixed random LogicLayer connectivity"; it would test a different topology class.

Recommended if pursued: implement as `CrossBlockLearnableConnLightLogicLayer` behind a separate flag and compare only after the current local-layer result is fully reported.

## Final Feasibility Call

Changing the success criterion to "preserve depth / avoid depth collapse" is reasonable and empirically supported. Allowing deeper LogicFFN is also feasible and makes the distinction stronger. Cross-block connectivity is possible but should be treated as a new experiment, not as a patch to the first minimal-intrusion implementation.
