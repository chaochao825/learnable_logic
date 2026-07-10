# LightLogic Distilled K-Expansion Evidence

## Goal Checks

| goal | status | evidence | note |
| --- | --- | --- | --- |
| Goal 6 seed0 sweep plus seed1/2 confirmation is implemented | PASS | rows=56; dataset_seed_count=6; full_sweep_dataset_seeds=2; k4_fixed_confirmation_dataset_seeds=6; k_values=2,4; init_modes=fixed,per_neuron; alphas=0,0.5,2; taus=1,2 | Teacher is continuous LightLogic; student is K-expanded hard network with learnable thresholds. Only dataset/seeds counted as full_sweep have the full K/init/alpha/tau grid. |
| Distillation/tuning improves deployable hard student accuracy | PARTIAL | improved_groups=11/12; worsened_groups=1/12 | Best row per dataset/seed/K/init is compared against the same group's initial hard student. |
| Teacher-KL improves beyond CE-only threshold tuning | PARTIAL | kl_wins=10/12; kl_losses=2/12 | Alpha=0 rows are CE-only; alpha>0 rows use teacher-logit KL. |
| Goal 5 data-weighted K rounding changes q to r/K under current constraints | DEGENERATE | weighted_rounding_changed_ratio_range=[0,0] | With independent per-entry integer choices, data weights do not change nearest rounding; Goal 5 needs coupled or task-aware constraints. |
| Best distilled student beats the matching K-expanded/calibrated hard baseline | PARTIAL | exact_k_wins=7/8; dataset_best_wins=6/8; missing_exact=4 | Exact comparison matches dataset/seed/K/init; dataset-best comparison is stricter. |

## Best Per Dataset

| dataset | seed | K | init | teacher_acc | initial_hard_acc | distilled_hard_acc | delta_vs_initial | alpha | tau | hard_gap_vs_teacher | expanded_gate_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 4 | fixed | 1.0 | 0.59375 | 0.8125 | 0.21875 | 0.5 | 2.0 | 0.1875 | 2048 |
| majority9 | 1 | 4 | fixed | 1.0 | 0.515625 | 0.734375 | 0.21875 | 0.5 | 2.0 | 0.265625 | 2048 |
| majority9 | 2 | 4 | fixed | 1.0 | 0.6015625 | 0.859375 | 0.257812 | 0.0 | 2.0 | 0.140625 | 2048 |
| random_sparse10 | 0 | 4 | fixed | 1.0 | 0.859375 | 0.9921875 | 0.132812 | 0.5 | 2.0 | 0.0078125 | 2048 |
| random_sparse10 | 1 | 4 | fixed | 1.0 | 0.76171875 | 0.97265625 | 0.210938 | 0.5 | 2.0 | 0.02734375 | 2048 |
| random_sparse10 | 2 | 4 | fixed | 1.0 | 0.93359375 | 0.9921875 | 0.0585938 | 0.5 | 2.0 | 0.0078125 | 2048 |

## Best Per K/Init

| dataset | seed | K | init | initial_hard_acc | best_distilled_hard_acc | delta | best_alpha | best_tau | CE_best | KL_best | KL_delta_vs_CE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 2 | fixed | 0.6328125 | 0.78125 | 0.148438 | 0.5 | 2.0 | 0.7421875 | 0.78125 | 0.0390625 |
| majority9 | 0 | 2 | per_neuron | 0.734375 | 0.71875 | -0.015625 | 0.5 | 2.0 | 0.6875 | 0.71875 | 0.03125 |
| majority9 | 0 | 4 | fixed | 0.59375 | 0.8125 | 0.21875 | 0.5 | 2.0 | 0.796875 | 0.8125 | 0.015625 |
| majority9 | 0 | 4 | per_neuron | 0.5859375 | 0.6875 | 0.101562 | 2.0 | 2.0 | 0.640625 | 0.6875 | 0.046875 |
| majority9 | 1 | 4 | fixed | 0.515625 | 0.734375 | 0.21875 | 0.5 | 2.0 | 0.7265625 | 0.734375 | 0.0078125 |
| majority9 | 2 | 4 | fixed | 0.6015625 | 0.859375 | 0.257812 | 0.0 | 2.0 | 0.859375 | 0.84375 | -0.015625 |
| random_sparse10 | 0 | 2 | fixed | 0.81640625 | 0.94140625 | 0.125 | 0.0 | 1.0 | 0.94140625 | 0.91015625 | -0.03125 |
| random_sparse10 | 0 | 2 | per_neuron | 0.85546875 | 0.9609375 | 0.105469 | 0.5 | 1.0 | 0.953125 | 0.9609375 | 0.0078125 |
| random_sparse10 | 0 | 4 | fixed | 0.859375 | 0.9921875 | 0.132812 | 0.5 | 2.0 | 0.96484375 | 0.9921875 | 0.0273438 |
| random_sparse10 | 0 | 4 | per_neuron | 0.8203125 | 0.90625 | 0.0859375 | 0.5 | 1.0 | 0.8828125 | 0.90625 | 0.0234375 |
| random_sparse10 | 1 | 4 | fixed | 0.76171875 | 0.97265625 | 0.210938 | 0.5 | 2.0 | 0.921875 | 0.97265625 | 0.0507812 |
| random_sparse10 | 2 | 4 | fixed | 0.93359375 | 0.9921875 | 0.0585938 | 0.5 | 2.0 | 0.98828125 | 0.9921875 | 0.00390625 |

## K-Baseline Comparison

| dataset | seed | K | init | distilled_hard_acc | K_exact_acc | delta_exact | K_dataset_best | delta_dataset_best |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 2 | fixed | 0.78125 | 0.6328125 | 0.148438 | 2/per_neuron:0.734375 | 0.046875 |
| majority9 | 0 | 2 | per_neuron | 0.71875 | 0.734375 | -0.015625 | 2/per_neuron:0.734375 | -0.015625 |
| majority9 | 0 | 4 | fixed | 0.8125 | 0.59375 | 0.21875 | 2/per_neuron:0.734375 | 0.078125 |
| majority9 | 0 | 4 | per_neuron | 0.6875 | 0.5859375 | 0.101562 | 2/per_neuron:0.734375 | -0.046875 |
| majority9 | 1 | 4 | fixed | 0.734375 |  |  |  |  |
| majority9 | 2 | 4 | fixed | 0.859375 |  |  |  |  |
| random_sparse10 | 0 | 2 | fixed | 0.94140625 | 0.81640625 | 0.125 | 4/fixed:0.859375 | 0.0820312 |
| random_sparse10 | 0 | 2 | per_neuron | 0.9609375 | 0.85546875 | 0.105469 | 4/fixed:0.859375 | 0.101562 |
| random_sparse10 | 0 | 4 | fixed | 0.9921875 | 0.859375 | 0.132812 | 4/fixed:0.859375 | 0.132812 |
| random_sparse10 | 0 | 4 | per_neuron | 0.90625 | 0.8203125 | 0.0859375 | 4/fixed:0.859375 | 0.046875 |
| random_sparse10 | 1 | 4 | fixed | 0.97265625 |  |  |  |  |
| random_sparse10 | 2 | 4 | fixed | 0.9921875 |  |  |  |  |

## LightLogic/OP Baseline Comparison

Baseline coverage: dlgn_op_hard_acc=2/6; light_iwp_anneal_hard_acc=2/6; light_iwp_gumbel_st_hard_acc=2/6; light_iwp_hard_acc=2/6; light_iwp_st_hard_acc=2/6

| dataset | seed | distilled_hard_acc | dlgn_op_hard_acc | light_iwp_anneal_hard_acc | light_iwp_gumbel_st_hard_acc | light_iwp_hard_acc | light_iwp_st_hard_acc | delta_vs_dlgn_op | delta_vs_light_iwp | delta_vs_light_iwp_anneal | delta_vs_light_iwp_gumbel_st | delta_vs_light_iwp_st |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 0.8125 | 0.890625 | 0.539062 | 0.507812 | 0.515625 | 1 | -0.078125 | 0.296875 | 0.273438 | 0.304688 | -0.1875 |
| majority9 | 1 | 0.734375 |  |  |  |  |  |  |  |  |  |  |
| majority9 | 2 | 0.859375 |  |  |  |  |  |  |  |  |  |  |
| random_sparse10 | 0 | 0.9921875 | 0.96875 | 0.484375 | 0.582031 | 0.820312 | 0.945312 | 0.0234375 | 0.171875 | 0.507812 | 0.410156 | 0.046875 |
| random_sparse10 | 1 | 0.97265625 |  |  |  |  |  |  |  |  |  |  |
| random_sparse10 | 2 | 0.9921875 |  |  |  |  |  |  |  |  |  |  |

Interpretation guardrails:
- `alpha=0` is CE-only student threshold tuning; only `alpha>0` uses teacher-logit KL.
- The student truth tables are fixed after K expansion; the current distillation prototype only learns thresholds.
- `weighted_rounding_changed_ratio=0` means Goal 5 is not an effective intervention under the current independent rounding formulation.
- The full K/init/alpha/tau sweep is seed-0 unless more full-sweep runs are supplied; seed 1/2 rows, when present, are scoped K=4/fixed confirmation runs.
