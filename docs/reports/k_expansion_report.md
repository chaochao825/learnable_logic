# LightLogic K-Expansion Evidence

## Goal Checks

| goal | status | evidence | note |
| --- | --- | --- | --- |
| Goal 3 K-gate truth-table expansion is implemented and swept | PASS | rows=45; k_values=2,4,8,16,32; calibration_modes=fixed,per_layer,per_neuron | Local q vs r/K error is reported separately from full-network thresholded accuracy. |
| K expansion reduces local truth-table approximation error as K grows | PASS | checked_groups=9; all local_mae trends are non-increasing with K; local_mae_range=[0.00379145,0.0529127] | Lower local error does not imply full-network accuracy improvement after thresholding. |
| Goal 4 layer-wise/per-neuron calibration improves fixed-threshold expansion | PARTIAL | wins=4/30 | Calibration is useful only where delta_acc is positive under the same dataset/seed/K. |

## Best Expanded Rows

| dataset | seed | best_k | best_calibration_mode | teacher_acc | expanded_acc | acc_gap | local_mae | expanded_gate_count | source_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 2 | per_neuron | 1.0 | 0.734375 | 0.265625 | 0.052912650629878044 | 1024 | lightlogic_k_expansion_bool_seed0_per_neuron_v1 |
| parity8 | 0 | 2 | fixed | 0.21875 | 0.5 | 0.28125 | 0.03316331887617707 | 1024 | lightlogic_k_expansion_bool_seed0_v1 |
| random_sparse10 | 0 | 4 | fixed | 1.0 | 0.859375 | 0.140625 | 0.02289935853332281 | 2048 | lightlogic_k_expansion_bool_seed0_v1 |

## Calibration Deltas

| dataset | seed | K | mode | fixed_acc | calibrated_acc | delta_acc | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| majority9 | 0 | 2 | per_layer | 0.6328125 | 0.6015625 | -0.03125 | LOSS |
| majority9 | 0 | 2 | per_neuron | 0.6328125 | 0.734375 | 0.101562 | WIN |
| majority9 | 0 | 4 | per_layer | 0.59375 | 0.625 | 0.03125 | WIN |
| majority9 | 0 | 4 | per_neuron | 0.59375 | 0.5859375 | -0.0078125 | LOSS |
| majority9 | 0 | 8 | per_layer | 0.5703125 | 0.515625 | -0.0546875 | LOSS |
| majority9 | 0 | 8 | per_neuron | 0.5703125 | 0.515625 | -0.0546875 | LOSS |
| majority9 | 0 | 16 | per_layer | 0.515625 | 0.515625 | 0 | TIE |
| majority9 | 0 | 16 | per_neuron | 0.515625 | 0.515625 | 0 | TIE |
| majority9 | 0 | 32 | per_layer | 0.5078125 | 0.5078125 | 0 | TIE |
| majority9 | 0 | 32 | per_neuron | 0.5078125 | 0.5078125 | 0 | TIE |
| parity8 | 0 | 2 | per_layer | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 2 | per_neuron | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 4 | per_layer | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 4 | per_neuron | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 8 | per_layer | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 8 | per_neuron | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 16 | per_layer | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 16 | per_neuron | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 32 | per_layer | 0.5 | 0.5 | 0 | TIE |
| parity8 | 0 | 32 | per_neuron | 0.5 | 0.5 | 0 | TIE |
| random_sparse10 | 0 | 2 | per_layer | 0.81640625 | 0.83203125 | 0.015625 | WIN |
| random_sparse10 | 0 | 2 | per_neuron | 0.81640625 | 0.85546875 | 0.0390625 | WIN |
| random_sparse10 | 0 | 4 | per_layer | 0.859375 | 0.80859375 | -0.0507812 | LOSS |
| random_sparse10 | 0 | 4 | per_neuron | 0.859375 | 0.8203125 | -0.0390625 | LOSS |
| random_sparse10 | 0 | 8 | per_layer | 0.8359375 | 0.828125 | -0.0078125 | LOSS |
| random_sparse10 | 0 | 8 | per_neuron | 0.8359375 | 0.8125 | -0.0234375 | LOSS |
| random_sparse10 | 0 | 16 | per_layer | 0.80078125 | 0.80078125 | 0 | TIE |
| random_sparse10 | 0 | 16 | per_neuron | 0.80078125 | 0.79296875 | -0.0078125 | LOSS |
| random_sparse10 | 0 | 32 | per_layer | 0.80078125 | 0.7890625 | -0.0117188 | LOSS |
| random_sparse10 | 0 | 32 | per_neuron | 0.80078125 | 0.79296875 | -0.0078125 | LOSS |

Interpretation guardrails:
- The current K-expansion student still thresholds back to one bit after each layer.
- Accuracy can fall even when local truth-table error improves, so use `expanded_acc` and `acc_gap` for claims.
