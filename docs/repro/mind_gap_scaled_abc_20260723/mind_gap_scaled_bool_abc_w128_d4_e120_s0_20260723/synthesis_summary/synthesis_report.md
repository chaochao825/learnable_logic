# Hard-LGN ABC Synthesis Report

Run directory: `runs/mind_gap_scaled_bool_abc_w128_d4_e120_s0_20260723`

ABC is used here as a structural baseline only. The optimized networks are not imported back into PyTorch, so post-ABC accuracy and gap are not re-evaluated.

Rows: 21 total, 21 with `abc_status=ok`.

## Method/Dataset Summary
| dataset | method | discrete_acc | acc_gap | unused_gate_ratio | pre_gate_count | pre_depth | pre_fanout_max | abc_post_and | abc_post_lev | and_reduction | level_delta |
|---|---|---|---|---|---|---|---|---|---|---|---|
| majority9 | dlgn | 0.6953 | 0.3047 | 0.9434 | 512 | 4 | 29 | 172 | 6 | 66.4% | 2 |
| majority9 | dlgn_anneal | 0.8203 | 0.1562 | 0.9473 | 512 | 4 | 29 | 168 | 5 | 67.2% | 1 |
| majority9 | gumbel_st | 0.4844 | 0.02344 | 0.9902 | 512 | 4 | 29 | 199 | 7 | 61.1% | 3 |
| majority9 | block_relaxed | 0.5 | 0.1406 | 0.9863 | 512 | 4 | 29 | 185 | 7 | 63.9% | 3 |
| majority9 | block_hard_refit | 0.8281 | 0.2891 | 0.9844 | 512 | 4 | 29 | 157 | 7 | 69.3% | 3 |
| majority9 | block_hard_task_refit | 0.7188 | 0.1797 | 0.9844 | 512 | 4 | 29 | 79 | 5 | 84.6% | 1 |
| majority9 | hard_st_cage | 0.8125 | 0.2734 | 0.9688 | 512 | 4 | 29 | 117 | 6 | 77.1% | 2 |
| parity8 | dlgn | 0.4531 | 0.1406 | 0.9727 | 512 | 4 | 32 | 176 | 8 | 65.6% | 4 |
| parity8 | dlgn_anneal | 0.3906 | 0.1094 | 0.9824 | 512 | 4 | 32 | 184 | 6 | 64.1% | 2 |
| parity8 | gumbel_st | 0.5938 | 0.1875 | 0.9922 | 512 | 4 | 32 | 174 | 6 | 66.0% | 2 |
| parity8 | block_relaxed | 0.4844 | 0.1094 | 0.9805 | 512 | 4 | 32 | 172 | 6 | 66.4% | 2 |
| parity8 | block_hard_refit | 0.4844 | 0.1094 | 0.9824 | 512 | 4 | 32 | 128 | 5 | 75.0% | 1 |
| parity8 | block_hard_task_refit | 0.3438 | 0.0625 | 0.9863 | 512 | 4 | 32 | 56 | 4 | 89.1% | 0 |
| parity8 | hard_st_cage | 0.3281 | 0.07812 | 0.9922 | 512 | 4 | 32 | 116 | 6 | 77.3% | 2 |
| random_sparse10 | dlgn | 0.5859 | 0.4141 | 0.8906 | 512 | 4 | 26 | 238 | 7 | 53.5% | 3 |
| random_sparse10 | dlgn_anneal | 0.8008 | 0.1992 | 0.873 | 512 | 4 | 26 | 236 | 7 | 53.9% | 3 |
| random_sparse10 | gumbel_st | 0.6133 | 0.09766 | 0.9941 | 512 | 4 | 26 | 193 | 6 | 62.3% | 2 |
| random_sparse10 | block_relaxed | 0.4766 | 0.05078 | 0.9844 | 512 | 4 | 26 | 194 | 6 | 62.1% | 2 |
| random_sparse10 | block_hard_refit | 0.4844 | 0 | 0.9922 | 512 | 4 | 26 | 128 | 7 | 75.0% | 3 |
| random_sparse10 | block_hard_task_refit | 0.4219 | 0.0625 | 0.9941 | 512 | 4 | 26 | 55 | 5 | 89.3% | 1 |
| random_sparse10 | hard_st_cage | 0.918 | 0.4023 | 0.8828 | 512 | 4 | 26 | 142 | 7 | 72.3% | 3 |

## Notes

- `abc_post_and` is the post-`strash; dc2` ABC AND-node count, not a PyTorch gate-count replacement.
- `and_reduction` is `(abc_pre_nd - abc_post_and) / abc_pre_nd`.
- `level_delta` is `abc_post_lev - abc_pre_lev`; positive values mean ABC increased the reported logic level.
- Accuracy columns are copied from `results.csv` before ABC optimization.
