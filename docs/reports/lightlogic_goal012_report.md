# LightLogic Goal Evidence

Merged runs: lightlogic_bool_seed0_goal012_sigmoid_v1, lightlogic_bool_seed0_goal012_sinusoidal_v1, lightlogic_digits_seed0_goal02_sinusoidal_v1

## Goal Checks

| goal | status | evidence | note |
| --- | --- | --- | --- |
| Goal 0 LightLogic baseline | PASS | light_iwp_rows=7; malformed_rows=0 | Rows report continuous_acc, discrete_acc, acc_gap, utilization, gate count, parameter count, and train time. |
| LightLogic parameter reduction vs OP/DLGN | PASS | lightlogic_bool_seed0_goal012_sigmoid_v1/parity8/seed0 param_ratio=0.25; lightlogic_bool_seed0_goal012_sigmoid_v1/majority9/seed0 param_ratio=0.25; lightlogic_bool_seed0_goal012_sigmoid_v1/random_sparse10/seed0 param_ratio=0.25; lightlogic_bool_seed0_goal012_sinusoidal_v1/parity8/seed0 param_ratio=0.25; lightlogic_bool_seed0_goal012_sinusoidal_v1/majority9/seed0 param_ratio=0.25; lightlogic_bool_seed0_goal012_sinusoidal_v1/random_sparse10/seed0 param_ratio=0.25; lightlogic_digits_seed0_goal02_sinusoidal_v1/digits/seed0 param_ratio=0.25 | For two-input gates, IWP uses four truth-table parameters versus sixteen OP parameters. |
| Goal 1 annealing+entropy improves LightLogic gap without hurting hard accuracy | PARTIAL | lightlogic_bool_seed0_goal012_sigmoid_v1/parity8/seed0/sigmoid delta_gap=-0.125 delta_hard=-0.09375 delta_util=-1 util_ok=False; lightlogic_bool_seed0_goal012_sigmoid_v1/majority9/seed0/sigmoid delta_gap=-0.492188 delta_hard=0.03125 delta_util=-1 util_ok=False; lightlogic_bool_seed0_goal012_sigmoid_v1/random_sparse10/seed0/sigmoid delta_gap=-0.347656 delta_hard=-0.125 delta_util=-1 util_ok=False | Current sigmoid annealing evidence must be checked for collapse through gate_utilization. |
| Goal 2 ST/Gumbel-ST improves final hard LightLogic accuracy | PARTIAL | lightlogic_bool_seed0_goal012_sigmoid_v1/parity8/seed0/sigmoid best_st=light_iwp_st delta_hard=0 delta_gap=0 delta_util=0 util_ok=True; lightlogic_bool_seed0_goal012_sigmoid_v1/majority9/seed0/sigmoid best_st=light_iwp_st delta_hard=0 delta_gap=-0.195312 delta_util=0 util_ok=True; lightlogic_bool_seed0_goal012_sigmoid_v1/random_sparse10/seed0/sigmoid best_st=light_iwp_st delta_hard=0.285156 delta_gap=0 delta_util=-0.0195312 util_ok=True; lightlogic_bool_seed0_goal012_sinusoidal_v1/parity8/seed0/sinusoidal best_st=light_iwp_gumbel_st delta_hard=0 delta_gap=-0.15625 delta_util=0 util_ok=True; lightlogic_bool_seed0_goal012_sinusoidal_v1/majority9/seed0/sinusoidal best_st=light_iwp_st delta_hard=0.484375 delta_gap=-0.398438 delta_util=-0.0078125 util_ok=True; lightlogic_bool_seed0_goal012_sinusoidal_v1/random_sparse10/seed0/sinusoidal best_st=light_iwp_st delta_hard=0.125 delta_gap=0.191406 delta_util=-0.0449219 util_ok=True; lightlogic_digits_seed0_goal02_sinusoidal_v1/digits/seed0/sinusoidal best_st=light_iwp_st delta_hard=0.0511111 delta_gap=0.0577778 delta_util=-0.222656 util_ok=False | This is same-architecture evidence against plain Light IWP, not a claim against tuned OP/DLGN. |
| Best current LightLogic-family hard accuracy is competitive with OP/DLGN | PARTIAL | lightlogic_bool_seed0_goal012_sigmoid_v1/parity8/seed0/sigmoid best_light=light_iwp delta_hard_vs_op=0.265625; lightlogic_bool_seed0_goal012_sigmoid_v1/majority9/seed0/sigmoid best_light=light_iwp_anneal delta_hard_vs_op=-0.351562; lightlogic_bool_seed0_goal012_sigmoid_v1/random_sparse10/seed0/sigmoid best_light=light_iwp_st delta_hard_vs_op=-0.0742188; lightlogic_bool_seed0_goal012_sinusoidal_v1/parity8/seed0/sinusoidal best_light=light_iwp delta_hard_vs_op=0.265625; lightlogic_bool_seed0_goal012_sinusoidal_v1/majority9/seed0/sinusoidal best_light=light_iwp_st delta_hard_vs_op=0.109375; lightlogic_bool_seed0_goal012_sinusoidal_v1/random_sparse10/seed0/sinusoidal best_light=light_iwp_st delta_hard_vs_op=-0.0234375; lightlogic_digits_seed0_goal02_sinusoidal_v1/digits/seed0/sinusoidal best_light=light_iwp_st delta_hard_vs_op=-0.00666667 | A lower parameter count alone is not an accuracy win; inspect discrete_acc and acc_gap. |

## Top Rows By Discrete Accuracy

| method | dataset | seed | continuous_acc | discrete_acc | acc_gap | gate_utilization | gate_count | parameter_count | train_time | source_run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light_iwp_st | majority9 | 0 | 0.9140625 | 1.0 | 0.0859375 | 0.984375 | 512 | 2048 | 3.934573855018243 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| dlgn_op | random_sparse10 | 0 | 1.0 | 0.96875 | 0.03125 | 0.947265625 | 512 | 8192 | 5.9145505130290985 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| dlgn_op | random_sparse10 | 0 | 1.0 | 0.96875 | 0.03125 | 0.947265625 | 512 | 8192 | 6.313869317993522 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_st | random_sparse10 | 0 | 0.57421875 | 0.9453125 | 0.37109375 | 0.94921875 | 512 | 2048 | 5.244349231943488 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| dlgn_op | digits | 0 | 0.9288888888888889 | 0.9022222222222223 | 0.026666666666666616 | 0.94296875 | 1280 | 20480 | 12.659381387988105 | lightlogic_digits_seed0_goal02_sinusoidal_v1 |
| light_iwp_st | digits | 0 | 0.7555555555555555 | 0.8955555555555555 | 0.14 | 0.71875 | 1280 | 5120 | 12.041532267117873 | lightlogic_digits_seed0_goal02_sinusoidal_v1 |
| light_iwp_st | random_sparse10 | 0 | 0.546875 | 0.89453125 | 0.34765625 | 0.98046875 | 512 | 2048 | 5.53163597593084 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| dlgn_op | majority9 | 0 | 1.0 | 0.890625 | 0.109375 | 0.98828125 | 512 | 8192 | 5.733491702005267 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| dlgn_op | majority9 | 0 | 1.0 | 0.890625 | 0.109375 | 0.98828125 | 512 | 8192 | 5.471586094005033 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp | digits | 0 | 0.9266666666666666 | 0.8444444444444444 | 0.0822222222222222 | 0.94140625 | 1280 | 5120 | 11.230224698083475 | lightlogic_digits_seed0_goal02_sinusoidal_v1 |
| light_iwp | random_sparse10 | 0 | 1.0 | 0.8203125 | 0.1796875 | 0.994140625 | 512 | 2048 | 6.2948054790031165 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp | random_sparse10 | 0 | 0.95703125 | 0.609375 | 0.34765625 | 1.0 | 512 | 2048 | 5.440145344939083 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_gumbel_st | random_sparse10 | 0 | 0.82421875 | 0.58203125 | 0.2421875 | 1.0 | 512 | 2048 | 6.339991758810356 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_gumbel_st | random_sparse10 | 0 | 0.68359375 | 0.56640625 | 0.1171875 | 1.0 | 512 | 2048 | 6.930132438894361 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_anneal | majority9 | 0 | 0.5390625 | 0.5390625 | 0.0 | 0.0 | 512 | 2048 | 3.6318035449367017 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp | majority9 | 0 | 1.0 | 0.515625 | 0.484375 | 0.9921875 | 512 | 2048 | 4.727723304182291 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp | majority9 | 0 | 1.0 | 0.5078125 | 0.4921875 | 1.0 | 512 | 2048 | 3.2362706849817187 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_gumbel_st | majority9 | 0 | 0.71875 | 0.5078125 | 0.2109375 | 1.0 | 512 | 2048 | 3.4913864589761943 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_gumbel_st | majority9 | 0 | 1.0 | 0.5078125 | 0.4921875 | 1.0 | 512 | 2048 | 4.937173892976716 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_st | majority9 | 0 | 0.8046875 | 0.5078125 | 0.296875 | 1.0 | 512 | 2048 | 3.2755812990944833 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp | parity8 | 0 | 0.375 | 0.5 | 0.125 | 1.0 | 512 | 2048 | 2.5205898210406303 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp | parity8 | 0 | 0.21875 | 0.5 | 0.28125 | 1.0 | 512 | 2048 | 3.531568788923323 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_gumbel_st | parity8 | 0 | 0.46875 | 0.5 | 0.03125 | 1.0 | 512 | 2048 | 2.302145409863442 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_gumbel_st | parity8 | 0 | 0.375 | 0.5 | 0.125 | 1.0 | 512 | 2048 | 3.3517618409823626 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_st | parity8 | 0 | 0.375 | 0.5 | 0.125 | 1.0 | 512 | 2048 | 2.1013984461314976 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_anneal | random_sparse10 | 0 | 0.484375 | 0.484375 | 0.0 | 0.0 | 512 | 2048 | 5.841853669146076 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| light_iwp_st | parity8 | 0 | 0.453125 | 0.46875 | 0.015625 | 0.9609375 | 512 | 2048 | 3.0406203230377287 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |
| light_iwp_anneal | parity8 | 0 | 0.40625 | 0.40625 | 0.0 | 0.0 | 512 | 2048 | 2.3432663220446557 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| dlgn_op | parity8 | 0 | 0.265625 | 0.234375 | 0.03125 | 0.91015625 | 512 | 8192 | 2.186367033980787 | lightlogic_bool_seed0_goal012_sigmoid_v1 |
| dlgn_op | parity8 | 0 | 0.265625 | 0.234375 | 0.03125 | 0.91015625 | 512 | 8192 | 4.5514336039777845 | lightlogic_bool_seed0_goal012_sinusoidal_v1 |

Interpretation guardrails:
- Goal 0 is satisfied when plain `light_iwp` rows exist with continuous/discrete/gap/utilization/time/gate metrics.
- Goal 1/2 are only supported where the same dataset/seed comparison improves hard accuracy or gap without collapsing utilization.
- OP/DLGN remains the direct 16-parameter-per-gate baseline; LightLogic parameter savings do not imply an accuracy win.
