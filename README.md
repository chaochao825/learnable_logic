# learnable_logic

LightLogic / Hard-LGN discretization prototypes, experiment summaries, and
selected publishable artifacts from the 210 server workflow.

This repository is a clean publish tree, not a raw dump of the entire
`/home/spco/sow_linear/hard_lgn_gap_proto` experiment directory. The original
run tree on 210 contains roughly 448MB of `runs/` artifacts, so this repo keeps:

- the core prototype scripts
- concise summaries of methods and results
- key Markdown reports
- small CSV tables that support the main claims

The raw large run directories remain on the 210 server.

## What was validated

1. Goal 0-2: LightLogic input-wise parametrization (`light_iwp`) is implemented
   and compared against 16-parameter OP/DLGN gates, annealing, ST, and
   Gumbel-ST.
2. Goal 3-4: K-expansion from learned continuous gates to larger truth-table
   blocks is implemented for `K=2,4,8,16,32`, with fixed, per-layer, and
   per-neuron calibration.
3. Goal 5-6: post-expansion threshold tuning and teacher-logit distillation are
   implemented and improve deployable hard accuracy in most tested groups.
4. Goal 7: real direct-trained `b>2` LUT networks are implemented and run,
   including matched-setting binarized-MNIST comparisons.
5. Goal 8: local LUT truth tables are exported, minimized with ABC, and
   augmented with local XAG-style AND/XOR/NOT estimates.

## Headline results

- LightLogic IWP uses 4 truth-table parameters per 2-input gate versus 16 for
  OP/DLGN, so the parameter count is 0.25x at matched topology.
- K-expansion reduces local approximation error monotonically with larger `K`,
  but thresholded network accuracy does not improve monotonically.
- Distilled K-expanded students improve hard accuracy in 11/12 tested
  dataset/seed/K/init groups.
- On matched binarized-MNIST settings (`width=800`, `layers=3`, `epochs=80`,
  `train/test=12000/3000`), direct `b=3` LUT training reaches `0.835667`
  discrete accuracy versus `0.786667` for the earlier Goal 8 `b=2` baseline.
- On Boolean tasks, direct `b>2` LUTs improve `majority9` and
  `random_sparse10`, stay flat on `parity8`, and increase logic cost.

## Repository layout

- `hard_lgn_gap_proto/`: core prototype scripts copied from the 210 source tree
- `docs/methods_and_results_summary_20260710.md`: consolidated method/result
  write-up
- `docs/reports/`: key stage reports
- `docs/tables/`: small CSV tables referenced by the summary

## Raw artifact location on 210

The publish source tree remains at:

- `/home/spco/sow_linear/hard_lgn_gap_proto`

The most important report directories are:

- `/home/spco/sow_linear/hard_lgn_gap_proto/runs/lightlogic_report_goal012_v5`
- `/home/spco/sow_linear/hard_lgn_gap_proto/runs/lightlogic_k_report_goal34_v4`
- `/home/spco/sow_linear/hard_lgn_gap_proto/runs/lightlogic_distill_report_goal56_v2`
- `/home/spco/sow_linear/hard_lgn_gap_proto/reports/lightlogic_b_lut_goal7_v2_configdoc_v1`
- `/home/spco/sow_linear/hard_lgn_gap_proto/reports/lut_xag_backend_v1`
