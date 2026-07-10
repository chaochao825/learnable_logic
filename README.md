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

## ViT-LGN attention and token-mixer extension

This repository also includes a curated ViT-LGN import from the 210 server under
`vit_lgn/`. It is a source/report snapshot, not a raw run dump.

Included directories and reports:

- `vit_lgn/goal6plus/`: earlier ViT-LGN Goal6plus baseline, CLS Top-K
  token-mixer ablations, and selected small JSON/CSV artifacts.
- `vit_lgn/attention_clean/`: packed-XNOR/popcount Top-K selector-majority
  attention source snapshot and 200k result summaries.
- `vit_lgn/sctm_scale/`: SCTM sparse CLS token mixer, auxiliary accumulator,
  and value-discretization source snapshot.
- `docs/vit_lgn_method_results_20260710.md`: earlier Goal6plus, CLS Top-K,
  and attention-clean summary.
- `docs/reports/vit_lgn_attention_sctm_report_20260710.md`: consolidated
  current method, configuration, result, directory, and checkpoint-location
  report.
- `docs/tables/vit_lgn_attention_clean_200k.csv`: parsed 200k attention-clean
  results.
- `docs/tables/vit_lgn_sctm_results.csv`: selected SCTM/auxiliary run
  summaries.
- `docs/tables/vit_lgn_weight_manifest_210.csv`: 210 checkpoint manifest.

Highlights:

- CLS-only static top-k mixers can replace late attention blocks in the
  goal6plus ViT-LGN 8k baseline with no accuracy collapse; best imported row is
  `b4,b5/static_weighted/K=8`, hard accuracy `0.6360` versus full-attention
  `0.6243`.
- The attention-clean `K8 aug const` run reaches best valid accuracy `0.7913`
  at 150k steps and final CIFAR-10 test/eval `0.7893` at 200k steps, with
  zero eval/inference hard-path gap.
- The strongest hardware-oriented SCTM line is currently the auxiliary VQ3
  long16k d16/e1024/h32 run at `0.7288` hard/soft accuracy.
- Attention-clean 200k checkpoints were not saved; SCTM checkpoint binaries are
  intentionally kept on 210 and listed in the weight manifest instead of being
  committed to GitHub.
