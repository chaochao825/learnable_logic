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

## Fully discrete ViT and logic-gate audit (2026-07-15)

The cumulative `vit_lgn/full_discrete/` branch now includes the scalable
Wmag/A8 model, enhancement ablations, integer transaction backend, minimal
shared 3x3 logic tree, and both scale launchers.  Completed 50k results and the
current deployment boundary are consolidated in:

- [`docs/full_discrete_logic_gate_report_20260715.md`](docs/full_discrete_logic_gate_report_20260715.md)
- [`docs/tables/full_discrete_50k_results_20260715.csv`](docs/tables/full_discrete_50k_results_20260715.csv)
- [`docs/tables/score_gap_50k_results_20260715.csv`](docs/tables/score_gap_50k_results_20260715.csv)
- [`docs/tables/full_discrete_scale_probe_20260715.csv`](docs/tables/full_discrete_scale_probe_20260715.csv)
- [`docs/repro/score_gap_50k_20260712/`](docs/repro/score_gap_50k_20260712/README.md): exact source snapshot and six raw results
- [`docs/repro/full_discrete_scale_probe_20260715/`](docs/repro/full_discrete_scale_probe_20260715/README.md): executable bounded probe and raw JSON

Current headline: the d6/e192 hard-discrete model reaches 75.30% on the fixed
5,000-example CIFAR-10 validation split, but its main capacity is still dense
Wmag7 shift/add projection.  The learned local logic-tree tables never left
identity in 50k, and the independent whole-model integer/RTL executor is not
yet complete.  The report deliberately separates a hard-discrete numerical
model, a transaction-level integer specification, and a complete logic-gate
executor.

## ScaleLogic-ViT scaling experiment (2026-07-16)

The next accuracy-first candidate fixes the earlier width-scaling confound by
using 12 heads at d12/e384, keeping `head_dim=32` and the Q/K XNOR width at 224.
It adds four early spatially shared Wmag4 depthwise 3x3 branches and retains
content-dependent hard Top-K routing.  A multiplier-free fixed Hadamard global
mixer was also implemented and exported, but matched 1k probes show that it is
a useful negative hardware ablation rather than the primary accuracy path.

- [`docs/logic_vit_scaling_design_20260716.md`](docs/logic_vit_scaling_design_20260716.md)
- [`docs/tables/logic_hadamard_smoke_20260716.csv`](docs/tables/logic_hadamard_smoke_20260716.csv)
- [`docs/reports/logic_hadamard_review_20260716.md`](docs/reports/logic_hadamard_review_20260716.md)
- [`docs/reports/global_lut_tree_review_20260716.md`](docs/reports/global_lut_tree_review_20260716.md)
- [`docs/tables/scalelogic_50k_live_20260716.csv`](docs/tables/scalelogic_50k_live_20260716.csv)

The first formal 5k point is recorded only as an intermediate diagnostic.  A
claim about scaling or the local inductive bias waits for the paired 50k
candidate/control results.

The same branch now also contains an accuracy-expensive nonlinear global
option: a six-stage group-shared A8-by-A8 ROM reduction tree, root/CLS fusion
ROM, and broadcast ROM in parallel with hard Top-K.  At d12/e384 its 12-block
hard table payload is 72 MiB.  It is fully exported as schema v3 and is kept out
of the 50k queue until a matched 1k probe demonstrates value over attention.
