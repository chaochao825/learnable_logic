# ViT-LGN Attention And Token-Mixer Snapshots

This directory contains clean source snapshots from the 210 ViT-LGN experiments. It is not a raw run dump.

- `goal6plus/`: earlier ViT-LGN Goal6plus baseline, CLS Top-K token-mixer ablations, and selected small JSON/CSV artifacts.
- `attention_clean/`: packed-XNOR Top-K selector-majority attention branch. The 200k runs used `--no-save-checkpoints`, so only logs/results exist on 210.
- `sctm_scale/`: Sparse CLS Patch Token Mixer, auxiliary accumulator, and value-discretization branch. Final checkpoints remain on 210 and are listed in `../docs/tables/vit_lgn_weight_manifest_210.csv`.
- `full_discrete/`: 34-server Wmag7/A8 full-discrete ViT, exact product-LUT and packed-XNOR transaction reference, normalization controls, local shift-add/Hadamard/LHVM/logic-tree experiments, exporter, launchers, and tests.

Excluded from this repository: CIFAR data, `runs/`, full logs, large diagnostic histograms, `__pycache__`, compiled extensions, and large `.pt` checkpoint binaries.

See `../docs/vit_lgn_method_results_20260710.md` for the earlier Goal6plus/CLS/attention-clean summary and `../docs/reports/vit_lgn_attention_sctm_report_20260710.md` for the current attention-clean/SCTM report.

For the full-discrete snapshot, see
`../docs/reports/full_discrete_logic_gate_report_20260716.md` and
`../docs/tables/full_discrete_results_20260716.csv`. The active 50k ScaleLogic
checkpoints remain on server 34.
