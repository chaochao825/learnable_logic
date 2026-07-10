# SCTM Scale Snapshot

Source snapshot for the Sparse CLS Patch Token Mixer experiments from:

```text
/home/spco/sow_linear/ViT-LGN_goal6plus_sctm_scale_20260628
```

Important entry points:

- `sctm_token_mixer_experiment.py`: SCTM, auxiliary accumulator, value quantization, and bitplane aggregation experiments.
- `logic_vit_tiny.py`: ViT-LGN model definition with SCTM mixer integration.
- `train_logic_vit_tiny.py`: training/evaluation driver used by the Goal6plus/SCTM branch.
- `launch_sctm_long16k_210.sh`: representative large 16k-step SCTM launch script.

Final checkpoint binaries are not committed. See `../../docs/tables/vit_lgn_weight_manifest_210.csv` for 210 paths.
