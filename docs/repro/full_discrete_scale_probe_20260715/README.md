# Frozen scale-checkpoint probe

`probe_scale_checkpoint.py` is the exact bounded script used to reproduce the
e192/e384 first-block diagnostic in the main report.  It refuses to run unless
the supplied source checkout is at commit
`47e0250cdbeb327925891ec6cf6bd436eca35dbb`, requires a clean checkout,
verifies both checkpoints are at step 50,000, binds each checkpoint to its
protocol and source hashes, verifies the CIFAR-10 archive and extracted files,
and records the exact sample indices and decoded input-batch hash.

The validated command was:

```bash
CUDA_VISIBLE_DEVICES='' python3 probe_scale_checkpoint.py \
  --repo-root /home/wangmeiqi/learnable_logic_full_discrete_scale_ablation_20260713 \
  --runs-root /home/wangmeiqi/learnable_logic_full_discrete_scale_ablation_20260713/runs \
  --data-root /home/wangmeiqi/ViT-LGN_goal6plus_nobias_retry_20260630/data/cifar-10 \
  --runs fdscale_d6e192_control_seed42 fdscale_d6e384_seed42 \
  --probe-images 16 --split-seed 20260711 --device cpu --threads 16
```

The command completed successfully on 2026-07-15.  Its stdout is frozen in
`probe_result.json`.  Key identifiers are:

- probe script SHA256:
  `10c10b9298fc8abda8177b515bbc5193954cc9a68789c990431ade5107172886`
- sample-index SHA256:
  `bae6f408811d712eb8d885b6e93ddc0bcdc8b5a17985a35cca6a142e6f43caaa`
- decoded input-batch SHA256:
  `523389b73e2f0fc3984c41b1a4d62af9a954ae1a3cb50925ef5d00d4550e826d`
- CIFAR-10 archive SHA256:
  `6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce`
- e192 checkpoint SHA256:
  `a16ddb2505d1d394653a372bc0f28895c1402cd5d24c51582974836af16095c2`
- e384 checkpoint SHA256:
  `ca0120b860d510e1c2d81772e9b7add44f67ae594822852589533bf9bf048a8e`

This is a diagnostic over 16 images and one block.  It supports a mechanism
hypothesis but is not an accuracy result or a substitute for a paired 50k
experiment.
