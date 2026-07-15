# Score-gap 50k frozen reproduction snapshot

This directory contains the exact source files and six small `result.json`
files used by the d2/e96 score-gap versus uniform experiment.  It is a frozen
artifact snapshot, not the active package source.

The snapshot was copied from:

```text
/home/spco/sow_linear/.codex_uploads/lgn_vit_50k_20260712
```

The upload was not a clean Git worktree, so a commit identifier is not
invented.  Reproducibility instead uses the file hashes embedded in every
result and verified in `source_manifest.csv`.  The imported-but-disabled
spatial-pyramid module was omitted from the original result's source-hash
dictionary; this snapshot adds its observed SHA256 explicitly.  The common
protocol hash is
`6a60ccd0810391b50c39bccd1678030431e5992856a870285a386b0c087ba9b7`.
It intentionally covers the full experiment matrix, including both variants
and all three seeds, which is why it is identical in the six result files.
The per-run result SHA256 values remain distinct.

From the `source_snapshot/` directory, the original invocation is equivalent
to:

```bash
python -m research_lgn_vit.experiments.run_long_horizon_value \
  --data-root /path/to/cifar-10 \
  --out-dir /path/to/results_50k \
  --variants uniform-final1 score-gap-final1 \
  --seeds 42 43 44 \
  --steps 50000 --batch-size 128 --eval-batch-size 256 \
  --valid-size 5000 --learning-rate 5e-4 --min-learning-rate 1e-5 \
  --warmup-steps 2000 --weight-decay 0.05 \
  --checkpoint-every 5000 --eval-every 5000 \
  --embed-dim 96 --depth 2 --num-heads 3 --attention-k 8 \
  --dataset-provenance cifar-10-python-local-archive
```

The original environment recorded Python 3.10.0, PyTorch 2.6.0+cu124,
torchvision 0.21.0+cu124, CUDA 12.4, deterministic algorithms, split seed
`20260711`, and dataset manifest SHA256
`2af4037d13d0e07fc4783dc5b84487fa8df4942699fe9c43076a3277e6fd49ed`.

The result summary used by the main report is
`../../tables/score_gap_50k_results_20260715.csv`.
