# Persistent Bit-State Smoke Runs (Server 210)

These are bounded implementation checks, not final benchmark evidence. They
were run in the clean remote checkout based on commit `708ae95`, with the
uncommitted `vit_lgn/bitstate` package uploaded from the integration branch.

Environment:

- Host profile: `210`
- Python: 3.10.19
- PyTorch: 2.5.1
- torchvision: 0.20.1
- scikit-learn: 1.7.2
- SciPy: 1.15.3
- Device: CPU

Checks completed:

- 10/10 bit-state unit tests passed.
- All 16 two-input truth tables matched exact Boolean execution.
- Local logic and binary Top-K blocks matched their Boolean reference paths.
- Gradients reached local, query, key, merge, and GroupSum-head gate logits.
- Deployment payload contained no floating values or floating tensors.
- A one-epoch schema run reported zero hard-carrier-to-bit accuracy and loss
  gap.
- A 30-epoch synthetic probe reached 87.5% peak and 85.94% final discrete
  accuracy, crossing 80% at epoch 9. Its temperature-1 soft diagnostic was
  only 53.12%, demonstrating why forward-path alignment and conventional
  soft-relaxation gap must be reported separately.
- A two-epoch sklearn-digits Gumbel-ST run validated the optional dataset and
  stochastic hard-forward paths; its accuracy is not a convergence result.

Representative schema command:

```bash
python -m vit_lgn.bitstate.train_bitstate \
  --dataset synthetic_patterns --method hard_st \
  --device cpu --workers 0 --epochs 1 \
  --train-limit 128 --eval-limit 64 \
  --state-width 16 --local-depth 1 --global-depth 1 \
  --heads 2 --qk-bits 2 --topk 2 --votes-per-class 4 \
  --batch-size 32 --inactive-batches 1 \
  --output-dir runs/bitstate_schema_smoke_20260723
```

The `bitstate_probe30_20260723` artifact predates the additional hard-path
columns. Exact hard-path equality was independently verified by the unit suite
and by the later `bitstate_schema_smoke_20260723` result.
