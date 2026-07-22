# Persistent Bit-State LGN

This package is the strict Boolean deployment branch of the ViT/LGN study. It
keeps hidden activations as bits from the image encoder through the GroupSum
classifier instead of converting a quantized integer carrier back to floating
point after every block.

## Execution contract

- Input: `uint8` images (floating `[0, 1]` tensors are quantized once for
  training convenience).
- Encoder: fixed uint8 threshold comparisons and patch-bit routing.
- Local blocks: fixed spatial wiring, trainable 2-input Boolean truth tables,
  and exact identity highways.
- Global blocks: Boolean query/key gates, XNOR-popcount scores, stable hard
  Top-K, bitwise-majority value aggregation, and a Boolean merge layer.
- Classifier: Boolean votes followed by integer GroupSum.
- Deployment parameters: integer indices, integer truth-table IDs, thresholds,
  and Boolean/integer metadata only.

`hard_st` training uses hard forward values and soft surrogate gradients.
`forward_bits` is a separate strict reference path and must match every hidden
boundary and final class vote count exactly.

## Quick verification

```bash
python -m unittest vit_lgn.bitstate.test_bitstate -v
```

One-run synthetic smoke experiment:

```bash
python -m vit_lgn.bitstate.train_bitstate \
  --dataset synthetic_patterns --method hard_st_cage \
  --epochs 2 --train-limit 512 --eval-limit 128 \
  --output-dir runs/bitstate_smoke
```

The runner also supports `sklearn_digits`, `mnist`, and `cifar10`. Every run
writes `summary.json` and the required comparison columns in `result.csv`.
It reports two distinct gaps: the conventional deterministic soft-relaxation
versus argmax gap, and the hard-carrier versus strict bit-execution gap. The
latter must be exactly zero and is checked before a result is written.

The implementation intentionally reports trainable 2-input gate count
separately from the unexpanded XNOR-popcount/Top-K routing fabric. ABC/Yosys
export should expand that fabric before reporting synthesized primitive count,
depth, and fanout.
