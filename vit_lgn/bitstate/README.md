# Persistent Bit-State LGN

This package is the strict Boolean deployment branch of the ViT/LGN study. It
keeps hidden activations as bits from the image encoder through the GroupSum
classifier instead of converting a quantized integer carrier back to floating
point after every block.

## Execution contract

- Input: `uint8` images (floating `[0, 1]` tensors are quantized once for
  training convenience).
- Encoder: fixed uint8 threshold comparisons and either direct patch-bit
  routing or redundant sparse popcount predicates with learned integer
  thresholds and polarity bits.
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

Wide-state regularized training keeps a lossless subset of the thermometer
bits and fills the remaining state with deployable sparse predicates:

```bash
python -m vit_lgn.bitstate.train_bitstate \
  --dataset cifar10 --method progressive_hard_st \
  --encoder-kind redundant_predicate --state-width 4096 \
  --predicate-fanin 9 --encoder-identity-width 192 \
  --soft-warmup-epochs 10 --epochs 100 --validation-size 5000 --augment \
  --state-balance-weight 0.05 --state-diversity-weight 0.02 \
  --state-flip-weight 0.02 --gate-entropy-weight 0.01 \
  --amp-bfloat16 --output-dir runs/bitstate_cifar10_w4096
```

For matched full-CIFAR H200 runs, `run_h200_long_cifar.sh` keeps the data
split, width, depth, optimizer, and seed fixed while selecting `dlgn`,
`anneal`, `gumbel`, or one of the progressive hard variants. Its defaults are
a 45k/5k train/validation split, the full 10k test split, width 4096, and 30
epochs. Environment variables such as `STATE_WIDTH`, `EPOCHS`, `TRAIN_LIMIT`,
and `EVAL_LIMIT` are intended for bounded capacity and throughput probes.

Completed campaigns can be consolidated without loading checkpoints:

```bash
python -m vit_lgn.bitstate.collect_results \
  remote_runs/h200_screen_20260723 \
  remote_runs/h200_commitment_screen_20260723 \
  --infer-legacy-entropy-unused \
  --output remote_runs/bitstate_results.csv
```

`unused_gate_ratio` follows the Mind-the-Gap gate-entropy definition. The
separate `activation_inactive_gate_ratio` field measures hard gates whose
outputs are constant on the sampled training batches. The collector flag
repairs older bit-state summaries by reading their best checkpoints; without
that flag it labels their legacy ratio explicitly instead of silently mixing
the two definitions.

The default `--gate-init-mode targeted` preserves the model's identity-biased
Boolean state initialization. `--gate-init-mode normal
--gate-init-normal-std 1` instead reproduces the DLGN/GLGN N(0,1) logit
initialization for a paper-aligned baseline. The initialization mode is stored
in every summary and should not be mixed during matched comparisons.

Each new run also records soft-vs-hard MAE and binary flip ratio at the
encoder, every local/global block boundary, and the vote head. For older
checkpoints, the same metrics and both unused-gate definitions can be generated
without retraining:

```bash
python -m vit_lgn.bitstate.analyze_checkpoint runs/example --device cuda
```

The anti-collapse diagnostics report hidden-state entropy, constant and
duplicate bit ratios, layer-to-layer flip rate, gate entropy, and gate
confidence. The compressed four-address LUT evaluator is algebraically
identical to mixing all 16 Boolean functions while avoiding a 16x activation
tensor at every gate layer.

`--group-sum-temperature` scales class vote counts only for the training and
reported cross-entropy; it cannot change integer-vote argmax predictions or
the deployment payload. Optional online distillation accepts the verified
attention-clean CIFAR-10 checkpoint through `--teacher-source-dir`,
`--teacher-checkpoint`, and `--teacher-alpha`. The teacher remains an explicit
floating training aid and is never exported with the Boolean student.
`--gate-init-strength` controls the initial selected-vs-alternative truth-table
logit margin; small positive values preserve the same hard initial circuit but
let gate argmax choices change much earlier than the legacy high-margin setup.
At the soft-to-hard boundary, `--hardening-logit-scale` can increase
truth-table commitment without changing any selected gate. Entropy pressure
can be ramped with `--gate-entropy-weight-start` and
`--gate-entropy-weight-end` so early exploration and late commitment are
controlled separately.

The default global token applies one majority comparison per state channel.
`--global-token-mode learned_count` replaces that information bottleneck with
one trainable count threshold per channel. Its forward value is still exactly
Boolean; deployment stores only rounded integer thresholds, and execution uses
patchwise popcount plus integer comparison. `hard_scale16_count_token` exposes
this as a separate H200 ablation without changing the matched baseline runs.
