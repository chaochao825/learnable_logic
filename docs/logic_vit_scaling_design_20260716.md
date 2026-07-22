# Hardware-aware scaling design for a logic-oriented ViT

Date: 2026-07-16

## Decision

The accuracy-first scalable candidate is **ScaleLogic-ViT**:

- 12 blocks, embedding width 384;
- 12 heads, so head dimension remains 32 and the Q/K XNOR width remains 224;
- signed A8 activations, Wmag7 shift/add projections and RMS-LUT normalization;
- hard XNOR-popcount Top-K=8 routing and integer `{8,4,2,1}` selected-V aggregation;
- four early spatially shared Wmag4 depthwise 3x3 residual branches;
- binary FFN gate, with Wmag7 gate/up/down channel maps;
- no softmax, exponential, GELU or general learned multiplier in the hard inference contract.

This is not claimed to be a completed two-input-gate executor.  It is the most
promising current hard-discrete numerical model, chosen after fixed global
mixing alternatives were implemented and measured rather than assumed useful.

## Why this design

### Evidence from the completed local experiments

The previous scale matrix kept six heads while doubling width.  That changed
head dimension from 32 to 64 and XNOR width from 224 to 448.  The frozen probe
showed mean selected score gap increasing from 2.36 to 4.34 and the minimum
selected-V weight bucket increasing from 7.54% to 30.34%.  Thus width and the
absolute gap quantizer were changed simultaneously.

ScaleLogic-ViT holds head dimension and XNOR width constant by using 12 heads
at width 384.  The local branch adds only 13,824 W4 kernel scalars and shares
each kernel over the 8x8 patch grid.  It supplies the local/translation bias
that low-bit ViTs otherwise lack without introducing dense learned routing.

The formal candidate contains:

| Quantity | Value |
|---|---:|
| Trainable shadow parameters | 28,372,992 |
| Shift/add weight scalars | 28,333,824 |
| Wmag magnitude-enable bits | 198,336,768 |
| ShiftAddLinear layers | 62 |
| Local W4 kernel scalars | 13,824 |
| Local W4 magnitude-enable bits | 55,296 |
| Attention layers | 12 |
| Head dimension / XNOR width | 32 / 224 |

Magnitude-enable bits are conditional shifted-input terms, **not** a standard
cell gate count.  They show that the static Boolean control payload is already
in the hundred-million-bit regime, comparable in scale but not numerically
equivalent to a reported number of logic gates.

### Relation to primary prior work

- [Convolutional Differentiable Logic Gate Networks](https://arxiv.org/abs/2411.04732)
  scales logic networks through spatial sharing, deep logic trees, logical OR
  pooling and residual initialization.  It reports 86.29% CIFAR-10 accuracy
  with 61M logic gates.  The transferable ideas here are fixed local wiring,
  spatial sharing and residual insertion; its gate count is not directly
  comparable with Wmag bitplanes.
- [BinaryViT](https://openaccess.thecvf.com/content/CVPR2023W/ECV/html/Le_BinaryViT_Pushing_Binary_Vision_Transformers_Towards_Convolutional_Models_CVPRW_2023_paper.html)
  finds that binary ViTs benefit from CNN-like average pooling, multi-branch
  local structure, residual affine control and a pyramid.  This supports
  retaining explicit spatial bias instead of relying on a pure token sequence.
- [Q-ViT](https://papers.nips.cc/paper_files/paper/2022/hash/deb921bff461a7b0a5c344a4871e7101-Abstract-Conference.html)
  identifies low-bit attention-map distortion and uses information
  rectification plus distribution-guided distillation.  Its high accuracy
  demonstrates that training-only rectification/distillation can be added
  later without changing the hard inference operator.
- [PackQViT](https://papers.nips.cc/paper_files/paper/2023/hash/1c92edb990a05f2269f0cc3afbb4c952-Abstract-Conference.html)
  emphasizes log2 quantization, fixed outlier treatment and integer-only
  softmax/norm/activation.  Our power-of-two scales and RMS ROM follow the same
  hardware principle, while hard Top-K removes softmax entirely.
- [FNet](https://arxiv.org/abs/2105.03824) establishes that an unparameterized
  Fourier transform can replace attention in some settings.  A complex FFT,
  however, needs twiddle multiplication and is content independent.
- [Monarch Mixer](https://arxiv.org/abs/2310.12109) generalizes structured
  transforms and scales sub-quadratically, but its learned block matrices are
  GEMM-oriented rather than a pure fixed-gate datapath.

## Fixed Hadamard global mixer: implementation and negative result

The repository now implements an exact integer global branch

```text
patch update = round((H * D_block * H * patch_code) / N) >> branch_shift
CLS update   = round(sum(patch_code) / N) >> branch_shift
```

where `H` is a Walsh-Hadamard butterfly and `D_block` is a frozen +/-1 mask.
For 64 patches it uses two six-stage butterflies: 768 add/sub operations per
channel, 64 sign controls, a `>>6` normalization and no multiplier.  A common
power-of-two A8 scale is shared by each 32-channel group.  The training value
is exactly the integer reference; a floating butterfly only supplies the STE
gradient.  The formal `54229c1` snapshot stores the sign masks and structural
ABI as schema v2; the current exporter is schema v5 after adding the nonlinear
global ROM-tree payload and its requantize/merge/topology boundary.
The branch input is A8, but its butterfly and branch values use explicitly
bounded wide accumulators (14/20 bits through the two 64-token transforms and
13 bits after `branch_shift=2` in the worst case).  The enclosing block, not
the mixer itself, performs the residual merge and requantizes that result to
A8.  This boundary is now part of the exported ABI.

The 1k-step paired probes show that hardware simplicity did not imply useful
visual content routing:

| d6/e192 method | 500-step acc. | 1k acc. | GiB | sec/step |
|---|---:|---:|---:|---:|
| Pure fixed Hadamard | 22.68% | 26.66% | 1.10 | 0.159 |
| 2/3 Hadamard, 1/3 hard attention | 30.76% | 31.92% | 1.92 | 0.172 |
| Attention plus weak Hadamard side branch | 41.02% | 47.42% | 3.18 | 0.214 |
| Hard attention control | **43.00%** | **49.30%** | 3.18 | 0.185 |

These are smoke diagnostics, not final accuracy claims.  They reject fixed
linear global mixing as the primary candidate under this training protocol.
The code is retained as a reproducible negative ablation and as a hardware
reference primitive.

## Formal 50k protocol

The formal candidate is
`scalelogic_d12e384_h12_local4_seed42_50k`.  It uses the same CIFAR-10 split,
seed, augmentation, optimizer, learning-rate schedule, W7/A8 precision and
50,000-step budget as the earlier scale matrix.  The paired control changes
only `local_layers=4` to `local_layers=0`; both use 12 heads.

At 5k the candidate reaches 59.58%, versus 55.54% for the historical
d12/e384-h6 row and 59.52% for d12/e192.  This is encouraging intermediate
evidence only.  A method conclusion requires the 50k result and the h12
no-local control.

At 10k the candidate reaches 66.64%.  The historical d12/e384-h6, d12/e192
and d6/e192 rows were 61.36%, 64.68% and 64.14%, respectively, at the same
step.  The gains are therefore +5.28, +1.96 and +2.50 points, but this still
cannot separate the 12-head score-width repair from the local branch.  GPU2
contention increased the second interval from 0.412 to 0.550 seconds/step; the
process and checkpoint remain healthy.  Exact live rows are in
[`docs/tables/scalelogic_50k_live_20260716.csv`](tables/scalelogic_50k_live_20260716.csv).

The curve continues to 68.30% at 15k and 69.60% at 20k.  The third interval
slowed to 0.780 seconds/step under sharing, then recovered to 0.414 seconds/step
without a memory increase (peak remains 11.756 GiB).  This supports run health,
not final method efficacy: only the 50k local4/local0 pair can attribute the
gain to the spatial branch rather than the corrected 12-head score width.

The active run is frozen to source commit `54229c1`, ordered source-set hash
`9e8af5d0bd8a42e3b3e913d6a94cbdb9f654c14f005c4db5e6697bb0c6997c94`
and protocol hash
`fe9b968f3be661a3ed8cbf77649978774ba525afddf725fe79c127523ec37198`.
Post-launch exporter/ABI review fixes are intentionally separate from this
training snapshot.  The review record is
[`docs/reports/logic_hadamard_review_20260716.md`](reports/logic_hadamard_review_20260716.md).

## Hardware interpretation

The hard operator set is:

- sign tests, XNOR and popcount for Q/K;
- fixed-depth compare/swap Top-K with deterministic key-index tie breaking;
- shifts/adds for Wmag7 projections and `{8,4,2,1}` V weights;
- an integer numerator and small positive denominator for selected V;
- 3x3 depthwise W4 conditional shifts/adds;
- a 16,130-entry Q0.15 reciprocal-square-root ROM for A8 RMSNorm;
- a comparator/MUX for the hard FFN gate.

Training may use FP32 shadows, AdamW and sigmoid STEs.  None of those belong in
the hardened payload.  Schema v5 now specifies exponent-aligned residual
addition, but fixed-width hardware still needs a maximum exponent range or
saturation rule.  Other remaining work is the selected-V divider/LUT, packed
kernels and whole-model bit-exact C++/RTL validation.

## Implemented nonlinear global option

A fixed linear FFT/Hadamard is too weak, so the repository now includes the
more expressive ConvLogic-style nonlinear tree.  Its primitive is a shared
two-input A8 fusion ROM:

- address: `(a_code << 8) | b_code` (65,536 entries);
- payload: one signed A8 output per address;
- six fixed reduction stages aggregate 64 patches globally;
- stage/group ROMs are shared across spatial nodes and channels within a group;
- a root/CLS ROM creates a global context;
- a broadcast ROM conditions every patch and CLS on that context;
- reduce/context tables initialize to rounded average, the broadcast table to
  projection-B, and a final `>>2` keeps the initial residual branch weak.

One 65,536x8 ROM is 64 KiB.  With 12 groups, six reduction stages plus
context/broadcast require 96 ROMs per d12/e384 block: 6,291,456 A8 entries or
6 MiB.  Across 12 blocks the hard payload is 72 MiB and the shadow table count
is 75,497,472 parameters.  This is deliberately expensive but scales linearly
in depth and channel groups, has no learned connections, and is not a claimed
gate count.

Hard forward has an independent scalar oracle and exports every int8 table.
Training uses four-entry bilinear interpolation only as a surrogate; the
returned value is the exact hard lookup.  It is added in parallel with hard
Top-K, not used as a replacement.  The matched 1k gate is negative: 45.58%
versus 48.80% for attention alone, despite 42,953 deployed table changes.
Its LUT gradient norm is only 0.00372 versus 6.222 for the base network, so the
loss is not caused by the shared gradient clip.  The mechanism is therefore
rejected from the 50k queue; raw pair-ROM capacity is not sample-efficient
global capacity under this protocol.
The implementation review is
[`docs/reports/global_lut_tree_review_20260716.md`](reports/global_lut_tree_review_20260716.md).
