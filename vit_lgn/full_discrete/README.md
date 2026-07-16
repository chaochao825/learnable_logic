# Full-discrete scalable ViT

This package is the accuracy/scale-first fully discrete branch of ViT-LGN.
It does not claim that storing a hard value in a floating tensor is sufficient
for deployment.  Each inference primitive has an explicit integer/bit-plane
interpretation:

- patch, QKV, output, FFN, and classifier matrices use signed bit-plane
  coefficients with power-of-two scales;
- Q/K routing uses threshold bits, XNOR/popcount, and hard Top-K with ties
  resolved by lower key index;
- selected V rows use score-gap weights `{8,4,2,1}`;
- FFN hidden gates are binary MUX/AND controls over integer activations;
- residuals are requantized to signed integer grids;
- normalization is selectable: Q0.15 RMS ROM, nearest-power-of-two Shift-RMS,
  requantization only, or a strict identity control.

For the four-magnitude-plane configuration, an integer coefficient is

`sign * (8*b3 + 4*b2 + 2*b1 + b0)`.

This preserves dense fan-in and scales linearly in width and plane count.  It
is intentionally different from making every scalar a single random two-input
gate, whose useful connectivity may not scale with gate count.

The accuracy-first launcher currently uses seven magnitude planes
`{64,32,16,8,4,2,1}` (signed range `[-127,127]`); the four-plane
`{8,4,2,1}` configuration is a later compression point, not the default claim.

The logic-friendly Shift-RMS path computes `S=sum(code**2)` and selects `k` by
constant comparisons `S >= dim * 2**(2*k-1)`.  Its deployed output keeps the
A8 code and replaces the exponent with `-k`; it has no reciprocal ROM or
per-element normalization multiply.  A frozen 50k checkpoint probe retained
73.60% versus 74.68% for Q15 RMS, while deleting every norm collapsed to
15.30%.  The completed from-scratch 50k queue gives 75.30% for Q15 RMS,
71.80% for Shift-RMS in both block and final positions, 74.60% for Shift-RMS
blocks with no final norm, 70.10% for requant-only, and 70.64% for no norm.
Thus block range conditioning remains necessary, while exponent-only
Shift-RMS plus a repaired final calibration path remains promising.

The training graph retains floating shadow parameters, AdamW state, and STE
surrogates.  `export_logic_payload.py` removes those objects and exports only
integer codes, signed scale exponents, U4 chunks/bitplanes, routing constants,
and structural metadata.  `logic_lut` evaluation replaces learned matrix
`F.linear` operations with the exact A8-by-U4 ROM transaction reference and
uses packed XNOR-popcount for Q/K.

This is not finished RTL.  The remaining deployment boundary is the
accumulator/residual-to-A8 exponent-only requantizer; the Python reference
still carries powers of two in floating tensors there.  Packed high-throughput
device kernels, a cycle-accurate executor, fixed-width residual/exponent
alignment, and synthesis/PPA are also still required.

## Minimal shared 3x3 logic-tree branch

`SharedLogicTreeConv3x3` is a deliberately narrower alternative to the older
learned dense LogicExpert.  It splits every signed-magnitude A8 channel into
eight bitplanes and instantiates one fixed depth-three tree per
`(channel, bitplane)`:

- eight leaves are gathered from a local zero-padded 3x3 window;
- leaf zero is always the same channel/bitplane at the centre;
- the other seven leaves cover seven neighbours, with the omitted direction
  rotating deterministically over channel and bitplane;
- seven two-input 4-bit LUTs use the fixed `8 -> 4 -> 2 -> 1` topology;
- LUT parameters are shared over H/W and no connection logits exist;
- the hard address is `(A << 1) | B`, so projection A is packed as `0xC`;
- every hard LUT starts as A, making the root bit-exactly equal to the centre
  bitplane at initialization;
- CLS bypasses the tree and the root has no learned W7 projection.

The Boolean executor uses fixed gather/slice routing plus LUT shift/mask
indexing and has tests that forbid `matmul`, `F.linear`, and `conv2d`.  Training
returns that exact hard value and uses a separate soft derivative surrogate.
A small alternating sub-threshold B probe is present only in shadow logits;
all exported hard tables remain `0xC`, while all seven gates receive gradient
from the first optimization step.

The surrounding PyTorch carrier bridge still calls the existing activation
quantizer to obtain an A8 code and power-of-two scale.  Thus this branch proves
the local tree itself is LUT/gather-only; it does not yet claim that exponent
selection and transaction decode are a packed RTL executor.

The paired 50k protocol is `launch_logic_tree_pair_210.sh`.  It runs
`local_layers=3`, `1`, and `0` from the same source and seed.  This separates a
single local injection from repeated early-layer application and its matched
control.  Every 5k validation also evaluates a temporary forced-`0xC` version
and records hard LUT flips plus root-bit changes.  All three rows completed:
the control reached 75.30%, one tree layer reached 75.76%, and three layers
reached 58.28%.  Learned-hard and forced-`0xC` accuracies were identical,
every hard LUT remained `0xC`, and every root-code change rate was zero.  The
tree therefore added no deployed hard expression in this protocol; the
one-layer difference is a training-surrogate effect or single-seed noise.

The complete method/result/probe audit is
[`docs/full_discrete_logic_gate_report_20260715.md`](../../docs/full_discrete_logic_gate_report_20260715.md).

## ScaleLogic-ViT and fixed global mixer ablation

`enhancements_hadamard.py` adds a parameter-free integer Walsh--Hadamard
global branch.  Its hard value is computed by two FWHT butterfly networks,
one frozen sign mask, rounded power-of-two shifts, a rounded patch mean for
CLS, and a CLS broadcast to every patch.  The exporter records the complete
sign mask and arithmetic ABI in the integer payload schema.

The fixed branch is available as a full attention replacement, a periodic
hybrid, or a weak parallel side branch.  All three 1k probes underperform the
matched content-dependent XNOR/popcount Top-K control, so the formal scalable
candidate keeps hard attention.  It instead uses 12 heads at width 384 to hold
head dimension 32/XNOR width 224 constant and applies the spatially shared
Wmag4 3x3 branch in the first four blocks.  This separates width scaling from
the score-gap quantizer width confound in the earlier six-head matrix.

The design rationale, exact static accounting, negative mixer ablation and
frozen 50k paired protocol are documented in
[`docs/logic_vit_scaling_design_20260716.md`](../../docs/logic_vit_scaling_design_20260716.md).
The report deliberately treats 5k accuracy as an intermediate diagnostic; the
method conclusion requires both the 50k local4 candidate and its local0 paired
control.

## Nonlinear A8 global LUT tree

`enhancements_global_lut.py` implements the hardware-expensive nonlinear
alternative to a fixed FFT/Hadamard.  In every enabled block it:

1. quantizes all tokens with one power-of-two scale per 32-channel group;
2. reduces 64 patches through six balanced stages of two-input A8-by-A8 ROMs;
3. fuses the root with CLS through another ROM;
4. broadcasts that context to CLS and every patch through a final ROM and a
   rounded right shift.

ROMs are shared across spatial tree nodes and channels within a group, while
stages, channel groups and blocks have independent payloads.  Hard forward is
only signed-code biasing, address concatenation, ROM indexing, fixed wiring and
shifts.  A four-entry bilinear interpolation is training-only and the returned
forward value remains the exact hard lookup.  At d12/e384, one block contains
96 ROMs, 6,291,456 learned A8 entries (6 MiB), and 12 blocks contain 72 MiB of
hard table payload.  These bytes are not reported as standard-cell gate count.

The current exporter is schema v3 and serializes every int8 reduce/context/
broadcast table.  This branch is evaluated in parallel with hard Top-K rather
than replacing content routing, because the fixed-mixer ablation showed that a
global path without content dependence is insufficient.
