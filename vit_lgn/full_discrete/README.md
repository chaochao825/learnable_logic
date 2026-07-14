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
15.30%.  This is diagnostic evidence only: the paired from-scratch 50k queue
must finish before Shift-RMS or no-norm is judged effective.

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
