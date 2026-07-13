# Full-discrete scalable ViT

This package is the accuracy/scale-first fully discrete branch of ViT-LGN.
It does not claim that storing a hard value in a floating tensor is sufficient
for deployment.  Each inference primitive has an explicit integer/bit-plane
interpretation:

- patch, QKV, output, FFN, and classifier matrices use signed bit-plane
  coefficients with power-of-two scales;
- Q/K routing uses threshold bits, XNOR/popcount, and hard Top-K;
- selected V rows use score-gap weights `{8,4,2,1}`;
- FFN hidden gates are binary MUX/AND controls over integer activations;
- residuals are requantized to signed integer grids;
- RMS normalization uses integer sum-of-squares plus a reciprocal-square-root
  LUT reference.

For the four-magnitude-plane configuration, an integer coefficient is

`sign * (8*b3 + 4*b2 + 2*b1 + b0)`.

This preserves dense fan-in and scales linearly in width and plane count.  It
is intentionally different from making every scalar a single random two-input
gate, whose useful connectivity may not scale with gate count.

The accuracy-first launcher currently uses seven magnitude planes
`{64,32,16,8,4,2,1}` (signed range `[-127,127]`); the four-plane
`{8,4,2,1}` configuration is a later compression point, not the default claim.

The training graph retains floating shadow parameters and STE surrogates.
Evaluation uses hard thresholds/selectors, integer-aligned signed V,
round-to-nearest integer division, and Q0.15 reciprocal-sqrt LUT payloads.  A
compact exporter is still required before RTL/PPA claims; this package is a
hard-forward QAT and integer-semantics reference, not finished RTL.
