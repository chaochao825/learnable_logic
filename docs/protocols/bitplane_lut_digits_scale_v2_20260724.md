# Bit-plane LUT scale screen protocol v2

Version 1 exposed all state planes to independent class-code refitting.  Its
bounded 320-sample smoke produced 97.97% inactive planes because a one-vs-rest
target has 90% negative bits.  That smoke is a design diagnosis, not a result.
Version 2 is frozen before full-data runs and changes two structural items:

1. all 512 raw digits bit-planes bypass every learned block as fixed wires;
2. empirical LUT fitting uses integer class-balanced counts, while learned vote
   planes remain the only GroupSum inputs.

## Matched comparison

- Dataset and split: sklearn digits, stratified 70/15/15, split seed
  `20260724`.
- Seeds: `0,1,2`.
- State widths: 672 and 832 Boolean planes.
- Protected carrier: 512 input planes in both widths.
- Learned vote planes: 160 versus 320, giving an exact 2x learned gate/table
  scale comparison.
- Blocks: two sequential hard-frozen blocks; one 4-input LUT per vote plane;
  16 deterministic candidates per LUT input.
- Training: 40 epochs per block, batch 128, AdamW `lr=0.03`, cosine decay,
  hard forward/soft backward, class-balanced bit-code loss 0.25, wiring
  utilization/fanout constraint 0.001.
- Hardening variants: argmax, truth refit, greedy wiring plus truth refit.
- Selection: best validation hard accuracy within each block.  Test is read
  once after both blocks have frozen.

## Pass conditions

- Strict executor exactly matches the hard carrier on all validation/test rows
  with zero real-valued runtime tensors.
- Refit must improve paired validation hard accuracy over argmax to count as a
  method gain.
- 320 learned vote planes must improve mean validation hard accuracy over 160
  under the same budget to support scale expansion.
- Protected input equality must be exact after every block.  Vote-plane
  inactivity, entropy, joint diversity, flip ratio, truth-address coverage,
  selected fanout, unused gates, gate families, payload bits, and timing remain
  mandatory diagnostics.  High entropy without higher hard accuracy is not a
  pass.
