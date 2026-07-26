# Bit-plane LUT scale screen protocol

## Question

Can a model whose learned payload contains only LUT truth bits and wiring
indices preserve an eight-bit state, improve hard classification accuracy when
width is increased, and benefit from per-block hard truth-table refitting?

## Frozen comparison

- Dataset: sklearn digits, original integer pixels in `[0,16]`.
- Input representation: 64 unsigned A8 symbols, exactly 512 Boolean planes.
- Split: stratified 70/15/15 train/validation/test with split seed `20260724`.
- Seeds: `0,1,2`.
- State widths: 640 and 1280 Boolean planes (80 and 160 A8 symbols).
- Blocks: two state-preserving layers, one 4-input LUT per output plane per
  block, 16 deterministic candidate wires per LUT input.
- Readout: fixed interleaved GroupSum, no learned classifier weights.
- Budget: 40 epochs per block, batch 128, AdamW, validation-only checkpoint
  selection inside each block.
- Declared variables: state width and hardening mode (`argmax`, truth-table
  refit, or greedy wiring plus truth-table refit).
- Training-only values: floating wiring/table logits and optimizer state are
  allowed.  They are excluded from the hardened artifact.

## Required measurements

For each block and final model record soft and hard loss/accuracy, their gaps,
time, epoch to 90% of the run's best validation hard accuracy, table fitting
error, address coverage, gate/table family distribution, inactive and unused
ratios, logical gate count, learned depth, selected-wiring fanout, hard payload
bits, per-plane entropy, A8-symbol entropy, unique-state ratio, and input/output
bit flip ratio.

The exported payload must contain only Boolean/integer tensors.  A standalone
strict replay of the complete validation and test splits must run under a
Torch dispatch audit with zero floating/complex tensors and agree exactly with
the hardened PyTorch model.

## Decisions

1. Truth refit is useful only if it raises mean paired validation hard accuracy
   over argmax without increasing inactive/unused gates materially.
2. Width scaling is supported only if 1280 planes improve mean validation hard
   accuracy over 640 planes under the same budget and do not merely duplicate
   state codes.
3. Information collapse is not inferred from marginal entropy alone.  A run
   must retain nontrivial per-plane entropy and joint state diversity while
   improving hard accuracy.
4. No result changes FullDiscrete's status as an integer accuracy baseline
   until the logic-native candidate passes three seeds and strict replay.
