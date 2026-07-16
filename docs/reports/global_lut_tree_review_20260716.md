# Nonlinear A8 global LUT-tree review

Date: 2026-07-16

## Reviewed claim

The module is a content-dependent global branch with a deployable hard value,
not merely a dense neural layer described as a LUT.  For each block and
32-channel group it uses six A8-by-A8 reduction ROMs over 64 patches, one
root/CLS ROM, and one token/context broadcast ROM.  Tables are shared over
spatial nodes and channels within a group; stages, groups and blocks have
independent payloads.

## Independent and fallback review

An independent Codex reviewer inspected the launcher and source freeze before
rate-limit exhaustion.  It found two launch blockers:

1. the 1k launcher had `warmup_steps=2000`, rejected by the training protocol;
2. its expected aggregate source hash predated later source edits.

Warmup is now 200 for both paired rows, and the final ordered source-set hash is
`71fec7ad6ad78acad28e456b143151c47e84aff59c3ff22f032f6010a8df3b0b`.
The reviewer then terminated with HTTP 429, so the remaining hard-path review
used the documented local fallback and an isolated 434 test tree.

## Adversarial findings fixed

- Direct integer-centred shadows could repeat the earlier logic-tree failure:
  learning might never cross a hard rounding boundary.  The table shadows now
  use deterministic +/-0.49 sub-threshold offsets.  Those offsets are removed
  from the bilinear surrogate, so initial soft functions remain rounded
  average/projection-B while the hard payload is unchanged.  Unit tests prove a
  0.02 update can create a counted hard flip.
- AdamW decay would move millions of unvisited truth-table entries toward zero
  and create spurious flips.  LUT payload parameters now use a zero-decay
  optimizer group; a partition test proves every trainable parameter appears
  exactly once.
- Hard-change statistics compare reduce, context and broadcast tables against
  their exact initialized integer payload at every validation point.  Accuracy
  without hard flips can therefore be rejected as a surrogate-only effect.
- Schema v3 exports all int8 tables, validates shapes/ranges/ROM-bit counts and
  reconstructs the model strictly from a checkpoint.  No FP32 shadow, dither or
  optimizer state enters the deployment payload.

## Verified boundaries

- Hard LUT/tree scalar oracle: exact for random tables and codes.
- Hard module AST: no matmul, einsum, linear, conv2d, log2 or pow call.
- Address ABI: signed code plus 128, then two 8-bit fields concatenated.
- d12/e384: 96 ROMs per block, 6 MiB/block, 72 MiB/12 blocks.
- These are ROM payload bytes, not a standard-cell gate count.
- Full isolated suite: 129/129 tests passed.
- Both launchers pass `bash -n`; remote and local ordered source hashes match.

The activation power-of-two exponent selector and the parallel residual merge
still use the repository's floating carrier reference.  Thus the ROM tree is a
bit-exact hard primitive inside a hard-discrete numerical model, not yet a
whole-model RTL executor.

The mechanism is not promoted to a 50k method until its matched 1k row beats
the attention-only control and shows nonzero deployed hard-table changes.
