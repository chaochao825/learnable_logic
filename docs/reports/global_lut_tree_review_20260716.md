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

Warmup is now 200 for both paired rows.  A second narrow review found that the
first launcher used a private GPU lock and could race the established 210
queues.  It now holds a pair-specific duplicate guard and the shared
`codex_lgn_vit_50k_gpuN` lock while waiting for three consecutive idle samples
and while running both rows.  The final ordered training source-set hash is
`f0e552495671556d8777b39d8bc6de056822556dfd8a81df90dfd281dde1d8c9`.

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
- Candidate evaluations record base-model and LUT-payload gradient L2 norms
  separately before the shared global clip.  This exposes whether the 18.9M
  d6/e192 table shadows suppress the base-network update.  The first queue was
  stopped before step 1 after this review finding; the new run names preserve
  that abandoned protocol manifest rather than overwriting it.
- Hard-change statistics compare reduce, context and broadcast tables against
  their exact initialized integer payload at every validation point.  Accuracy
  without hard flips can therefore be rejected as a surrogate-only effect.
- Schema v5 exports all int8 tables, validates shapes/ranges/ROM-bit counts and
  reconstructs the model strictly from a checkpoint.  It additionally freezes
  the group input requantizer, signed branch-shift rounding, content/LUT
  exponent-aligned add, enclosing three-way residual add and final A8
  requantizer.  Block, token, group and embedding dimensions are linked back to
  the main topology.  Expected mixer mode also forces one content router and
  one LUT tree per block; deleting or renaming either fails closed.  No FP32 shadow, dither or
  optimizer state enters the deployment payload.
- Hard lookup inputs now fail on signed-A8 overflow instead of silently
  clamping an invalid independent-executor transaction.

## Verified boundaries

- Hard LUT/tree scalar oracle: exact for random tables and codes.
- Hard module AST: no matmul, einsum, linear, conv2d, log2 or pow call.
- Address ABI: signed code plus 128, then two 8-bit fields concatenated.
- d12/e384: 96 ROMs per block, 6 MiB/block, 72 MiB/12 blocks.
- d12/e384 performs 49,536 ROM reads per image per block, or 594,432 across 12
  blocks.  With groups parallel and one port per active table this is 4,128
  serialized cycles/block; providing 32 independent channel ports reduces the
  transaction component to 129 cycles/block.  Channel sharing is a storage
  property, not free read bandwidth.
- These are ROM payload bytes, not a standard-cell gate count.
- Full isolated schema-v5 suite on 434: 131/131 tests passed.
- Both launchers pass `bash -n`; remote and local ordered source hashes match.

The schema now makes the exponent selection and parallel/residual merge
self-describing, but the PyTorch reference still transports exact
`integer_code * 2**exponent` values in floating tensors.  The payload contains
no floating inference state; it also explicitly declares that a packed
C++/CUDA/RTL executor is not included.  Thus this is a bit-exact transaction
specification, not yet a cycle-accurate whole-model hardware implementation.
The exponent-aligned add explicitly forbids wrap and requires
`8 + exponent_span + ceil(log2(branches))` signed bits.  Because the current
system contract has only a minimum exponent and no maximum, a fixed-width RTL
implementation still needs a maximum exponent range or a defined saturation
policy.

## Matched 1k decision

The frozen paired run has now completed:

| method | 500-step acc. | 1k acc. | parameters | GiB | sec/step at 1k |
|---|---:|---:|---:|---:|---:|
| hard attention control | 40.70% | **48.80%** | 3.57M | 3.18 | 0.165 |
| attention + A8 global LUT tree | 36.46% | 45.58% | 22.45M | 4.48 | 0.345 |

The LUT tree misses the promotion gate by 3.22 percentage points and is not
scheduled for 50k.  This is not a frozen-hard-function failure: 42,953 of
18,874,368 deployed table entries changed by step 1k.  Nor is it a global-clip
artifact: the LUT pre-clip gradient norm is 0.00372 versus 6.222 for the base
model (about 0.06%).  The added branch also doubles reference training time and
adds 1.31 GiB peak allocation.

The flip distribution is highly concentrated: blocks 0--1 account for 34,445
of 42,953 changes (80.2%), while the four deeper trees together change only
8,508 entries.  A 65,536-entry pair ROM fragments the training signal over an
enormous address space; increasing raw table capacity therefore increases
nominal expressivity without providing useful sample-efficient capacity.  Its
initial average/root-broadcast behavior also injects a global low-pass bias,
which is harmful beside the already content-selective Top-K path.

The practical conclusion is to retain hard content routing and use LUTs for
well-covered scalar functions or small factorized corrections, rather than a
full A8-by-A8 ROM at every stage/group/block.  Exact rows and hashes are in
[`docs/tables/global_lut_smoke_20260716.csv`](../tables/global_lut_smoke_20260716.csv).
