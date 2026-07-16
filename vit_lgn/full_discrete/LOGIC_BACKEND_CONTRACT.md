# Logic-gate inference backend contract

## Scope and model level

The first implementation is a **bit-exact transaction-level reference** for
the hard inference primitives.  It is not cycle-accurate RTL and it does not
claim that PyTorch indexing has hardware-like performance.  Training keeps
FP32 shadow parameters, STE gradients, and AdamW; deployment exports only the
hardened integer payload.

The optional `logic_lut` evaluation path must avoid dense floating-point
`linear`/`matmul` for Wmag4 projections and Q/K XNOR-popcount.  Phase 1 retains a
power-of-two floating carrier at the boundary between an exact integer
accumulator and the existing activation requantizer.  Replacing that boundary
with an exponent-only requantizer is a separate, explicitly tracked phase.

## Numeric formats

- Activation code: signed A8.  The base quantizer emits `[-127, 127]`, while
  LUT-enhanced paths may emit the full two's-complement range `[-128, 127]`.
- Weight payload: an independent sign plus one or two unsigned magnitude
  nibbles.  Wmag4 uses one nibble; the current Wmag7 checkpoint uses a low nibble and
  a three-bit high nibble shifted left by four.
- Activation scale: one signed power-of-two exponent per input vector.
- Weight scale: one signed power-of-two exponent per output channel.
- Real interpretation: `code * 2**exponent`.
- Product ROM data: signed 12-bit integer, range `[-1920, 1920]`.
- Accumulation: exact signed integer addition.  The Python oracle uses int64;
  hardware may use the minimum derived width below and must not wrap.

For Wmag7 fan-in `F`, the required signed accumulator width is

`1 + ceil(log2(128 * 127 * F + 1))`.

Representative widths are 21 bits for `F=48`, 23 bits for `F=192`, 24 bits
for `F=384`, 25 bits for `F=768` or `F=1024`, and 26 bits for `F=1536`.

## Wmag4 product LUT

The canonical table is addressed by a two's-complement activation byte and an
unsigned magnitude nibble:

- high address bits: `activation_code & 0xff`, range `[0, 255]`;
- low address bits: magnitude nibble, range `[0, 15]`.

The table has `256 * 16 = 4096` signed-12 entries (6 KiB).  Entry `(a, m)` is
exactly `twos_complement_int8(a) * m`.  Weight sign is applied by conditional
two's-complement negation after the nibble products are accumulated.

A Wmag4 coefficient needs one table read.  A Wmag7 coefficient is reconstructed
exactly as `low4_product + (high3_product << 4)` and therefore needs two table
reads, a fixed shift, an add, and optional sign negation.  This preserves the
existing Wmag7 checkpoint; Wmag4 compression is allowed only when explicitly
requested and recorded in payload metadata.  A deployment may realize the
table as ROM, mux/adder logic, or an equivalent minimized Boolean network, but
it must match every entry exactly.  The independent bit-plane shift/add oracle
is the reference used to check the LUT implementation.

## Bit-packed XNOR-popcount

Q/K threshold bits are flattened in `(head_dim, qk_lane)` row-major order.
Bit zero is the first flattened element.  Words contain at most 63 payload
bits so signed int64 shifts remain portable.  The final word is zero-padded;
padding bits are excluded from the score.

For each query/key pair:

`score = valid_bit_count - popcount(packed_q XOR packed_k)`.

The score is an exact non-negative int32.  The model-level hard Top-K ABI is
score descending and then key index ascending.  QAT hard selection, ordinary
evaluation, `logic_lut` evaluation, and exported hardware must all use this
same stable ordering because XNOR-popcount scores frequently tie.

## RMS reciprocal-square-root ROM

For A8, `mean_square` is clamped to `[1, 16129]`.  ROM entry `n` is

`clip(round(32768 / sqrt(n)), 0, 65535)`.

and is interpreted as unsigned Q0.15.  The table is generated offline; hard
inference may only index the table and perform fixed-width integer arithmetic.
Address zero is reserved and maps to the clamped address-one value.
The signed A8-by-Q0.15 product is retained as a signed 23-bit integer with an
exponent delta of `-15` until activation requantization.  It must not be right
shifted to a whole integer first; doing so collapses the normalized value to a
few levels.

## Exponent-only Shift-RMS alternative

For fixed dimension `D`, Shift-RMS avoids the reciprocal table and product.
From the exact A8 sum of squares `S`, it selects

`k = sum(r=1..7, S >= D * 2**(2*r-1))`.

This is nearest logarithmic power-of-two RMS, with boundary ties upward and
`k` saturated at zero to match the Q15 path's mean-square clamp.  The input
power-of-two exponent cancels under normalization: the output pair is the
unchanged A8 code and exponent `-k`, not `input_exponent-k`.  Hardware uses an
unsigned square/add tree, seven constant shift-comparisons, and an exponent
rewrite.  `torch.ldexp` in the QAT reference is only a floating carrier.

## Rounding, saturation, and unresolved items

- Wmag4 product and accumulation are exact, with no rounding or saturation.
- V aggregation keeps the existing signed round-to-nearest integer division.
- The Phase-1 bridge from accumulator/exponents to activation requantization
  keeps the current nearest-power-of-two rule; it is not yet the final integer
  requantizer.
- Exponent overflow bounds, residual alignment, classifier output saturation,
  and whole-model integer requantization must be frozen before RTL is claimed
  bit-exact.

## Verification gates

1. Exhaust all 4096 A8-by-U4 LUT entries against mathematical integer products.
2. Compare randomized LUT-linear accumulators with int64 and bit-plane oracles.
3. Compare packed XNOR scores against direct Boolean equality for non-word-
   aligned sizes and boundary patterns.
4. Compare every RMS ROM entry with the frozen offline table generator.
5. Run a small Wmag4 model through both fake-quant hard eval and `logic_lut` eval;
   report exact/maximum differences layer by layer.
6. Fail the `logic_lut` eval test if `F.linear`, floating `matmul`, or runtime
   reciprocal square root is invoked on the hard path.
