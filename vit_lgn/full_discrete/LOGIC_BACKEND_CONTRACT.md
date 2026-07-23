# Logic-gate inference backend contract

## Scope and model level

The first implementation is a **bit-exact transaction-level reference** for
the hard inference primitives.  It is not cycle-accurate RTL and it does not
claim that PyTorch indexing has hardware-like performance.  Training keeps
FP32 shadow parameters, STE gradients, and AdamW; deployment exports only the
hardened integer payload.

The optional `logic_lut` path in the training model remains a diagnostic
floating carrier. It is not the deployment claim. Schema v6 is executed by
`StrictIntegerExecutor`, whose only public image input is CPU `uint8` and whose
only activation state is an `int64` code plus an `int32` signed exponent.
Exact exponent selection, residual alignment, requantization, RMS lookup,
attention, FFN, and classifier output are implemented without a floating or
complex tensor. `IntegerRuntimeAudit` intercepts every executed Torch operator
and rejects any violation.

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

## Audit payload versus compact hardware payload

Schema v6 deliberately carries several equivalent weight views: signed int16
codes, signs, unpacked magnitude bit planes, and U4 chunks. This makes the
independent integer-matrix, bit-plane, and ROM oracles directly checkable, but
it is not a minimal storage format. Capacity reports must therefore state both
the serialized audit-payload bits and the logical bit-packed coefficient bits.
They may not silently call either one the synthesized gate count.

A compact backend may retain exactly one equivalent Wmag representation plus
scale exponents and shared ROMs, after bit-exact equivalence is proven. Removing
redundant views is a packaging optimization and cannot be reported as an
accuracy method. Transformer block count is likewise only algorithmic depth;
total Boolean depth requires lowering popcount, Top-K, shift/add, RMS lookup,
comparison, and requantization networks. Dense fanout is measured from
nonzero hard coefficients per input, not confused with patch-projection fan-in.

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
- Schema v6 selects the nearest power-of-two scale with exact squared-boundary
  comparisons and nearest-even ties. Signed right shifts use nearest-even
  rounding; overflow is checked before exact left alignment and accumulation.
- Residuals align to the elementwise minimum exponent, add in checked int64,
  and requantize over the same last-axis grouping as the QAT reference.
- The classifier emits A16 integer codes with one output exponent per sample.
- A finite hardware exponent-range/saturation contract is still required
  before fixed-width cycle-accurate RTL is claimed. The transaction reference
  intentionally raises rather than wrapping when int64 cannot represent an
  exact intermediate.

## Verification gates

1. Exhaust all 4096 A8-by-U4 LUT entries against mathematical integer products.
2. Compare randomized LUT-linear accumulators with int64 and bit-plane oracles.
3. Compare packed XNOR scores against direct Boolean equality for non-word-
   aligned sizes and boundary patterns.
4. Compare every RMS ROM entry with the frozen offline table generator.
5. Run a small Wmag4 model through the QAT hard carrier and schema-v6 integer
   executor; require exact final dyadic logits and predictions.
6. Require the ROM-linear and integer-matrix acceleration backends to emit
   identical codes and exponents.
7. Run the complete executor under `IntegerRuntimeAudit`; fail on every
   floating/complex tensor input or output at every intercepted Torch operator.
8. Parse `integer_executor.py` and reject float literals, true division, and
   floating transcendental/activation calls.
