"""Bit-exact transaction-level integer primitives for a logic backend.

This module is deliberately a slow CPU reference, not a training kernel.  It
defines the arithmetic that an RTL/bit-packed implementation must reproduce:

* A8 is two's-complement ``[-128, 127]``.
* The physical product ROM is signed A8 by an unsigned U4 magnitude nibble
  ``[0, 15]``; the weight sign is carried and applied independently.
* Repository weights use one explicit sign plus Wmag1--Wmag8 magnitude planes.
  A magnitude is decomposed into four-bit chunks, so Wmag7 is exactly
  ``low4 + (high3 << 4)`` and Wmag4 needs one ROM access.
* Linear products are chunked ROM lookups and accumulation is checked int64.
  Integer bias, when supplied, is in the same accumulator units.
* Packed Boolean words use LSB-first ordering and at most 63 payload bits, so
  every word remains a non-negative signed-int64 value.
* Reciprocal-square-root entries are unsigned Q0.15 payloads.  Generation uses
  integer arithmetic and round-to-nearest, ties-to-even; index zero aliases
  index one by default.  (The value 1.0 is encoded as ``0x8000``.)

No cycle timing, reset, or valid protocol is modeled here. Scale/exponent
alignment and output requantization are modeled by ``integer_executor.py``;
fixed-width cycle control remains a later RTL concern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

import torch


A8_MIN = -128
A8_MAX = 127
INT64_MIN = -(1 << 63)
INT64_MAX = (1 << 63) - 1
Q15_ONE = 1 << 15
Q15_MAX = (1 << 16) - 1

RoundingMode = Literal["nearest_even", "floor"]

_INTEGER_DTYPES = {
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_BYTE_POPCOUNT = tuple(bin(value).count("1") for value in range(256))


def _magnitude_qmax(magnitude_bits: int) -> int:
    if magnitude_bits < 1 or magnitude_bits > 8:
        raise ValueError("magnitude_bits must be in [1,8]")
    return (1 << magnitude_bits) - 1


def _as_cpu_integer_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for this transaction reference")
    if value.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")
    return value


def _check_tensor_range(name: str, value: torch.Tensor, lower: int, upper: int) -> None:
    if value.numel() == 0:
        return
    minimum = int(value.min().item())
    maximum = int(value.max().item())
    if minimum < lower or maximum > upper:
        raise ValueError(
            f"{name} values must be in [{lower},{upper}], got [{minimum},{maximum}]"
        )


def _checked_int64(value: int, context: str) -> int:
    if value < INT64_MIN or value > INT64_MAX:
        raise OverflowError(f"signed int64 overflow while {context}: {value}")
    return value


class A8U4MagnitudeProductROM:
    """Two-dimensional signed-A8 by unsigned-U4-magnitude product ROM.

    Runtime ``lookup`` performs address translation and table selection only.
    The immutable table is generated once by the Python reference constructor;
    an exporter can serialize the same payload as combinational or synchronous
    ROM.  Its invariant shape is ``(256, 16)``; sign and chunk position are
    separate gates/wiring rather than extra ROM address dimensions.
    """

    def __init__(self) -> None:
        self._table = tuple(
            tuple(
                (raw_address if raw_address < 128 else raw_address - 256)
                * magnitude_nibble
                for magnitude_nibble in range(16)
            )
            for raw_address in range(256)
        )

    @property
    def shape(self) -> tuple[int, int]:
        return len(self._table), len(self._table[0])

    @property
    def payload(self) -> tuple[tuple[int, ...], ...]:
        return self._table

    def lookup(self, activation: int, magnitude_nibble: int) -> int:
        activation = int(activation)
        magnitude_nibble = int(magnitude_nibble)
        if activation < A8_MIN or activation > A8_MAX:
            raise ValueError(f"A8 code must be in [{A8_MIN},{A8_MAX}]")
        if magnitude_nibble < 0 or magnitude_nibble > 15:
            raise ValueError("U4 magnitude nibble must be in [0,15]")
        return self._table[activation & 0xFF][magnitude_nibble]

    def lookup_address(self, raw_address: int, magnitude_nibble: int) -> int:
        """Read a physical ROM address using raw A8 two's-complement bits."""

        raw_address = int(raw_address)
        magnitude_nibble = int(magnitude_nibble)
        if raw_address < 0 or raw_address > 255:
            raise ValueError("raw A8 address must be in [0,255]")
        if magnitude_nibble < 0 or magnitude_nibble > 15:
            raise ValueError("U4 magnitude nibble must be in [0,15]")
        return self._table[raw_address][magnitude_nibble]

    def lookup_tensor(
        self, activation: torch.Tensor, magnitude_nibble: torch.Tensor
    ) -> torch.Tensor:
        activation = _as_cpu_integer_tensor("activation", activation)
        magnitude_nibble = _as_cpu_integer_tensor(
            "magnitude_nibble", magnitude_nibble
        )
        if activation.shape != magnitude_nibble.shape:
            raise ValueError(
                "activation and magnitude_nibble tensors must have identical shapes"
            )
        _check_tensor_range("activation", activation, A8_MIN, A8_MAX)
        _check_tensor_range("magnitude_nibble", magnitude_nibble, 0, 15)
        flat_activation = activation.reshape(-1).tolist()
        flat_magnitude = magnitude_nibble.reshape(-1).tolist()
        values = [
            self.lookup(a_code, nibble)
            for a_code, nibble in zip(flat_activation, flat_magnitude)
        ]
        return torch.tensor(values, dtype=torch.int64).reshape(activation.shape)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "activation_address": "uint8 two's-complement view of signed A8",
            "magnitude_address": "unsigned U4 [0,15]",
            "entries": 256 * 16,
            "payload": "signed 12-bit product",
            "weight_sign": "independent conditional negate after chunk reconstruction",
        }


_DEFAULT_PRODUCT_ROM = A8U4MagnitudeProductROM()


def split_magnitude_nibbles(magnitude: int, magnitude_bits: int) -> tuple[int, ...]:
    """Return low-to-high U4 chunks for a W1--W8 unsigned magnitude."""

    qmax = _magnitude_qmax(magnitude_bits)
    magnitude = int(magnitude)
    if magnitude < 0 or magnitude > qmax:
        raise ValueError(f"magnitude must be in [0,{qmax}]")
    chunk_count = (magnitude_bits + 3) // 4
    return tuple((magnitude >> (4 * index)) & 0xF for index in range(chunk_count))


def lut_signed_product(
    activation: int,
    weight: int,
    *,
    magnitude_bits: int = 4,
    rom: A8U4MagnitudeProductROM | None = None,
) -> int:
    """Reference one A8/signed-Wmag product using U4 ROM chunks."""

    activation = int(activation)
    weight = int(weight)
    qmax = _magnitude_qmax(magnitude_bits)
    if activation < A8_MIN or activation > A8_MAX:
        raise ValueError(f"A8 code must be in [{A8_MIN},{A8_MAX}]")
    if weight < -qmax or weight > qmax:
        raise ValueError(f"signed W{magnitude_bits} code must be in [{-qmax},{qmax}]")
    if rom is None:
        rom = _DEFAULT_PRODUCT_ROM

    magnitude = -weight if weight < 0 else weight
    unsigned_product = 0
    for chunk_index, nibble in enumerate(
        split_magnitude_nibbles(magnitude, magnitude_bits)
    ):
        unsigned_product += rom.lookup(activation, nibble) << (4 * chunk_index)
    return -unsigned_product if weight < 0 else unsigned_product


def wmag_shiftadd_product(
    activation: int,
    weight: int,
    *,
    magnitude_bits: int = 4,
) -> int:
    """Reference one A8/signed-Wmag product as conditional shifts and adds."""

    activation = int(activation)
    weight = int(weight)
    qmax = _magnitude_qmax(magnitude_bits)
    if activation < A8_MIN or activation > A8_MAX:
        raise ValueError(f"A8 code must be in [{A8_MIN},{A8_MAX}]")
    if weight < -qmax or weight > qmax:
        raise ValueError(f"signed W{magnitude_bits} code must be in [{-qmax},{qmax}]")

    magnitude = -weight if weight < 0 else weight
    unsigned_product = 0
    for bit_index in range(magnitude_bits):
        if magnitude & (1 << bit_index):
            unsigned_product += activation << bit_index
    return -unsigned_product if weight < 0 else unsigned_product


def w4_shiftadd_product(activation: int, weight: int) -> int:
    """Compatibility wrapper for a four-magnitude-plane (Wmag4) weight."""

    return wmag_shiftadd_product(activation, weight, magnitude_bits=4)


def _prepare_linear_inputs(
    input_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    bias: torch.Tensor | None,
    magnitude_bits: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, ...]]:
    input_codes = _as_cpu_integer_tensor("input_codes", input_codes)
    weight_codes = _as_cpu_integer_tensor("weight_codes", weight_codes)
    if input_codes.ndim < 1:
        raise ValueError("input_codes must have at least one dimension")
    if weight_codes.ndim != 2:
        raise ValueError("weight_codes must have shape [out_features,in_features]")
    if input_codes.shape[-1] != weight_codes.shape[1]:
        raise ValueError("input and weight in_features do not match")
    _check_tensor_range("input_codes", input_codes, A8_MIN, A8_MAX)
    qmax = _magnitude_qmax(magnitude_bits)
    _check_tensor_range("weight_codes", weight_codes, -qmax, qmax)

    out_features = int(weight_codes.shape[0])
    if bias is None:
        bias_values = [0] * out_features
    else:
        bias = _as_cpu_integer_tensor("bias", bias)
        if bias.shape != (out_features,):
            raise ValueError("bias must have shape [out_features]")
        bias_values = [
            _checked_int64(int(value), "loading bias") for value in bias.tolist()
        ]

    leading_shape = tuple(int(size) for size in input_codes.shape[:-1])
    flat_input = input_codes.reshape(-1, input_codes.shape[-1])
    return flat_input, weight_codes, bias_values, leading_shape


def lut_linear_accumulator(
    input_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    magnitude_bits: int = 4,
    rom: A8U4MagnitudeProductROM | None = None,
    input_block: int = 64,
    output_block: int = 32,
) -> torch.Tensor:
    """Generic W1--W8 blocked linear accumulator using U4 product-ROM reads.

    The output shape is ``input_codes.shape[:-1] + (out_features,)`` and its
    dtype is signed int64.  Overflow raises instead of silently wrapping or
    saturating.  Blocking changes traversal/storage only, never arithmetic.
    """

    if input_block < 1 or output_block < 1:
        raise ValueError("input_block and output_block must be positive")
    if rom is None:
        rom = _DEFAULT_PRODUCT_ROM
    flat_input, weight_codes, bias_values, leading_shape = _prepare_linear_inputs(
        input_codes, weight_codes, bias, magnitude_bits
    )
    input_rows = flat_input.tolist()
    weight_rows = weight_codes.tolist()
    in_features = int(weight_codes.shape[1])
    out_features = int(weight_codes.shape[0])
    output_rows: list[list[int]] = []

    for input_row in input_rows:
        output_row = list(bias_values)
        for output_start in range(0, out_features, output_block):
            output_stop = min(output_start + output_block, out_features)
            for input_start in range(0, in_features, input_block):
                input_stop = min(input_start + input_block, in_features)
                for output_index in range(output_start, output_stop):
                    accumulator = output_row[output_index]
                    weight_row = weight_rows[output_index]
                    for input_index in range(input_start, input_stop):
                        product = lut_signed_product(
                            input_row[input_index],
                            weight_row[input_index],
                            magnitude_bits=magnitude_bits,
                            rom=rom,
                        )
                        accumulator = _checked_int64(
                            accumulator + product,
                            f"accumulating output lane {output_index}",
                        )
                    output_row[output_index] = accumulator
        output_rows.append(output_row)

    output = torch.tensor(output_rows, dtype=torch.int64)
    return output.reshape(leading_shape + (out_features,))


def shiftadd_linear_accumulator_oracle(
    input_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    magnitude_bits: int = 4,
    input_block: int = 64,
    output_block: int = 32,
) -> torch.Tensor:
    """Blocked linear oracle using four conditional shifted copies per weight."""

    if input_block < 1 or output_block < 1:
        raise ValueError("input_block and output_block must be positive")
    flat_input, weight_codes, bias_values, leading_shape = _prepare_linear_inputs(
        input_codes, weight_codes, bias, magnitude_bits
    )
    input_rows = flat_input.tolist()
    weight_rows = weight_codes.tolist()
    in_features = int(weight_codes.shape[1])
    out_features = int(weight_codes.shape[0])
    output_rows: list[list[int]] = []

    for input_row in input_rows:
        output_row = list(bias_values)
        for output_start in range(0, out_features, output_block):
            output_stop = min(output_start + output_block, out_features)
            for input_start in range(0, in_features, input_block):
                input_stop = min(input_start + input_block, in_features)
                for output_index in range(output_start, output_stop):
                    accumulator = output_row[output_index]
                    weight_row = weight_rows[output_index]
                    for input_index in range(input_start, input_stop):
                        product = wmag_shiftadd_product(
                            input_row[input_index],
                            weight_row[input_index],
                            magnitude_bits=magnitude_bits,
                        )
                        accumulator = _checked_int64(
                            accumulator + product,
                            f"accumulating output lane {output_index}",
                        )
                    output_row[output_index] = accumulator
        output_rows.append(output_row)

    output = torch.tensor(output_rows, dtype=torch.int64)
    return output.reshape(leading_shape + (out_features,))


def linear_a8_w4_lut(
    input_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    rom: A8U4MagnitudeProductROM | None = None,
    input_block: int = 64,
    output_block: int = 32,
) -> torch.Tensor:
    """Compatibility wrapper: one U4 magnitude chunk plus independent sign."""

    return lut_linear_accumulator(
        input_codes,
        weight_codes,
        bias,
        magnitude_bits=4,
        rom=rom,
        input_block=input_block,
        output_block=output_block,
    )


def linear_a8_w4_shiftadd_oracle(
    input_codes: torch.Tensor,
    weight_codes: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    input_block: int = 64,
    output_block: int = 32,
) -> torch.Tensor:
    """Compatibility wrapper for the four-plane shift/add oracle."""

    return shiftadd_linear_accumulator_oracle(
        input_codes,
        weight_codes,
        bias,
        magnitude_bits=4,
        input_block=input_block,
        output_block=output_block,
    )


@dataclass(frozen=True)
class PackedBoolWords:
    """LSB-first Boolean payload packed in non-negative signed-int64 words."""

    words: torch.Tensor
    valid_bits: int
    word_bits: int = 63

    def __post_init__(self) -> None:
        words = _as_cpu_integer_tensor("words", self.words)
        if words.dtype != torch.int64:
            raise TypeError("packed words must have dtype torch.int64")
        if self.valid_bits < 1:
            raise ValueError("valid_bits must be positive")
        if self.word_bits < 1 or self.word_bits > 63:
            raise ValueError("word_bits must be in [1,63]")
        expected_words = (self.valid_bits + self.word_bits - 1) // self.word_bits
        if words.ndim < 1 or words.shape[-1] != expected_words:
            raise ValueError("last packed-word dimension does not match valid_bits")
        if words.numel() and int(words.min().item()) < 0:
            raise ValueError("packed words must be non-negative")
        maximum_word = (1 << self.word_bits) - 1
        if words.numel() and int(words.max().item()) > maximum_word:
            raise ValueError("packed word exceeds configured word_bits")


def pack_bool_words(bits: torch.Tensor, *, word_bits: int = 63) -> PackedBoolWords:
    """Pack the last Boolean dimension into LSB-first signed-int64 words."""

    if not isinstance(bits, torch.Tensor):
        raise TypeError("bits must be a torch.Tensor")
    if bits.device.type != "cpu":
        raise ValueError("bits must be on CPU for this transaction reference")
    if bits.dtype != torch.bool:
        raise TypeError("bits must have dtype torch.bool")
    if bits.ndim < 1 or bits.shape[-1] < 1:
        raise ValueError("bits must have a non-empty final dimension")
    if word_bits < 1 or word_bits > 63:
        raise ValueError("word_bits must be in [1,63]")

    valid_bits = int(bits.shape[-1])
    word_count = (valid_bits + word_bits - 1) // word_bits
    flat_bits = bits.reshape(-1, valid_bits).tolist()
    packed_rows: list[list[int]] = []
    for bit_row in flat_bits:
        word_row: list[int] = []
        for word_index in range(word_count):
            start = word_index * word_bits
            stop = min(start + word_bits, valid_bits)
            word = 0
            for bit_index in range(start, stop):
                if bit_row[bit_index]:
                    word |= 1 << (bit_index - start)
            word_row.append(word)
        packed_rows.append(word_row)
    words = torch.tensor(packed_rows, dtype=torch.int64).reshape(
        bits.shape[:-1] + (word_count,)
    )
    return PackedBoolWords(words=words, valid_bits=valid_bits, word_bits=word_bits)


def _byte_lut_popcount(value: int) -> int:
    if value < 0:
        raise ValueError("popcount input must be non-negative")
    count = 0
    while value:
        count += _BYTE_POPCOUNT[value & 0xFF]
        value >>= 8
    return count


def xnor_popcount_word_score(
    query_words: list[int] | tuple[int, ...],
    key_words: list[int] | tuple[int, ...],
    *,
    valid_bits: int,
    word_bits: int = 63,
) -> int:
    """XNOR-popcount one packed vector pair using an 8-bit popcount LUT."""

    if valid_bits < 1:
        raise ValueError("valid_bits must be positive")
    if word_bits < 1 or word_bits > 63:
        raise ValueError("word_bits must be in [1,63]")
    expected_words = (valid_bits + word_bits - 1) // word_bits
    if len(query_words) != expected_words or len(key_words) != expected_words:
        raise ValueError("packed word count does not match valid_bits")

    score = 0
    for word_index, (query_word, key_word) in enumerate(zip(query_words, key_words)):
        remaining = valid_bits - word_index * word_bits
        payload_bits = min(word_bits, remaining)
        mask = (1 << payload_bits) - 1
        query_word = int(query_word)
        key_word = int(key_word)
        if query_word < 0 or key_word < 0:
            raise ValueError("packed words must be non-negative")
        xnor_word = ~(query_word ^ key_word) & mask
        score += _byte_lut_popcount(xnor_word)
    return score


def xnor_popcount_matrix_packed(
    query: PackedBoolWords, key: PackedBoolWords
) -> torch.Tensor:
    """Score ``[...,queries,bits]`` against ``[...,keys,bits]`` payloads."""

    if query.valid_bits != key.valid_bits or query.word_bits != key.word_bits:
        raise ValueError("query and key packing contracts must match")
    if query.words.ndim < 2 or key.words.ndim < 2:
        raise ValueError("packed tensors must include item and word dimensions")
    if query.words.shape[:-2] != key.words.shape[:-2]:
        raise ValueError("query and key batch dimensions must match")

    batch_shape = tuple(int(size) for size in query.words.shape[:-2])
    query_count = int(query.words.shape[-2])
    key_count = int(key.words.shape[-2])
    word_count = int(query.words.shape[-1])
    query_batches = query.words.reshape(-1, query_count, word_count).tolist()
    key_batches = key.words.reshape(-1, key_count, word_count).tolist()
    score_batches: list[list[list[int]]] = []
    for query_batch, key_batch in zip(query_batches, key_batches):
        score_rows: list[list[int]] = []
        for query_words in query_batch:
            score_rows.append(
                [
                    xnor_popcount_word_score(
                        query_words,
                        key_words,
                        valid_bits=query.valid_bits,
                        word_bits=query.word_bits,
                    )
                    for key_words in key_batch
                ]
            )
        score_batches.append(score_rows)
    scores = torch.tensor(score_batches, dtype=torch.int64)
    return scores.reshape(batch_shape + (query_count, key_count))


def xnor_popcount_matrix(
    query_bits: torch.Tensor, key_bits: torch.Tensor, *, word_bits: int = 63
) -> torch.Tensor:
    """Pack Boolean Q/K vectors and return integer XNOR-popcount scores."""

    if query_bits.ndim < 2 or key_bits.ndim < 2:
        raise ValueError("query_bits and key_bits need item and bit dimensions")
    if query_bits.shape[-1] != key_bits.shape[-1]:
        raise ValueError("query and key bit dimensions must match")
    return xnor_popcount_matrix_packed(
        pack_bool_words(query_bits, word_bits=word_bits),
        pack_bool_words(key_bits, word_bits=word_bits),
    )


def _rsqrt_q15_entry(index: int, rounding: RoundingMode) -> int:
    if index <= 0:
        raise ValueError("reciprocal-square-root index must be positive")
    scaled_square = Q15_ONE * Q15_ONE
    floor_value = math.isqrt(scaled_square // index)
    if rounding == "floor":
        return min(floor_value, Q15_MAX)
    if rounding != "nearest_even":
        raise ValueError(f"unsupported rounding mode: {rounding!r}")

    # Compare S/sqrt(index) with floor+1/2 without floating point:
    #   4*S^2 ? index*(2*floor+1)^2.
    midpoint = 2 * floor_value + 1
    comparison = 4 * scaled_square - index * midpoint * midpoint
    increment = comparison > 0 or (comparison == 0 and floor_value & 1)
    rounded = floor_value + int(increment)
    return min(rounded, Q15_MAX)


@dataclass(frozen=True)
class ReciprocalSqrtQ15ROM:
    """Integer-generated, direct-index Q0.15 reciprocal-square-root ROM."""

    maximum_index: int = 127 * 127
    rounding: RoundingMode = "nearest_even"
    zero_value: int = Q15_ONE

    def __post_init__(self) -> None:
        if self.maximum_index < 1:
            raise ValueError("maximum_index must be positive")
        if self.rounding not in ("nearest_even", "floor"):
            raise ValueError(f"unsupported rounding mode: {self.rounding!r}")
        if self.zero_value < 0 or self.zero_value > Q15_MAX:
            raise ValueError(f"zero_value must be in [0,{Q15_MAX}]")
        entries = (self.zero_value,) + tuple(
            _rsqrt_q15_entry(index, self.rounding)
            for index in range(1, self.maximum_index + 1)
        )
        object.__setattr__(self, "entries", entries)

    entries: tuple[int, ...] = field(init=False, repr=False)

    def lookup(self, index: int) -> int:
        index = int(index)
        if index < 0 or index > self.maximum_index:
            raise ValueError(f"ROM index must be in [0,{self.maximum_index}]")
        return self.entries[index]

    def lookup_tensor(self, index: torch.Tensor) -> torch.Tensor:
        index = _as_cpu_integer_tensor("index", index)
        _check_tensor_range("index", index, 0, self.maximum_index)
        values = [self.entries[int(value)] for value in index.reshape(-1).tolist()]
        return torch.tensor(values, dtype=torch.int64).reshape(index.shape)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "address_range": [0, self.maximum_index],
            "payload_bits": 16,
            "payload_format": "unsigned Q0.15 (1.0 == 0x8000)",
            "rounding": self.rounding,
            "zero_policy": f"return configured payload 0x{self.zero_value:04x}",
            "generation_arithmetic": "integer isqrt plus exact midpoint comparison",
        }
