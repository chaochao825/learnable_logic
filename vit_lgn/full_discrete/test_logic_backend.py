from __future__ import annotations

import ast
from pathlib import Path
import unittest

import torch

from vit_lgn.full_discrete.logic_backend import (
    A8U4MagnitudeProductROM,
    INT64_MAX,
    PackedBoolWords,
    Q15_ONE,
    ReciprocalSqrtQ15ROM,
    linear_a8_w4_lut,
    linear_a8_w4_shiftadd_oracle,
    lut_linear_accumulator,
    lut_signed_product,
    pack_bool_words,
    shiftadd_linear_accumulator_oracle,
    split_magnitude_nibbles,
    w4_shiftadd_product,
    wmag_shiftadd_product,
    xnor_popcount_matrix,
    xnor_popcount_matrix_packed,
    xnor_popcount_word_score,
)


class ProductROMTest(unittest.TestCase):
    def test_a8_u4_rom_boundaries_and_shape(self) -> None:
        rom = A8U4MagnitudeProductROM()
        self.assertEqual(rom.shape, (256, 16))
        self.assertEqual(rom.lookup(-128, 0), 0)
        self.assertEqual(rom.lookup(-128, 15), -1920)
        self.assertEqual(rom.lookup(127, 15), 1905)
        with self.assertRaises(ValueError):
            rom.lookup(1, 16)

    def test_physical_payload_uses_raw_twos_complement_addresses(self) -> None:
        rom = A8U4MagnitudeProductROM()
        expected = {0x00: 0, 0x7F: 127, 0x80: -128, 0xFF: -1}
        for raw_address, signed_value in expected.items():
            self.assertEqual(rom.lookup_address(raw_address, 15), signed_value * 15)
            self.assertEqual(rom.payload[raw_address][15], signed_value * 15)

    def test_every_rom_entry_matches_positive_shiftadd_oracle(self) -> None:
        rom = A8U4MagnitudeProductROM()
        for activation in range(-128, 128):
            for magnitude in range(16):
                self.assertEqual(
                    rom.lookup(activation, magnitude),
                    w4_shiftadd_product(activation, magnitude),
                )

    def test_w7_uses_low4_plus_shifted_high3_and_independent_sign(self) -> None:
        rom = A8U4MagnitudeProductROM()
        self.assertEqual(split_magnitude_nibbles(127, 7), (15, 7))
        unsigned = rom.lookup(-128, 15) + (rom.lookup(-128, 7) << 4)
        self.assertEqual(unsigned, -16256)
        self.assertEqual(lut_signed_product(-128, -127, magnitude_bits=7), 16256)
        self.assertEqual(
            lut_signed_product(-128, -127, magnitude_bits=7),
            wmag_shiftadd_product(-128, -127, magnitude_bits=7),
        )
        self.assertEqual(len(split_magnitude_nibbles(15, 4)), 1)

    def test_all_signed_w7_products_match_shiftadd(self) -> None:
        for activation in range(-128, 128):
            for weight in range(-127, 128):
                self.assertEqual(
                    lut_signed_product(activation, weight, magnitude_bits=7),
                    wmag_shiftadd_product(activation, weight, magnitude_bits=7),
                )

    def test_lookup_tensor_requires_integer_codes_in_range(self) -> None:
        rom = A8U4MagnitudeProductROM()
        activation = torch.tensor([[-128, 0, 127]], dtype=torch.int16)
        magnitude = torch.tensor([[15, 15, 0]], dtype=torch.int8)
        expected = torch.tensor([[-1920, 0, 0]], dtype=torch.int64)
        torch.testing.assert_close(rom.lookup_tensor(activation, magnitude), expected)
        with self.assertRaises(TypeError):
            rom.lookup_tensor(activation.float(), magnitude)
        with self.assertRaises(ValueError):
            rom.lookup_tensor(torch.tensor([128]), torch.tensor([0]))


class BlockedLinearTest(unittest.TestCase):
    @staticmethod
    def _scalar_oracle(
        inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        output = []
        for input_row in inputs.reshape(-1, inputs.shape[-1]).tolist():
            row = []
            for output_index, weight_row in enumerate(weights.tolist()):
                accumulator = int(bias[output_index])
                for activation, weight in zip(input_row, weight_row):
                    accumulator += int(activation) * int(weight)
                row.append(accumulator)
            output.append(row)
        return torch.tensor(output, dtype=torch.int64).reshape(
            inputs.shape[:-1] + (weights.shape[0],)
        )

    def test_random_lut_shiftadd_and_scalar_paths_are_identical(self) -> None:
        generator = torch.Generator().manual_seed(20260713)
        inputs = torch.randint(
            -128, 128, (2, 3, 19), generator=generator, dtype=torch.int16
        )
        weights = torch.randint(
            -15, 16, (7, 19), generator=generator, dtype=torch.int8
        )
        bias = torch.randint(
            -1000, 1001, (7,), generator=generator, dtype=torch.int64
        )
        expected = self._scalar_oracle(inputs, weights, bias)
        lut = linear_a8_w4_lut(
            inputs, weights, bias, input_block=5, output_block=3
        )
        shiftadd = linear_a8_w4_shiftadd_oracle(
            inputs, weights, bias, input_block=4, output_block=2
        )
        torch.testing.assert_close(lut, expected)
        torch.testing.assert_close(shiftadd, expected)

    def test_block_sizes_do_not_change_boundary_result(self) -> None:
        inputs = torch.tensor(
            [[-128, 127, -128, 127], [127, -128, 127, -128]],
            dtype=torch.int16,
        )
        weights = torch.tensor(
            [[-15, 15, 15, -15], [15, 15, -15, -15]], dtype=torch.int8
        )
        first = linear_a8_w4_lut(
            inputs, weights, input_block=1, output_block=1
        )
        second = linear_a8_w4_lut(
            inputs, weights, input_block=99, output_block=99
        )
        torch.testing.assert_close(first, second)
        torch.testing.assert_close(
            first, self._scalar_oracle(inputs, weights, torch.zeros(2, dtype=torch.int64))
        )

    def test_generic_w1_w4_w7_w8_lut_and_shiftadd_match(self) -> None:
        generator = torch.Generator().manual_seed(88)
        inputs = torch.randint(-128, 128, (3, 17), generator=generator, dtype=torch.int16)
        for magnitude_bits in (1, 4, 7, 8):
            qmax = (1 << magnitude_bits) - 1
            weights = torch.randint(
                -qmax,
                qmax + 1,
                (6, 17),
                generator=generator,
                dtype=torch.int16,
            )
            lut = lut_linear_accumulator(
                inputs, weights, magnitude_bits=magnitude_bits, input_block=5
            )
            shiftadd = shiftadd_linear_accumulator_oracle(
                inputs, weights, magnitude_bits=magnitude_bits, output_block=2
            )
            torch.testing.assert_close(lut, shiftadd)

    def test_int64_overflow_is_explicit(self) -> None:
        inputs = torch.tensor([[1]], dtype=torch.int8)
        weights = torch.tensor([[1]], dtype=torch.int8)
        bias = torch.tensor([INT64_MAX], dtype=torch.int64)
        with self.assertRaises(OverflowError):
            linear_a8_w4_lut(inputs, weights, bias)
        with self.assertRaises(OverflowError):
            linear_a8_w4_shiftadd_oracle(inputs, weights, bias)

    def test_largest_current_fanin_w7_accumulator(self) -> None:
        inputs = torch.full((1, 1536), 127, dtype=torch.int16)
        weights = torch.full((1, 1536), 127, dtype=torch.int16)
        expected = torch.tensor([[24_774_144]], dtype=torch.int64)
        torch.testing.assert_close(
            lut_linear_accumulator(inputs, weights, magnitude_bits=7), expected
        )
        torch.testing.assert_close(
            shiftadd_linear_accumulator_oracle(
                inputs, weights, magnitude_bits=7
            ),
            expected,
        )


class PackedXNORTest(unittest.TestCase):
    def test_lsb_first_packing_at_63_bit_boundary(self) -> None:
        bits = torch.zeros(1, 65, dtype=torch.bool)
        bits[0, 0] = True
        bits[0, 62] = True
        bits[0, 63] = True
        bits[0, 64] = False
        packed = pack_bool_words(bits)
        self.assertEqual(packed.valid_bits, 65)
        self.assertEqual(packed.word_bits, 63)
        self.assertEqual(packed.words.shape, (1, 2))
        self.assertEqual(int(packed.words[0, 0]), (1 << 62) | 1)
        self.assertEqual(int(packed.words[0, 1]), 1)

    def test_random_batched_scores_match_boolean_oracle(self) -> None:
        generator = torch.Generator().manual_seed(19)
        for bit_count in (1, 7, 62, 63, 64, 126, 127, 224, 448):
            query = torch.randint(
                0, 2, (2, 3, 5, bit_count), generator=generator, dtype=torch.bool
            )
            key = torch.randint(
                0, 2, (2, 3, 7, bit_count), generator=generator, dtype=torch.bool
            )
            actual = xnor_popcount_matrix(query, key)
            expected = (query.unsqueeze(-2) == key.unsqueeze(-3)).sum(dim=-1)
            torch.testing.assert_close(actual, expected)

    def test_padding_bits_are_masked(self) -> None:
        query = PackedBoolWords(
            words=torch.tensor([[0b11111111]], dtype=torch.int64),
            valid_bits=5,
            word_bits=8,
        )
        key = PackedBoolWords(
            words=torch.tensor([[0b00011111]], dtype=torch.int64),
            valid_bits=5,
            word_bits=8,
        )
        score = xnor_popcount_matrix_packed(query, key)
        self.assertEqual(int(score[0, 0]), 5)
        self.assertEqual(
            xnor_popcount_word_score(
                [0b11111111], [0b00011111], valid_bits=5, word_bits=8
            ),
            5,
        )

    def test_pack_contract_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            pack_bool_words(torch.ones(3, 4, dtype=torch.int8))
        with self.assertRaises(ValueError):
            pack_bool_words(torch.ones(3, 4, dtype=torch.bool), word_bits=64)
        with self.assertRaises(ValueError):
            PackedBoolWords(torch.tensor([[-1]], dtype=torch.int64), 1)


class ReciprocalSqrtROMTest(unittest.TestCase):
    def test_known_q15_values_and_zero_policy(self) -> None:
        rom = ReciprocalSqrtQ15ROM(maximum_index=1 << 14)
        self.assertEqual(rom.lookup(0), 32768)
        self.assertEqual(rom.lookup(1), 32768)
        self.assertEqual(rom.lookup(2), 23170)
        self.assertEqual(rom.lookup(4), 16384)
        self.assertEqual(rom.lookup(16), 8192)
        self.assertEqual(rom.lookup(1 << 14), 256)
        self.assertEqual(rom.deployment_contract()["rounding"], "nearest_even")

    def test_every_entry_satisfies_exact_nearest_midpoints(self) -> None:
        rom = ReciprocalSqrtQ15ROM(maximum_index=1 << 14)
        four_scaled_square = 4 * Q15_ONE * Q15_ONE
        for index in range(1, rom.maximum_index + 1):
            value = rom.lookup(index)
            lower_midpoint = 2 * value - 1
            upper_midpoint = 2 * value + 1
            lower_comparison = four_scaled_square - index * lower_midpoint**2
            upper_comparison = four_scaled_square - index * upper_midpoint**2
            self.assertGreaterEqual(lower_comparison, 0)
            self.assertLessEqual(upper_comparison, 0)
            if lower_comparison == 0 or upper_comparison == 0:
                self.assertEqual(value & 1, 0)

    def test_floor_mode_and_tensor_lookup(self) -> None:
        rom = ReciprocalSqrtQ15ROM(maximum_index=16, rounding="floor", zero_value=0)
        indices = torch.tensor([[0, 1, 2, 4, 16]], dtype=torch.int16)
        expected = torch.tensor([[0, 32768, 23170, 16384, 8192]], dtype=torch.int64)
        torch.testing.assert_close(rom.lookup_tensor(indices), expected)
        with self.assertRaises(ValueError):
            rom.lookup(17)


class ForbiddenFloatPrimitiveTest(unittest.TestCase):
    def test_backend_has_no_float_matrix_or_transcendental_calls(self) -> None:
        source_path = Path(__file__).with_name("logic_backend.py")
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden_attributes = {
            "linear",
            "matmul",
            "mm",
            "bmm",
            "addmm",
            "einsum",
            "rsqrt",
            "sqrt",
            "log2",
            "pow",
        }
        observed = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(observed & forbidden_attributes)
        self.assertFalse(any(isinstance(node, ast.MatMult) for node in ast.walk(tree)))
        self.assertFalse(any(isinstance(node, ast.Div) for node in ast.walk(tree)))
        self.assertFalse(
            any(
                isinstance(node, ast.Constant) and isinstance(node.value, float)
                for node in ast.walk(tree)
            )
        )
        forbidden_builtins = {"float", "complex", "pow"}
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in forbidden_builtins
                for node in ast.walk(tree)
            )
        )


if __name__ == "__main__":
    unittest.main()
