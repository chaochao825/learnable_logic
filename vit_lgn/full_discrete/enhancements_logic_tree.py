"""Spatially shared 3x3 Boolean logic trees for A8 token states.

The Boolean tree core in this module is deliberately small and explicit:

* signed-magnitude A8 activations are exposed as eight Boolean bitplanes;
* every tree reads eight fixed sites from a local 3x3 neighbourhood;
* seven two-input, four-entry truth-table gates form an 8->4->2->1 tree;
* truth tables are shared over all spatial positions; and
* the hard path uses only indexing, shifts, masks, and integer/Boolean tensors.

No dense connection selector, matrix multiplication, linear layer, or
convolution is part of the hard tree executor.  The surrounding PyTorch
transaction bridge still reuses the project's power-of-two activation
quantizer and floating carrier; it is not presented as the final RTL bridge.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .shiftadd import PowerOfTwoActivationQuantizer


# Truth-table order is [00, 01, 10, 11], with address (A << 1) | B.
# Therefore the packed little-address-endian nibble for the projection A is
# 0b1100 == 0xC.  It must not be confused with categorical function id 3.
TRUTH_TABLE_A: Tuple[int, int, int, int] = (0, 0, 1, 1)
TRUTH_NIBBLE_A = 0xC


# Site zero is deliberately the centre.  With every gate initialized to A,
# the root is exactly leaf zero, so this ordering gives identity-at-start in
# the module's encoded Boolean state.
SITE_OFFSETS: Tuple[Tuple[int, int], ...] = (
    (0, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


class _HardPayloadSoftGradient(torch.autograd.Function):
    """Return an exact hard payload while differentiating a soft proxy."""

    @staticmethod
    def forward(ctx, hard: Tensor, soft: Tensor) -> Tensor:  # type: ignore[override]
        del ctx, soft
        return hard

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        del ctx
        return None, grad_output


def hard_truth_table_gate(left: Tensor, right: Tensor, truth_nibble: Tensor) -> Tensor:
    """Evaluate packed two-input truth tables without arithmetic projection.

    Args:
        left/right: broadcast-compatible Boolean or 0/1 integer tensors.
        truth_nibble: broadcast-compatible integer tensor in [0, 15].

    Returns:
        A Boolean tensor.  Bit ``2*A+B`` of each nibble is the selected output.
    """

    left_u8 = left.to(torch.uint8)
    right_u8 = right.to(torch.uint8)
    nibble_u8 = truth_nibble.to(torch.uint8)
    address = torch.bitwise_or(torch.bitwise_left_shift(left_u8, 1), right_u8)
    selected = torch.bitwise_right_shift(nibble_u8, address)
    return torch.bitwise_and(selected, 1).to(torch.bool)


class SharedLogicTreeConv3x3(nn.Module):
    """A spatially shared bank of depth-three Boolean 3x3 logic trees.

    There is one tree for every channel and activation bitplane.  A tree has
    eight fixed leaves and seven trainable truth tables.  Leaf zero is always
    the centre position at the same channel and bitplane.  The remaining seven
    leaves use seven of the eight neighbours; the omitted neighbour rotates
    deterministically with channel index.  This keeps routing fixed and costs
    no learned dense connection matrix.

    The input/output carrier remains ``[B, 1 + H*W, C]`` for compatibility with
    the surrounding ViT.  Internally the root remains Boolean until the final
    signed-magnitude transaction boundary; there is no W7 output projection.
    """

    TREE_DEPTH = 3
    NUM_LEAVES = 8
    NUM_GATES = 7
    NUM_TRUTH_ENTRIES = 4

    def __init__(
        self,
        dim: int,
        grid_size: int,
        activation_bits: int = 8,
        init_strength: float = 2.0,
        truth_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if grid_size <= 0:
            raise ValueError(f"grid_size must be positive, got {grid_size}")
        if activation_bits != 8:
            raise ValueError(
                "SharedLogicTreeConv3x3 currently defines the minimal A8 ABI; "
                f"got activation_bits={activation_bits}"
            )
        if init_strength <= 0:
            raise ValueError(f"init_strength must be positive, got {init_strength}")
        if truth_temperature <= 0:
            raise ValueError(
                f"truth_temperature must be positive, got {truth_temperature}"
            )

        self.dim = int(dim)
        self.grid_size = int(grid_size)
        self.activation_bits = int(activation_bits)
        self.init_strength = float(init_strength)
        self.truth_temperature = float(truth_temperature)
        self.input_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)

        # [channel, bitplane, gate, truth entry]
        self.truth_table_logits = nn.Parameter(
            torch.empty(
                self.dim,
                self.activation_bits,
                self.NUM_GATES,
                self.NUM_TRUTH_ENTRIES,
            )
        )

        leaf_site = torch.empty(
            self.dim, self.activation_bits, self.NUM_LEAVES, dtype=torch.long
        )
        for channel in range(self.dim):
            for bitplane in range(self.activation_bits):
                omitted_neighbour = 1 + ((channel + bitplane) % 8)
                neighbours = [
                    site for site in range(1, 9) if site != omitted_neighbour
                ]
                leaf_site[channel, bitplane] = torch.tensor(
                    [0, *neighbours], dtype=torch.long
                )
        self.register_buffer("leaf_site", leaf_site, persistent=True)

        offsets = torch.tensor(SITE_OFFSETS, dtype=torch.int8)
        self.register_buffer("site_offsets", offsets, persistent=True)
        self.register_buffer(
            "_root_changed_by_bitplane",
            torch.zeros(self.activation_bits, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_root_total_by_bitplane",
            torch.zeros(self.activation_bits, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_root_code_changed", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_root_code_total", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_negative_zero_count", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "_saturated_code_count", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize every hard gate as the projection A=[0,0,1,1].

        A perfectly symmetric soft table is independent of input B, which
        would leave half of each tree without gradient.  A small alternating,
        sub-threshold probe breaks that *soft* symmetry while preserving all
        four hard signs and therefore the exact 0xC deployment payload.
        """

        with torch.no_grad():
            base = self.truth_table_logits.new_tensor(
                [-self.init_strength, -self.init_strength, self.init_strength, self.init_strength]
            )
            probe_magnitude = min(0.25, self.init_strength / 4.0)
            probe = self.truth_table_logits.new_tensor(
                [-probe_magnitude, probe_magnitude, -probe_magnitude, probe_magnitude]
            )
            channel = torch.arange(self.dim, device=self.truth_table_logits.device).view(-1, 1, 1)
            bitplane = torch.arange(
                self.activation_bits, device=self.truth_table_logits.device
            ).view(1, -1, 1)
            gate = torch.arange(self.NUM_GATES, device=self.truth_table_logits.device).view(1, 1, -1)
            probe_sign = ((channel + bitplane + gate).remainder(2) * 2 - 1).to(
                self.truth_table_logits.dtype
            )
            initialized = base.view(1, 1, 1, 4) + probe_sign.unsqueeze(-1) * probe
            self.truth_table_logits.copy_(initialized)

    @property
    def num_trees(self) -> int:
        return self.dim * self.activation_bits

    @property
    def num_unique_shared_lut_gates(self) -> int:
        return self.num_trees * self.NUM_GATES

    @property
    def gate_evaluations_per_image(self) -> int:
        return self.num_unique_shared_lut_gates * self.grid_size * self.grid_size

    def hard_truth_table_bits(self) -> Tensor:
        """Return hardened truth entries in [00,01,10,11] order."""

        return self.truth_table_logits.ge(0)

    def hard_truth_nibbles(self) -> Tensor:
        """Pack hardened truth tables as little-address-endian 4-bit LUTs."""

        bits = self.hard_truth_table_bits().to(torch.uint8)
        shifts = torch.arange(4, device=bits.device, dtype=torch.uint8)
        return torch.sum(torch.bitwise_left_shift(bits, shifts), dim=-1).to(torch.uint8)

    @staticmethod
    def hard_bitplanes_from_code(code: Tensor) -> Tensor:
        """Split signed integer codes into sign + seven magnitude bitplanes."""

        code_i16 = code.to(torch.int16)
        sign = code_i16.lt(0).unsqueeze(-1)
        magnitude = code_i16.abs().to(torch.uint8)
        shifts = torch.arange(7, device=code.device, dtype=torch.uint8)
        magnitude_bits = torch.bitwise_and(
            torch.bitwise_right_shift(magnitude.unsqueeze(-1), shifts), 1
        ).to(torch.bool)
        return torch.cat((sign, magnitude_bits), dim=-1)

    @staticmethod
    def decode_signed_magnitude(bitplanes: Tensor) -> Tensor:
        """Decode sign + seven magnitude planes to a differentiable code."""

        if bitplanes.shape[-1] != 8:
            raise ValueError(f"expected 8 bitplanes, got shape {tuple(bitplanes.shape)}")
        values = bitplanes.to(torch.float32)
        sign = values[..., 0]
        weights = values.new_tensor([1, 2, 4, 8, 16, 32, 64])
        magnitude = torch.sum(values[..., 1:] * weights, dim=-1)
        return magnitude * (1.0 - 2.0 * sign)

    def _soft_bitplanes(self, values: Tensor, scale: Tensor) -> Tensor:
        """Continuous signed-magnitude proxy used only for backward gradients."""

        normalized = values / scale
        sign = torch.sigmoid(-4.0 * normalized).unsqueeze(-1)
        magnitude = normalized.abs().clamp(0.0, 127.0)
        planes = []
        for bit in range(7):
            period = float(1 << (bit + 1))
            half_period = float(1 << bit)
            remainder = torch.remainder(magnitude, period)
            rise = torch.sigmoid(4.0 * (remainder - (half_period - 0.5)))
            fall = torch.sigmoid(4.0 * ((period - 0.5) - remainder))
            planes.append(rise * fall)
        return torch.cat((sign, torch.stack(planes, dim=-1)), dim=-1)

    def _spatial_views(self, bitplanes: Tensor) -> Tensor:
        """Build the nine fixed 3x3 sites without a convolution operator."""

        if bitplanes.ndim != 5:
            raise ValueError(
                "bitplanes must have shape [B,H,W,C,A], "
                f"got {tuple(bitplanes.shape)}"
            )
        batch, height, width, channels, bits = bitplanes.shape
        if channels != self.dim or bits != self.activation_bits:
            raise ValueError(
                f"expected C={self.dim}, A={self.activation_bits}; "
                f"got C={channels}, A={bits}"
            )

        # F.pad is only routing/storage here; no weighted spatial arithmetic is
        # hidden behind conv2d.  Layout is temporarily [B,C,A,H,W].
        channel_first = bitplanes.permute(0, 3, 4, 1, 2)
        padded = F.pad(channel_first, (1, 1, 1, 1), mode="constant", value=0)
        views = []
        for delta_y, delta_x in SITE_OFFSETS:
            y0 = 1 + delta_y
            x0 = 1 + delta_x
            view = padded[..., y0 : y0 + height, x0 : x0 + width]
            views.append(view.permute(0, 3, 4, 1, 2))
        return torch.stack(views, dim=-1)

    def fixed_leaf_bitplanes(self, bitplanes: Tensor) -> Tensor:
        """Gather eight deterministic leaves for every channel/bitplane tree."""

        views = self._spatial_views(bitplanes)
        batch, height, width, channels, bits, _ = views.shape
        index = self.leaf_site.view(1, 1, 1, channels, bits, self.NUM_LEAVES)
        index = index.expand(batch, height, width, channels, bits, self.NUM_LEAVES)
        return torch.gather(views, dim=-1, index=index)

    @staticmethod
    def _hard_tree(leaves: Tensor, truth_nibbles: Tensor) -> Tensor:
        """Evaluate 8->4->2->1 hard trees by direct nibble indexing."""

        if leaves.shape[-1] != 8:
            raise ValueError(f"expected eight leaves, got shape {tuple(leaves.shape)}")
        nibbles = truth_nibbles.view(
            1, 1, 1, truth_nibbles.shape[0], truth_nibbles.shape[1], 7
        )
        level1 = [
            hard_truth_table_gate(leaves[..., 2 * gate], leaves[..., 2 * gate + 1], nibbles[..., gate])
            for gate in range(4)
        ]
        level2 = [
            hard_truth_table_gate(level1[0], level1[1], nibbles[..., 4]),
            hard_truth_table_gate(level1[2], level1[3], nibbles[..., 5]),
        ]
        return hard_truth_table_gate(level2[0], level2[1], nibbles[..., 6])

    @staticmethod
    def _soft_gate(left: Tensor, right: Tensor, table: Tensor) -> Tensor:
        """Multilinear relaxation of a four-entry Boolean truth table."""

        not_left = 1.0 - left
        not_right = 1.0 - right
        return (
            table[..., 0] * not_left * not_right
            + table[..., 1] * not_left * right
            + table[..., 2] * left * not_right
            + table[..., 3] * left * right
        )

    @classmethod
    def _soft_tree(cls, leaves: Tensor, truth_tables: Tensor) -> Tensor:
        """Evaluate the differentiable proxy with the same fixed topology."""

        tables = truth_tables.view(
            1, 1, 1, truth_tables.shape[0], truth_tables.shape[1], 7, 4
        )
        level1 = [
            cls._soft_gate(
                leaves[..., 2 * gate], leaves[..., 2 * gate + 1], tables[..., gate, :]
            )
            for gate in range(4)
        ]
        level2 = [
            cls._soft_gate(level1[0], level1[1], tables[..., 4, :]),
            cls._soft_gate(level1[2], level1[3], tables[..., 5, :]),
        ]
        return cls._soft_gate(level2[0], level2[1], tables[..., 6, :])

    def forward_hard_bitplanes(self, bitplanes: Tensor) -> Tensor:
        """Run the complete fixed-routing Boolean tree and return root bits."""

        leaves = self.fixed_leaf_bitplanes(bitplanes)
        return self._hard_tree(leaves, self.hard_truth_nibbles())

    def _split_tokens(self, tokens: Tensor) -> Tuple[Tensor, Tensor]:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B,N,C], got {tuple(tokens.shape)}")
        if tokens.shape[-1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {tokens.shape[-1]}")
        expected_patches = self.grid_size * self.grid_size
        if tokens.shape[1] != expected_patches + 1:
            raise ValueError(
                f"expected {expected_patches + 1} tokens (CLS + grid), got {tokens.shape[1]}"
            )
        return tokens[:, :1], tokens[:, 1:]

    def hard_reference(
        self, tokens: Tensor, *, return_state: bool = False
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """Integer/Boolean transaction reference for deployment equivalence."""

        cls_token, patches = self._split_tokens(tokens)
        code, scale = self.input_quantizer.integer_code_and_scale(patches)
        batch = patches.shape[0]
        hard_bits = self.hard_bitplanes_from_code(code)
        hard_grid = hard_bits.view(
            batch, self.grid_size, self.grid_size, self.dim, self.activation_bits
        )
        leaves = self.fixed_leaf_bitplanes(hard_grid)
        roots = self._hard_tree(leaves, self.hard_truth_nibbles())
        root_code = self.decode_signed_magnitude(roots).to(torch.int16)
        if not self.training:
            with torch.no_grad():
                changed = roots.ne(hard_grid)
                self._root_changed_by_bitplane.add_(
                    changed.sum(dim=(0, 1, 2, 3)).to(torch.int64)
                )
                per_bitplane = roots.numel() // self.activation_bits
                self._root_total_by_bitplane.add_(
                    torch.full_like(self._root_total_by_bitplane, per_bitplane)
                )
                input_code_grid = code.view(
                    batch, self.grid_size, self.grid_size, self.dim
                ).to(torch.int16)
                self._root_code_changed.add_(
                    root_code.ne(input_code_grid).sum().to(torch.int64)
                )
                self._root_code_total.add_(root_code.numel())
                negative_zero = roots[..., 0] & ~roots[..., 1:].any(dim=-1)
                self._negative_zero_count.add_(negative_zero.sum().to(torch.int64))
                self._saturated_code_count.add_(
                    root_code.abs().eq(127).sum().to(torch.int64)
                )
        patch_output = root_code.view(batch, -1, self.dim).to(patches.dtype) * scale
        output = torch.cat((cls_token, patch_output), dim=1)
        if not return_state:
            return output
        state = {
            "input_code": code,
            "input_scale": scale,
            "input_bitplanes": hard_grid,
            "leaves": leaves,
            "root_bitplanes": roots,
            "root_code": root_code,
        }
        return output, state

    def forward(self, tokens: Tensor) -> Tensor:
        with torch.no_grad():
            hard_output, state = self.hard_reference(tokens, return_state=True)
        if not self.training:
            return hard_output

        cls_token, patches = self._split_tokens(tokens)
        batch = patches.shape[0]
        input_scale = state["input_scale"]
        del state
        soft_bits = self._soft_bitplanes(patches, input_scale)
        soft_bits = soft_bits.view(
            batch, self.grid_size, self.grid_size, self.dim, self.activation_bits
        )
        # The final autograd wrapper supplies the exact hard forward value.
        # The proxy must stay genuinely soft here: using an STE whose numerical
        # value is the projection-A table would make every right subtree have
        # zero input derivative at initialization.
        leaf_proxy = self.fixed_leaf_bitplanes(soft_bits)
        soft_tables = torch.sigmoid(self.truth_table_logits / self.truth_temperature)
        root_proxy = self._soft_tree(leaf_proxy, soft_tables)
        soft_code = self.decode_signed_magnitude(root_proxy)
        soft_patches = soft_code.view(batch, -1, self.dim).to(patches.dtype) * input_scale
        soft_output = torch.cat((cls_token, soft_patches), dim=1)
        return _HardPayloadSoftGradient.apply(hard_output, soft_output)

    def deployment_contract(self) -> Dict[str, object]:
        leaf_offsets = self.site_offsets[self.leaf_site]
        return {
            "operator": "shared_logic_tree_conv3x3",
            "activation_format": "signed_magnitude_a8",
            "activation_bitplanes": self.activation_bits,
            "tree_depth": self.TREE_DEPTH,
            "leaves_per_tree": self.NUM_LEAVES,
            "gates_per_tree": self.NUM_GATES,
            "truth_entries_per_gate": self.NUM_TRUTH_ENTRIES,
            "truth_address": "(A<<1)|B",
            "identity_truth_nibble": TRUTH_NIBBLE_A,
            "num_trees": self.num_trees,
            "num_unique_shared_lut_gates": self.num_unique_shared_lut_gates,
            "gate_evaluations_per_image": self.gate_evaluations_per_image,
            "spatial_sharing": True,
            "learned_routing": False,
            "leaf_zero": "same_channel_same_bitplane_center",
            "leaf_site": self.leaf_site.detach().cpu().tolist(),
            "leaf_offsets_dy_dx": leaf_offsets.detach().cpu().tolist(),
            "root_state": "boolean_bitplane_inside_tree_core",
            "training_relaxation": "hard_A_with_alternating_subthreshold_B_gradient_probe",
            "tree_core_executor": ["truth_table_index", "shift", "mask", "fixed_gather"],
            "tree_core_forbidden_ops": ["matmul", "linear", "conv2d"],
            "transaction_bridge": "PyTorch_power_of_two_quantizer_and_float_carrier_pending_integer_exponent_executor",
            "cls_policy": "exact_bypass",
        }

    @torch.no_grad()
    def reset_statistics(self) -> None:
        self._root_changed_by_bitplane.zero_()
        self._root_total_by_bitplane.zero_()
        self._root_code_changed.zero_()
        self._root_code_total.zero_()
        self._negative_zero_count.zero_()
        self._saturated_code_count.zero_()

    @torch.no_grad()
    def statistics(self) -> Dict[str, object]:
        truth_bits = self.hard_truth_table_bits().cpu()
        expected_a = torch.tensor(TRUTH_TABLE_A, dtype=torch.bool).view(1, 1, 1, 4)
        entry_flips = truth_bits.ne(expected_a)
        nibbles = self.hard_truth_nibbles().cpu()
        changed = self._root_changed_by_bitplane.detach().cpu()
        totals = self._root_total_by_bitplane.detach().cpu()
        rates = torch.where(
            totals > 0,
            changed.to(torch.float64) / totals.to(torch.float64),
            torch.zeros_like(changed, dtype=torch.float64),
        )
        margin = self.truth_table_logits.detach().abs().float().flatten()
        quantiles = torch.quantile(
            margin,
            margin.new_tensor([0.0, 0.1, 0.5, 0.9, 1.0]),
        ).cpu()
        code_total = int(self._root_code_total.detach().cpu())
        return {
            "unique_shared_lut_gates": self.num_unique_shared_lut_gates,
            "hard_truth_entry_flips_from_A": int(entry_flips.sum()),
            "hard_truth_entry_flips_by_gate": entry_flips.sum(dim=(0, 1, 3)).tolist(),
            "hard_truth_entry_flips_by_bitplane": entry_flips.sum(dim=(0, 2, 3)).tolist(),
            "hard_lut_nibbles_not_0xC": int(nibbles.ne(TRUTH_NIBBLE_A).sum()),
            "hard_lut_nibbles_not_0xC_by_gate": nibbles.ne(TRUTH_NIBBLE_A).sum(
                dim=(0, 1)
            ).tolist(),
            "truth_nibble_histogram_0_to_15": torch.bincount(
                nibbles.flatten().to(torch.int64), minlength=16
            ).tolist(),
            "absolute_logit_margin_quantiles_0_10_50_90_100": quantiles.tolist(),
            "root_bit_changes_by_bitplane": changed.tolist(),
            "root_bit_totals_by_bitplane": totals.tolist(),
            "root_bit_change_rate_by_bitplane": rates.tolist(),
            "root_code_changed": int(self._root_code_changed.detach().cpu()),
            "root_code_total": code_total,
            "root_code_change_rate": (
                int(self._root_code_changed.detach().cpu()) / code_total
                if code_total else 0.0
            ),
            "negative_zero_count": int(self._negative_zero_count.detach().cpu()),
            "saturated_root_code_count": int(self._saturated_code_count.detach().cpu()),
        }


__all__ = [
    "SITE_OFFSETS",
    "TRUTH_NIBBLE_A",
    "TRUTH_TABLE_A",
    "SharedLogicTreeConv3x3",
    "hard_truth_table_gate",
]
