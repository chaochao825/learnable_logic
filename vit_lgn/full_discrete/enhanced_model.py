from __future__ import annotations

import math

import torch
import torch.nn as nn

from .enhancements_expert import FourModeStateSelectedFFN, ParallelBitSliceLogicFFN
from .enhancements_lut import GroupwiseDiscreteActivationLUT, MonotonicHeadGapLUT
from .enhancements_spatial import DiscreteDepthwiseLocalBranch
from .model import DiscreteGatedFFN, FullDiscreteViT
from .shiftadd import PowerOfTwoActivationQuantizer, ShiftAddLinear, _ste


class GroupLUTDiscreteFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, groups: int,
                 weight_bits: int, activation_bits: int) -> None:
        super().__init__()
        self.gate = ShiftAddLinear(dim, hidden_dim, weight_bits, activation_bits)
        self.up = ShiftAddLinear(dim, hidden_dim, weight_bits, activation_bits)
        self.activation_lut = GroupwiseDiscreteActivationLUT(
            hidden_dim, groups=groups, initialization="relu"
        )
        self.down = ShiftAddLinear(hidden_dim, dim, weight_bits, activation_bits)
        self.hidden_quantizer = PowerOfTwoActivationQuantizer(activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        hard_gate = (gate_logits >= 0).to(x.dtype)
        gate = _ste(hard_gate, torch.sigmoid(gate_logits)) if self.training else hard_gate
        hidden = self.hidden_quantizer(gate * self.up(x))
        return self.down(self.activation_lut(hidden))

    def deployment_contract(self) -> dict[str, object]:
        return {
            "activation": self.activation_lut.deployment_contract(),
            "projections": "signed bit-plane shift-add",
            "general_multipliers": 0,
        }


class StateSelectedFFNAdapter(nn.Module):
    def __init__(self, module: FourModeStateSelectedFFN, control: str, block_index: int) -> None:
        super().__init__()
        self.module = module
        self.control = control
        self.block_index = int(block_index)
        self.register_buffer("state_counts", torch.zeros(4, dtype=torch.int64), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, state = self.module(
            x, control=self.control, script_step=self.block_index, return_state=True
        )
        if not self.training:
            with torch.no_grad():
                self.state_counts += torch.bincount(state.detach(), minlength=4).to(
                    self.state_counts.device
                )
        return output

    def reset_state_statistics(self) -> None:
        self.state_counts.zero_()

    def state_statistics(self) -> list[int]:
        return [int(value) for value in self.state_counts.cpu().tolist()]

    def deployment_contract(self) -> dict[str, object]:
        return {**self.module.deployment_contract(), "selected_control": self.control}


def _copy_common_ffn(source: DiscreteGatedFFN, target: nn.Module) -> None:
    destination = target.base if hasattr(target, "base") else target
    for name in ("gate", "up", "down"):
        getattr(destination, name).load_state_dict(getattr(source, name).state_dict())


class EnhancedFullDiscreteViT(FullDiscreteViT):
    """FullDiscreteViT with independently switchable, matched enhancement families."""

    def __init__(
        self,
        *args,
        learned_gap: bool = False,
        gap_max: int = 63,
        group_lut_groups: int = 0,
        local_layers: int = 0,
        local_branch_shift: int = 2,
        logic_expert_width: int = 0,
        logic_expert_count: int = 1,
        state_control: str = "none",
        state_expert_width: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if state_control not in {"none", "dynamic", "static", "random", "script"}:
            raise ValueError("unsupported state_control")
        if sum((group_lut_groups > 0, logic_expert_width > 0, state_control != "none")) > 1:
            raise ValueError("FFN enhancement families must be evaluated separately")
        depth = len(self.blocks)
        if local_layers < 0 or local_layers > depth:
            raise ValueError("local_layers must be in [0,depth]")
        dim = self.position.shape[-1]
        grid_size = math.isqrt(self.patch_embed.num_patches)
        if grid_size * grid_size != self.patch_embed.num_patches and local_layers:
            raise ValueError("local branches require a square patch grid")
        weight_bits = self.weight_magnitude_bits
        activation_bits = self.activation_bits

        if learned_gap:
            for block in self.blocks:
                block.attn.gap_lut = MonotonicHeadGapLUT(
                    heads=block.attn.heads, max_gap=gap_max, gap_shift=1
                )

        for index, block in enumerate(self.blocks):
            original_ffn = block.ffn
            hidden_dim = original_ffn.gate.out_features if isinstance(original_ffn, DiscreteGatedFFN) else dim * 4
            if group_lut_groups:
                block.ffn = GroupLUTDiscreteFFN(
                    dim, hidden_dim, group_lut_groups, weight_bits, activation_bits
                )
                _copy_common_ffn(original_ffn, block.ffn)
            elif logic_expert_width:
                block.ffn = ParallelBitSliceLogicFFN(
                    dim, hidden_dim, logic_expert_width,
                    num_logic_experts=logic_expert_count,
                    logic_output_shift=2,
                    activation_bits=activation_bits,
                    weight_bits=weight_bits,
                )
                _copy_common_ffn(original_ffn, block.ffn)
            elif state_control != "none":
                width = state_expert_width or max(dim, 256)
                selected = FourModeStateSelectedFFN(
                    dim, hidden_dim, width,
                    experts_per_mode=1,
                    logic_output_shift=2,
                    activation_bits=activation_bits,
                    weight_bits=weight_bits,
                )
                _copy_common_ffn(original_ffn, selected)
                block.ffn = StateSelectedFFNAdapter(selected, state_control, index)

        self.local_branches = nn.ModuleDict({
            str(index): DiscreteDepthwiseLocalBranch(
                dim=dim,
                grid_size=grid_size,
                weight_bits=min(weight_bits, 4),
                activation_bits=activation_bits,
                branch_shift=local_branch_shift,
                zero_init=True,
            )
            for index in range(local_layers)
        })
        self.enhancement_config = {
            "learned_gap": learned_gap,
            "group_lut_groups": group_lut_groups,
            "local_layers": local_layers,
            "logic_expert_width": logic_expert_width,
            "logic_expert_count": logic_expert_count,
            "state_control": state_control,
            "state_expert_width": state_expert_width,
            "gap_max": gap_max,
            "local_branch_shift": local_branch_shift,
            "effective_state_expert_width": (
                state_expert_width or max(dim, 256) if state_control != "none" else 0
            ),
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(images)
        cls = self.parameter_quantizer(self.cls_token).expand(images.shape[0], -1, -1)
        position = self.parameter_quantizer(self.position)
        tokens = self.token_quantizer(torch.cat((cls, tokens), dim=1) + position)
        for index, block in enumerate(self.blocks):
            key = str(index)
            if key in self.local_branches:
                tokens = self.local_branches[key](tokens)
            tokens = block(tokens)
        return self.head(self.norm(tokens)[:, 0])

    def deployment_contract(self) -> dict[str, object]:
        base = super().deployment_contract()
        base["enhancements"] = self.enhancement_config
        block_contracts = []
        for block in self.blocks:
            gap_contract = (
                block.attn.gap_lut.deployment_contract()
                if block.attn.gap_lut is not None
                else {"weights": [1, 2, 4, 8], "mapping": "fixed score-gap"}
            )
            ffn_contract = (
                block.ffn.deployment_contract()
                if hasattr(block.ffn, "deployment_contract")
                else {"type": type(block.ffn).__name__}
            )
            block_contracts.append({"gap": gap_contract, "ffn": ffn_contract})
        base["attention"] = (
            "threshold bits + XNOR/popcount + hard Top-K + " +
            ("per-head monotone learned gap LUT" if self.enhancement_config["learned_gap"]
             else "fixed {8,4,2,1} score-gap")
        )
        base["block_contracts"] = block_contracts
        base["local_branch_contracts"] = {
            key: branch.deployment_contract() for key, branch in self.local_branches.items()
        }
        return base

    def reset_state_statistics(self) -> None:
        for block in self.blocks:
            if isinstance(block.ffn, StateSelectedFFNAdapter):
                block.ffn.reset_state_statistics()

    def state_statistics(self) -> dict[str, list[int]]:
        return {
            str(index): block.ffn.state_statistics()
            for index, block in enumerate(self.blocks)
            if isinstance(block.ffn, StateSelectedFFNAdapter)
        }


def enhanced_full_discrete_vit(**kwargs) -> EnhancedFullDiscreteViT:
    return EnhancedFullDiscreteViT(**kwargs)
