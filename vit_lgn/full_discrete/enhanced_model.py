from __future__ import annotations

import math

import torch
import torch.nn as nn

from .enhancements_expert import FourModeStateSelectedFFN, ParallelBitSliceLogicFFN
from .enhancements_global_lut import A8GlobalLUTTreeMixer
from .enhancements_hadamard import FixedHadamardGlobalMixer
from .enhancements_logic_tree import SharedLogicTreeConv3x3
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


class ParallelContentHadamardMixer(nn.Module):
    """Keep content routing and add a weak fixed integer global side branch."""

    def __init__(self, content: nn.Module, fixed: FixedHadamardGlobalMixer) -> None:
        super().__init__()
        self.content = content
        self.fixed = fixed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.content(x) + self.fixed(x)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "operator": "parallel_content_and_fixed_hadamard",
            "content": "xnor_popcount_hard_topk",
            "fixed": self.fixed.deployment_contract(),
            "merge": "power_of_two_exponent_align_then_integer_add",
            "general_multipliers": 0,
        }


class ParallelContentGlobalLUTMixer(nn.Module):
    """Keep hard content routing and add a nonlinear ROM-tree global branch."""

    def __init__(self, content: nn.Module, lut_tree: A8GlobalLUTTreeMixer) -> None:
        super().__init__()
        self.content = content
        self.lut_tree = lut_tree

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.content(x) + self.lut_tree(x)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "operator": "parallel_content_and_a8_global_lut_tree",
            "content": "xnor_popcount_hard_topk",
            "lut_tree": self.lut_tree.deployment_contract(),
            "merge": "power_of_two_exponent_align_then_integer_add",
            "general_multipliers_hard_forward": 0,
        }


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
        local_operator: str = "depthwise_shiftadd",
        local_branch_shift: int = 2,
        global_mixer: str = "attention",
        hadamard_group_size: int = 32,
        hadamard_branch_shift: int = 2,
        hybrid_attention_period: int = 3,
        global_lut_group_size: int = 32,
        global_lut_branch_shift: int = 2,
        logic_expert_width: int = 0,
        logic_expert_count: int = 1,
        state_control: str = "none",
        state_expert_width: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        if state_control not in {"none", "dynamic", "static", "random", "script"}:
            raise ValueError("unsupported state_control")
        if local_operator not in {"depthwise_shiftadd", "logic_tree3x3"}:
            raise ValueError("unsupported local_operator")
        if global_mixer not in {
            "attention", "hadamard", "hybrid", "parallel", "parallel_lut_tree"
        }:
            raise ValueError("unsupported global_mixer")
        if global_mixer == "hadamard" and learned_gap:
            raise ValueError("learned_gap is only defined for the attention mixer")
        if hybrid_attention_period < 1:
            raise ValueError("hybrid_attention_period must be positive")
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

        if global_mixer in {"hadamard", "hybrid", "parallel"}:
            for index, block in enumerate(self.blocks):
                keep_attention = (
                    global_mixer == "hybrid"
                    and (index + 1) % hybrid_attention_period == 0
                )
                if keep_attention:
                    continue
                fixed_mixer = FixedHadamardGlobalMixer(
                    dim=dim,
                    patch_tokens=self.patch_embed.num_patches,
                    block_index=index,
                    activation_bits=activation_bits,
                    group_size=hadamard_group_size,
                    branch_shift=hadamard_branch_shift,
                )
                if global_mixer == "parallel":
                    block.attn = ParallelContentHadamardMixer(
                        content=block.attn, fixed=fixed_mixer
                    )
                else:
                    block.attn = fixed_mixer

        if global_mixer == "parallel_lut_tree":
            for index, block in enumerate(self.blocks):
                block.attn = ParallelContentGlobalLUTMixer(
                    content=block.attn,
                    lut_tree=A8GlobalLUTTreeMixer(
                        dim=dim,
                        patch_tokens=self.patch_embed.num_patches,
                        block_index=index,
                        activation_bits=activation_bits,
                        group_size=global_lut_group_size,
                        branch_shift=global_lut_branch_shift,
                    ),
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

        if local_operator == "logic_tree3x3":
            local_branch_factory = lambda: SharedLogicTreeConv3x3(
                dim=dim,
                grid_size=grid_size,
                activation_bits=activation_bits,
            )
        else:
            local_branch_factory = lambda: DiscreteDepthwiseLocalBranch(
                dim=dim,
                grid_size=grid_size,
                weight_bits=min(weight_bits, 4),
                activation_bits=activation_bits,
                branch_shift=local_branch_shift,
                zero_init=True,
            )
        self.local_branches = nn.ModuleDict(
            {str(index): local_branch_factory() for index in range(local_layers)}
        )
        self.enhancement_config = {
            "learned_gap": learned_gap,
            "group_lut_groups": group_lut_groups,
            "local_layers": local_layers,
            "local_operator": local_operator,
            "global_mixer": global_mixer,
            "hadamard_group_size": hadamard_group_size,
            "hadamard_branch_shift": hadamard_branch_shift,
            "hybrid_attention_period": hybrid_attention_period,
            "global_lut_group_size": global_lut_group_size,
            "global_lut_branch_shift": global_lut_branch_shift,
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
        # super().__init__ propagates the requested backend before enhancement
        # FFNs are installed.  Re-run propagation so every replacement
        # ShiftAddLinear shares the same inference ABI.
        self.set_inference_backend(self.inference_backend)

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
            ffn_contract = (
                block.ffn.deployment_contract()
                if hasattr(block.ffn, "deployment_contract")
                else {"type": type(block.ffn).__name__}
            )
            if isinstance(block.attn, ParallelContentHadamardMixer):
                gap_contract = (
                    block.attn.content.gap_lut.deployment_contract()
                    if block.attn.content.gap_lut is not None
                    else {"weights": [1, 2, 4, 8], "mapping": "fixed score-gap"}
                )
                mixer_contract = block.attn.deployment_contract()
                block_contract = {
                    "gap": gap_contract,
                    "global_mixer": mixer_contract,
                    "ffn": ffn_contract,
                }
            elif isinstance(block.attn, ParallelContentGlobalLUTMixer):
                gap_contract = (
                    block.attn.content.gap_lut.deployment_contract()
                    if block.attn.content.gap_lut is not None
                    else {"weights": [1, 2, 4, 8], "mapping": "fixed score-gap"}
                )
                block_contract = {
                    "gap": gap_contract,
                    "global_mixer": block.attn.deployment_contract(),
                    "ffn": ffn_contract,
                }
            elif isinstance(block.attn, FixedHadamardGlobalMixer):
                mixer_contract = block.attn.deployment_contract()
                block_contract = {
                    "global_mixer": mixer_contract,
                    "ffn": ffn_contract,
                }
            else:
                gap_contract = (
                    block.attn.gap_lut.deployment_contract()
                    if block.attn.gap_lut is not None
                    else {"weights": [1, 2, 4, 8], "mapping": "fixed score-gap"}
                )
                mixer_contract = {
                    "operator": "xnor_popcount_hard_topk",
                    "gap": gap_contract,
                }
                # Keep the v1 convenience key for callers that only inspect
                # attention experiments while also exposing the generic mixer.
                block_contract = {
                    "gap": gap_contract,
                    "global_mixer": mixer_contract,
                    "ffn": ffn_contract,
                }
            block_contracts.append(block_contract)
        if self.enhancement_config["global_mixer"] == "hadamard":
            base["attention"] = "none"
            base["global_mixer"] = (
                "fixed H-D-H/N integer butterfly + CLS mean/broadcast"
            )
        elif self.enhancement_config["global_mixer"] == "hybrid":
            base["attention"] = (
                "one hard XNOR/Top-K content mixer every "
                f"{self.enhancement_config['hybrid_attention_period']} blocks"
            )
            base["global_mixer"] = (
                "fixed H-D-H/N in remaining blocks; final periodic block is attention"
            )
        elif self.enhancement_config["global_mixer"] == "parallel":
            base["attention"] = (
                "hard XNOR/Top-K in every block, parallel with fixed Hadamard"
            )
            base["global_mixer"] = (
                "content-dependent routing + weak fixed H-D-H/N side branch"
            )
        elif self.enhancement_config["global_mixer"] == "parallel_lut_tree":
            base["attention"] = (
                "hard XNOR/Top-K in every block, parallel with nonlinear A8 ROM tree"
            )
            base["global_mixer"] = (
                "content-dependent routing + group-shared A8 pair-LUT reduction/broadcast"
            )
        else:
            base["attention"] = (
                "threshold bits + XNOR/popcount + hard Top-K + " +
                ("per-head monotone learned gap LUT"
                 if self.enhancement_config["learned_gap"]
                 else "fixed {8,4,2,1} score-gap")
            )
            base["global_mixer"] = "content-dependent hard attention"
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
