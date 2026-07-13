from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .shiftadd import PowerOfTwoActivationQuantizer, ShiftAddLinear, _ste


@torch.no_grad()
def _shiftadd_payload(layer: ShiftAddLinear) -> dict[str, torch.Tensor]:
    sign, planes, scale = layer.bitplane_codes()
    return {
        "weight_sign": sign.to(torch.uint8),
        "weight_magnitude_planes": planes.to(torch.uint8),
        "weight_scale_power_of_two": scale,
        "input_activation_bits": layer.input_quantizer.bits,
        "output_activation_bits": layer.output_quantizer.bits,
        "input_code_range": [layer.input_quantizer.qmin, layer.input_quantizer.qmax],
        "output_code_range": [layer.output_quantizer.qmin, layer.output_quantizer.qmax],
        "weight_integer_range": [-layer.qmax, layer.qmax],
        "scale_exponent_floor": -24,
        "rounding": "IEEE ties-to-even for weight, activation, and scale exponent",
        "saturation": "clamp to the listed signed code ranges",
    }


def _positive_float(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


class BitSliceBooleanLogicExpert(nn.Module):
    """A scalable bank of hard two-input Boolean gates.

    Each input channel is quantized to a signed integer and exposed as one sign
    bit plus ``activation_bits - 1`` magnitude bits.  Every expert gate selects
    two bit-slice wires and owns a four-entry truth table in ``00, 01, 10, 11``
    order.  The selected wires and table payload are hard in the forward pass;
    softmax/sigmoid values are used only as STE backward surrogates.

    The gate outputs are mapped from ``{0,1}`` to ``{-1,+1}`` and returned to
    the model dimension through a signed bit-plane ``ShiftAddLinear``.  Thus an
    exported gate is two fixed wires, a 4-bit LUT, and a shift-add output row,
    not a dense floating matrix multiply.
    """

    def __init__(
        self,
        dim: int,
        expert_width: int,
        *,
        activation_bits: int = 8,
        weight_bits: int = 4,
        connection_temperature: float = 1.0,
        lut_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.expert_width = _positive_int("expert_width", expert_width)
        self.activation_bits = _positive_int("activation_bits", activation_bits)
        if self.activation_bits < 2 or self.activation_bits > 16:
            raise ValueError("activation_bits must be in [2,16]")
        self.weight_bits = _positive_int("weight_bits", weight_bits)
        if self.weight_bits > 8:
            raise ValueError("weight_bits must be in [1,8]")
        self.connection_temperature = _positive_float(
            "connection_temperature", connection_temperature
        )
        self.lut_temperature = _positive_float("lut_temperature", lut_temperature)

        # Signed-magnitude encoding uses exactly activation_bits bits per channel.
        self.magnitude_bits = self.activation_bits - 1
        self.source_count = self.dim * self.activation_bits
        self.input_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)
        self.left_connection_logits = nn.Parameter(
            torch.empty(self.expert_width, self.source_count)
        )
        self.right_connection_logits = nn.Parameter(
            torch.empty(self.expert_width, self.source_count)
        )
        self.truth_table_logits = nn.Parameter(torch.empty(self.expert_width, 4))
        self.output_projection = ShiftAddLinear(
            self.expert_width,
            self.dim,
            magnitude_bits=self.weight_bits,
            output_bits=self.activation_bits,
            input_bits=self.activation_bits,
            bias=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # A small dense training carrier with one initially preferred hard wire.
        # The deployment payload retains only the two argmax indices per gate.
        nn.init.normal_(self.left_connection_logits, mean=0.0, std=0.02)
        nn.init.normal_(self.right_connection_logits, mean=0.0, std=0.02)
        with torch.no_grad():
            gate = torch.arange(self.expert_width)
            left = gate.remainder(self.source_count)
            right = (gate * 17 + 1).remainder(self.source_count)
            self.left_connection_logits[gate, left] += 1.0
            self.right_connection_logits[gate, right] += 1.0

            # Cycle through AND, OR, XOR and XNOR so the initial hard network is
            # useful and heterogeneous instead of a bank of constant gates.
            seed_tables = torch.tensor(
                [
                    [-2.0, -2.0, -2.0, 2.0],
                    [-2.0, 2.0, 2.0, 2.0],
                    [-2.0, 2.0, 2.0, -2.0],
                    [2.0, -2.0, -2.0, 2.0],
                ],
                dtype=self.truth_table_logits.dtype,
                device=self.truth_table_logits.device,
            )
            self.truth_table_logits.copy_(
                seed_tables[gate.remainder(seed_tables.shape[0])]
            )

    def hard_bit_slices_from_code(self, code: torch.Tensor) -> torch.Tensor:
        """Return signed-magnitude bit slices for an integer-code tensor."""

        if code.shape[-1] != self.dim:
            raise ValueError(
                f"expected final dimension {self.dim}, got {code.shape[-1]}"
            )
        integer = code.to(torch.int64)
        qmax = (1 << self.magnitude_bits) - 1
        if bool((integer.abs() > qmax).any()):
            raise ValueError(f"integer code exceeds signed {self.activation_bits}-bit range")
        sign = integer.lt(0).unsqueeze(-1)
        magnitude = integer.abs()
        planes = torch.stack(
            [
                torch.bitwise_and(
                    torch.bitwise_right_shift(magnitude, bit), 1
                ).bool()
                for bit in range(self.magnitude_bits)
            ],
            dim=-1,
        )
        return torch.cat((sign, planes), dim=-1).flatten(-2).to(
            self.left_connection_logits.dtype
        )

    def _soft_bit_slices(self, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # This is deliberately only a training derivative surrogate.  The hard
        # forward payload above remains exact signed-magnitude bits.
        qmax = (1 << self.magnitude_bits) - 1
        normalized = (x / scale).clamp(-qmax, qmax)
        magnitude = normalized.abs()
        temperature = 4.0
        sign = torch.sigmoid(-temperature * normalized).unsqueeze(-1)
        planes: list[torch.Tensor] = []
        for bit in range(self.magnitude_bits):
            half_period = float(1 << bit)
            period = 2.0 * half_period
            phase = torch.remainder(magnitude, period)
            rise = torch.sigmoid(temperature * (phase - (half_period - 0.5)))
            fall = torch.sigmoid(temperature * ((period - 0.5) - phase))
            planes.append(rise * fall)
        return torch.cat((sign, torch.stack(planes, dim=-1)), dim=-1).flatten(-2)

    def bit_slices(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize ``x`` and expose hard-forward differentiable bit slices."""

        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {x.shape[-1]}")
        code, scale = self.input_quantizer.integer_code_and_scale(x)
        hard = self.hard_bit_slices_from_code(code)
        if not self.training:
            return hard
        return _ste(hard, self._soft_bit_slices(x, scale))

    def _connection_selector(self, logits: torch.Tensor) -> torch.Tensor:
        index = logits.argmax(dim=-1)
        hard = F.one_hot(index, num_classes=self.source_count).to(logits.dtype)
        if not self.training:
            return hard
        soft = torch.softmax(logits / self.connection_temperature, dim=-1)
        return _ste(hard, soft)

    def _truth_table(self) -> torch.Tensor:
        hard = self.truth_table_logits.ge(0).to(self.truth_table_logits.dtype)
        if not self.training:
            return hard
        soft = torch.sigmoid(self.truth_table_logits / self.lut_temperature)
        return _ste(hard, soft)

    def boolean_gate_outputs(self, bit_slices: torch.Tensor) -> torch.Tensor:
        """Apply the hard connection and LUT bank to externally supplied bits."""

        if bit_slices.shape[-1] != self.source_count:
            raise ValueError(
                f"expected {self.source_count} source bits, got {bit_slices.shape[-1]}"
            )
        bit_slices = bit_slices.to(self.left_connection_logits.dtype)
        left_selector = self._connection_selector(self.left_connection_logits)
        right_selector = self._connection_selector(self.right_connection_logits)
        left = torch.matmul(bit_slices, left_selector.transpose(0, 1))
        right = torch.matmul(bit_slices, right_selector.transpose(0, 1))
        table = self._truth_table()
        return (
            (1.0 - left) * (1.0 - right) * table[:, 0]
            + (1.0 - left) * right * table[:, 1]
            + left * (1.0 - right) * table[:, 2]
            + left * right * table[:, 3]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_bits = self.boolean_gate_outputs(self.bit_slices(x))
        bipolar = gate_bits.mul(2.0).sub(1.0)
        return self.output_projection(bipolar)

    @torch.no_grad()
    def hard_payload(self) -> dict[str, torch.Tensor]:
        """Return the exact per-gate payload needed by a deployment exporter."""

        left = self.left_connection_logits.argmax(dim=-1).to(torch.int32)
        right = self.right_connection_logits.argmax(dim=-1).to(torch.int32)
        truth = self.truth_table_logits.ge(0).to(torch.uint8)
        return {
            "left_source": left,
            "right_source": right,
            "truth_table_00_01_10_11": truth,
            "output_projection": _shiftadd_payload(self.output_projection),
        }

    def deployment_contract(self) -> dict[str, object]:
        return {
            "input_encoding": (
                f"{self.activation_bits}-bit signed magnitude "
                "(sign then little-endian magnitude planes per channel)"
            ),
            "source_bit_count": self.source_count,
            "boolean_gate_count": self.expert_width,
            "connections_per_gate": 2,
            "truth_table_bits_per_gate": 4,
            "hard_connection": "argmax index; two fixed wires after export",
            "hard_lut": "thresholded 4-bit payload in 00,01,10,11 order",
            "output_projection": self.output_projection.deployment_contract(),
            "general_multipliers": 0,
            "training_only": (
                "dense connection softmax and sigmoid truth-table STE surrogates"
            ),
        }


class DiscreteShiftAddFFNBranch(nn.Module):
    """The conventional discrete gated branch used beside LogicExperts."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        *,
        activation_bits: int = 8,
        weight_bits: int = 4,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.activation_bits = _positive_int("activation_bits", activation_bits)
        self.weight_bits = _positive_int("weight_bits", weight_bits)
        self.gate = ShiftAddLinear(
            self.dim, self.hidden_dim, self.weight_bits, self.activation_bits
        )
        self.up = ShiftAddLinear(
            self.dim, self.hidden_dim, self.weight_bits, self.activation_bits
        )
        self.down = ShiftAddLinear(
            self.hidden_dim, self.dim, self.weight_bits, self.activation_bits
        )
        self.hidden_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)
        hard_gate = gate_logits.ge(0).to(x.dtype)
        gate = (
            _ste(hard_gate, torch.sigmoid(gate_logits))
            if self.training
            else hard_gate
        )
        return self.down(self.hidden_quantizer(gate * self.up(x)))

    def deployment_contract(self) -> dict[str, object]:
        return {
            "gate": "one threshold bit per hidden channel",
            "projections": 3,
            "projection_implementation": "signed bit-plane shift-add",
            "general_multipliers": 0,
        }

    @torch.no_grad()
    def hard_payload(self) -> dict[str, object]:
        return {
            "gate_projection": _shiftadd_payload(self.gate),
            "up_projection": _shiftadd_payload(self.up),
            "down_projection": _shiftadd_payload(self.down),
            "gate_threshold": 0,
        }


class ParallelBitSliceLogicFFN(nn.Module):
    """Discrete gated FFN with one or more parallel Boolean LogicExperts.

    Increasing ``expert_width`` grows gates within an expert; increasing
    ``num_logic_experts`` adds independent banks.  Their outputs are accumulated
    beside the conventional discrete FFN and requantized.  ``logic_output_shift``
    is an exact power-of-two attenuation and therefore remains a wiring shift.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        expert_width: int,
        *,
        num_logic_experts: int = 1,
        logic_output_shift: int = 0,
        activation_bits: int = 8,
        weight_bits: int = 4,
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.expert_width = _positive_int("expert_width", expert_width)
        self.num_logic_experts = _positive_int(
            "num_logic_experts", num_logic_experts
        )
        self.logic_output_shift = _nonnegative_int(
            "logic_output_shift", logic_output_shift
        )
        self.activation_bits = _positive_int("activation_bits", activation_bits)
        self.weight_bits = _positive_int("weight_bits", weight_bits)
        self.base = DiscreteShiftAddFFNBranch(
            self.dim,
            self.hidden_dim,
            activation_bits=self.activation_bits,
            weight_bits=self.weight_bits,
        )
        self.logic_experts = nn.ModuleList(
            [
                BitSliceBooleanLogicExpert(
                    self.dim,
                    self.expert_width,
                    activation_bits=self.activation_bits,
                    weight_bits=self.weight_bits,
                )
                for _ in range(self.num_logic_experts)
            ]
        )
        self.output_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base(x)
        logic = torch.stack([expert(x) for expert in self.logic_experts], dim=0).sum(0)
        logic = logic * float(2.0 ** -self.logic_output_shift)
        return self.output_quantizer(base + logic)

    def deployment_contract(self) -> dict[str, object]:
        return {
            "parallel_branches": 1 + self.num_logic_experts,
            "base_branch": self.base.deployment_contract(),
            "logic_expert_count": self.num_logic_experts,
            "boolean_gate_count": self.num_logic_experts * self.expert_width,
            "logic_output_scale": f"right shift {self.logic_output_shift}",
            "logic_expert": self.logic_experts[0].deployment_contract(),
            "merge": "integer addition followed by activation requantization",
            "general_multipliers": 0,
        }

    @torch.no_grad()
    def hard_payload(self) -> dict[str, object]:
        return {
            "base": self.base.hard_payload(),
            "logic_experts": [expert.hard_payload() for expert in self.logic_experts],
            "logic_output_right_shift": self.logic_output_shift,
        }


class FourModeStateSelectedFFN(nn.Module):
    """A global/scripted 2-bit state selects four real Logic-FFN modes.

    All four modes have independent Boolean connections and truth-table payloads.
    State therefore changes the computed Boolean function; it does not merely
    rotate, permute, or sign-flip the final feature tensor.  The conventional
    shift-add FFN branch is shared so experiments isolate the mode-specific
    LogicExpert contribution.

    Controls are matched within one module and one parameter budget:

    ``dynamic``
        A shift-add controller reads the CLS/global vector and hard-argmaxes one
        of four states.  Softmax is an STE backward surrogate only.
    ``static``
        Select a fixed state for every sample.
    ``random``
        Select a supplied or uniformly sampled state for every sample.
    ``script``
        Select an externally supplied state or a 2-bit entry from ``script``.

    With ``matched_ste=True`` (the default), static/random/script controls use
    the same controller-softmax backward selector as dynamic while preserving
    their requested hard forward state.  This provides a controlled hard-state
    ablation without changing trainable parameter count or surrogate topology.
    """

    CONTROL_MODES = ("dynamic", "static", "random", "script")
    STATE_COUNT = 4

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        expert_width: int,
        *,
        experts_per_mode: int = 1,
        logic_output_shift: int = 0,
        activation_bits: int = 8,
        weight_bits: int = 4,
        state_temperature: float = 1.0,
        static_state: int = 0,
        script: Sequence[int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()
        self.dim = _positive_int("dim", dim)
        self.hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.expert_width = _positive_int("expert_width", expert_width)
        self.experts_per_mode = _positive_int("experts_per_mode", experts_per_mode)
        self.logic_output_shift = _nonnegative_int(
            "logic_output_shift", logic_output_shift
        )
        self.activation_bits = _positive_int("activation_bits", activation_bits)
        self.weight_bits = _positive_int("weight_bits", weight_bits)
        self.state_temperature = _positive_float(
            "state_temperature", state_temperature
        )
        static_state = self._validate_state_scalar(static_state, "static_state")
        if not script:
            raise ValueError("script must contain at least one 2-bit state")
        script_values = [
            self._validate_state_scalar(value, "script entry") for value in script
        ]

        self.base = DiscreteShiftAddFFNBranch(
            self.dim,
            self.hidden_dim,
            activation_bits=self.activation_bits,
            weight_bits=self.weight_bits,
        )
        self.mode_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        BitSliceBooleanLogicExpert(
                            self.dim,
                            self.expert_width,
                            activation_bits=self.activation_bits,
                            weight_bits=self.weight_bits,
                        )
                        for _ in range(self.experts_per_mode)
                    ]
                )
                for _ in range(self.STATE_COUNT)
            ]
        )
        self.state_controller = ShiftAddLinear(
            self.dim,
            self.STATE_COUNT,
            magnitude_bits=self.weight_bits,
            output_bits=max(self.activation_bits, 8),
            input_bits=self.activation_bits,
            bias=False,
        )
        self.output_quantizer = PowerOfTwoActivationQuantizer(self.activation_bits)
        self.register_buffer(
            "static_state_code", torch.tensor(static_state, dtype=torch.int64)
        )
        self.register_buffer(
            "script_codes", torch.tensor(script_values, dtype=torch.int64)
        )

    @classmethod
    def _validate_state_scalar(cls, value: int, name: str) -> int:
        value = int(value)
        if value < 0 or value >= cls.STATE_COUNT:
            raise ValueError(f"{name} must be a 2-bit code in [0,3]")
        return value

    @staticmethod
    def state_bits(state_code: torch.Tensor) -> torch.Tensor:
        """Return little-endian ``[bit0, bit1]`` state payloads."""

        integer = state_code.to(torch.int64)
        if bool(((integer < 0) | (integer > 3)).any()):
            raise ValueError("state_code must contain only 2-bit values in [0,3]")
        return torch.stack(
            (
                torch.bitwise_and(integer, 1),
                torch.bitwise_and(torch.bitwise_right_shift(integer, 1), 1),
            ),
            dim=-1,
        ).to(torch.uint8)

    def _expand_state(
        self, state: int | torch.Tensor, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        tensor = torch.as_tensor(state, device=device)
        if tensor.is_floating_point():
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError("control state must be finite")
            if not bool((tensor == tensor.round()).all()):
                raise ValueError("control state must contain integer 2-bit codes")
        tensor = tensor.to(torch.int64)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch_size)
        elif tensor.ndim == 1 and tensor.shape[0] == batch_size:
            pass
        else:
            raise ValueError("control state must be scalar or have shape [batch]")
        if bool(((tensor < 0) | (tensor >= self.STATE_COUNT)).any()):
            raise ValueError("control state must contain only 2-bit codes in [0,3]")
        return tensor

    def _global_vector(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            # ViT CLS is already a global carrier and avoids an extra general
            # divide that token-mean pooling would require.
            return x[:, 0, :]
        raise ValueError("expected [batch,dim] or [batch,tokens,dim] input")

    def controller_logits(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"expected final dimension {self.dim}, got {x.shape[-1]}")
        return self.state_controller(self._global_vector(x))

    def resolve_hard_state(
        self,
        x: torch.Tensor,
        *,
        control: str = "dynamic",
        control_state: int | torch.Tensor | None = None,
        script_step: int = 0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve a matched control and return ``(state_code, logits)``."""

        if control not in self.CONTROL_MODES:
            raise ValueError(
                f"control must be one of {self.CONTROL_MODES}, got {control!r}"
            )
        batch_size = x.shape[0]
        logits = self.controller_logits(x)
        if control == "dynamic":
            if control_state is not None:
                raise ValueError("dynamic control does not accept control_state")
            state = logits.argmax(dim=-1)
        elif control_state is not None:
            state = self._expand_state(control_state, batch_size, x.device)
        elif control == "static":
            state = self.static_state_code.to(x.device).expand(batch_size)
        elif control == "script":
            step = int(script_step) % int(self.script_codes.numel())
            state = self.script_codes[step].to(x.device).expand(batch_size)
        else:
            if generator is None:
                state = torch.randint(
                    self.STATE_COUNT, (batch_size,), device=x.device
                )
            else:
                generator_device = getattr(generator, "device", torch.device("cpu"))
                state = torch.randint(
                    self.STATE_COUNT,
                    (batch_size,),
                    device=generator_device,
                    generator=generator,
                ).to(x.device)
        return state, logits

    def state_selector(
        self,
        x: torch.Tensor,
        *,
        control: str = "dynamic",
        control_state: int | torch.Tensor | None = None,
        script_step: int = 0,
        generator: torch.Generator | None = None,
        matched_ste: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state, logits = self.resolve_hard_state(
            x,
            control=control,
            control_state=control_state,
            script_step=script_step,
            generator=generator,
        )
        hard = F.one_hot(state, num_classes=self.STATE_COUNT).to(x.dtype)
        if self.training and matched_ste:
            soft = torch.softmax(logits / self.state_temperature, dim=-1)
            return state, _ste(hard, soft)
        return state, hard

    def _mode_output(self, mode: int, x: torch.Tensor) -> torch.Tensor:
        outputs = [expert(x) for expert in self.mode_experts[mode]]
        return torch.stack(outputs, dim=0).sum(0) * float(
            2.0 ** -self.logic_output_shift
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        control: str = "dynamic",
        control_state: int | torch.Tensor | None = None,
        script_step: int = 0,
        generator: torch.Generator | None = None,
        matched_ste: bool = True,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        state, selector = self.state_selector(
            x,
            control=control,
            control_state=control_state,
            script_step=script_step,
            generator=generator,
            matched_ste=matched_ste,
        )
        mode_outputs = torch.stack(
            [self._mode_output(mode, x) for mode in range(self.STATE_COUNT)], dim=1
        )
        selector_shape = [selector.shape[0], self.STATE_COUNT] + [
            1
        ] * (mode_outputs.ndim - 2)
        selected = (mode_outputs * selector.reshape(selector_shape)).sum(dim=1)
        output = self.output_quantizer(self.base(x) + selected)
        if return_state:
            return output, state
        return output

    def matched_control_contract(self) -> dict[str, object]:
        return {
            "controls": list(self.CONTROL_MODES),
            "shared_parameter_object": True,
            "same_four_mode_payload": True,
            "same_state_controller": True,
            "matching_scope": "parameter count and topology only; effective bank updates differ",
            "matched_ste_default": "same controller surrogate, including phantom controller gradients",
            "required_reporting": "state histogram and per-mode sample/update counts",
            "recommended_random_control": (
                "supply recorded [batch] 2-bit control_state for exact paired replay"
            ),
        }

    @torch.no_grad()
    def hard_payload(self) -> dict[str, object]:
        return {
            "base": self.base.hard_payload(),
            "state_controller": _shiftadd_payload(self.state_controller),
            "static_state_code": self.static_state_code.to(torch.uint8),
            "script_codes": self.script_codes.to(torch.uint8),
            "mode_experts": [
                [expert.hard_payload() for expert in bank] for bank in self.mode_experts
            ],
            "logic_output_right_shift": self.logic_output_shift,
            "merge_rounding": (
                f"align power-of-two exponents, integer add, A{self.activation_bits} "
                "requant ties-to-even"
            ),
            "control_comparison_warning": (
                "payloads are parameter/topology matched, not effective-update matched"
            ),
        }

    def deployment_contract(self) -> dict[str, object]:
        return {
            "state_bits": 2,
            "state_count": self.STATE_COUNT,
            "state_scope": "one CLS/global-derived or scripted code per sample",
            "dynamic_state": (
                "shift-add 4-logit controller followed by hard argmax/priority compare"
            ),
            "script_state": "2-bit ROM/counter entry or external 2-bit code",
            "mode_action": (
                "selects four independent Boolean source-connection and LUT payloads; "
                "not a feature rotation"
            ),
            "mode_mux": "2-bit 4-to-1 output mux; inactive modes may be clock-gated",
            "base_branch": self.base.deployment_contract(),
            "experts_per_mode": self.experts_per_mode,
            "boolean_gate_count_total": (
                self.STATE_COUNT * self.experts_per_mode * self.expert_width
            ),
            "per_expert": self.mode_experts[0][0].deployment_contract(),
            "logic_output_scale": f"right shift {self.logic_output_shift}",
            "controls": self.matched_control_contract(),
            "general_multipliers": 0,
            "training_only": (
                "state softmax STE and dense connection-selection surrogates"
            ),
        }


__all__ = [
    "BitSliceBooleanLogicExpert",
    "DiscreteShiftAddFFNBranch",
    "ParallelBitSliceLogicFFN",
    "FourModeStateSelectedFFN",
]
