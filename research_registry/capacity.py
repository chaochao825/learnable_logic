from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CapacityMetrics:
    input_patch_bits: int
    state_width: int
    bits_per_state_value: int
    state_storage_bits_per_token: int
    tokens: int
    trainable_parameters: int | None
    hard_payload_tensor_bits: int | None
    gate_count: int | None
    logic_depth: int | None
    deployed_depth_lower_bound_excluding_topk: int | None
    fanout_max: int | None
    qk_bits: int | None
    topk: int | None
    message_levels: int | None
    fixed_xnor_per_sample: int | None
    value_count_inputs_per_sample: int | None
    count_threshold_outputs_per_sample: int | None
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parameter_count(model: Any | None) -> int | None:
    if model is None:
        return None
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _optional_model_metric(model: Any | None, name: str) -> int | None:
    if model is None or not hasattr(model, name):
        return None
    value = getattr(model, name)
    return int(value() if callable(value) else value)


def _payload_tensor_bits(value: Any) -> int:
    try:
        import torch
    except ImportError:
        return 0
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size() * 8
    if isinstance(value, dict):
        return sum(_payload_tensor_bits(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_payload_tensor_bits(item) for item in value)
    return 0


def _ceil_log2(value: int) -> int:
    return max(0, (int(value) - 1).bit_length())


def bitstate_capacity(config: Any, model: Any | None = None) -> dict[str, Any]:
    """Return capacity metrics for a persistent Boolean-state model."""

    raw_patch_bits = (
        int(config.in_channels)
        * int(config.patch_size) ** 2
        * int(config.threshold_levels)
    )
    grid = int(config.image_size) // int(config.patch_size)
    message_mode = getattr(config, "message_mode", "majority")
    count_thresholds = tuple(getattr(config, "message_count_thresholds", ()))
    message_levels = 2
    notes = [f"message_mode={message_mode}"]
    if message_mode != "majority":
        message_levels = len(count_thresholds) + 1
        notes.append(f"count_thresholds={count_thresholds}")
    global_depth = int(config.global_depth)
    heads = int(config.heads)
    tokens = 1 + grid * grid
    state_width = int(config.state_width)
    topk = int(config.topk)
    qk_bits = int(config.qk_bits)
    count_output_width = 0
    if message_mode != "majority":
        head_width = state_width // heads
        count_output_width = int(head_width * config.message_count_fraction)
        count_output_width -= count_output_width % len(count_thresholds)
    encoder_depth = (
        _optional_model_metric(getattr(model, "encoder", None), "logic_depth")
        if model is not None
        else (2 if config.encoder_kind == "redundant_predicate" else 1)
    )
    per_global_depth = (
        1
        + 1
        + _ceil_log2(qk_bits)
        + _ceil_log2(topk)
        + 1
        + 1
    )
    deployed_depth = (
        int(encoder_depth)
        + int(config.local_depth)
        + global_depth * per_global_depth
        + 1
    )
    payload_bits = None
    if model is not None and hasattr(model, "deployment_payload"):
        payload_bits = _payload_tensor_bits(model.deployment_payload())

    metrics = CapacityMetrics(
        input_patch_bits=raw_patch_bits,
        state_width=state_width,
        bits_per_state_value=1,
        state_storage_bits_per_token=state_width,
        tokens=tokens,
        trainable_parameters=_parameter_count(model),
        hard_payload_tensor_bits=payload_bits,
        gate_count=_optional_model_metric(model, "gate_count"),
        logic_depth=_optional_model_metric(model, "logic_depth"),
        deployed_depth_lower_bound_excluding_topk=deployed_depth,
        fanout_max=_optional_model_metric(model, "fanout_max"),
        qk_bits=qk_bits,
        topk=topk,
        message_levels=message_levels,
        fixed_xnor_per_sample=(
            global_depth * heads * tokens * tokens * qk_bits
        ),
        value_count_inputs_per_sample=(
            global_depth * tokens * state_width * topk
        ),
        count_threshold_outputs_per_sample=(
            global_depth * tokens * heads * count_output_width
        ),
        notes=tuple(notes),
    )
    return metrics.to_dict()


def full_discrete_capacity(model: Any) -> dict[str, Any]:
    """Return model-wide storage and routing capacity for FullDiscreteViT."""

    patch = model.patch_embed
    projection = patch.projection
    dim = int(projection.out_features)
    input_values = int(projection.in_features)
    input_bits = int(getattr(patch.input_quantizer, "bits", 8))
    blocks = list(model.blocks)
    attention = blocks[0].attn if blocks else None
    dense_shapes = [
        (int(module.in_features), int(module.out_features))
        for module in model.modules()
        if module.__class__.__name__ == "ShiftAddLinear"
    ]
    dense_fanin_max = max((shape[0] for shape in dense_shapes), default=0)
    dense_fanout_upper_bound = max(
        (shape[1] for shape in dense_shapes), default=0
    )
    metrics = CapacityMetrics(
        input_patch_bits=input_values * input_bits,
        state_width=dim,
        bits_per_state_value=int(model.activation_bits),
        state_storage_bits_per_token=dim * int(model.activation_bits),
        tokens=int(patch.num_patches) + 1,
        trainable_parameters=_parameter_count(model),
        hard_payload_tensor_bits=None,
        gate_count=None,
        logic_depth=len(blocks),
        deployed_depth_lower_bound_excluding_topk=None,
        fanout_max=dense_fanout_upper_bound,
        qk_bits=(
            int(attention.head_dim * attention.qk_lanes)
            if attention is not None
            else None
        ),
        topk=int(attention.topk) if attention is not None else None,
        message_levels=256,
        fixed_xnor_per_sample=(
            len(blocks)
            * int(attention.heads)
            * int(patch.num_patches + 1) ** 2
            * int(attention.head_dim * attention.qk_lanes)
            if attention is not None
            else None
        ),
        value_count_inputs_per_sample=(
            len(blocks)
            * int(patch.num_patches + 1)
            * dim
            * int(attention.topk)
            if attention is not None
            else None
        ),
        count_threshold_outputs_per_sample=0,
        notes=(
            "A8 code plus signed integer exponent per quantization group",
            "hard V aggregation retains integer magnitude instead of majority-only state",
            f"dense structural fan-in maximum={dense_fanin_max}",
            "fanout is a dense structural upper bound before zero-weight pruning",
            "transformer-block depth is not total synthesized Boolean depth",
        ),
    )
    return metrics.to_dict()
