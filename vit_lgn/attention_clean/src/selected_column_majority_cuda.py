from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn


AccumulationDType = Literal["auto", "float16", "float32"]


_MAJORITY_LUT_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _validate_binary_tensor(x: torch.Tensor, name: str) -> None:
    is_binary = torch.all((x == 0) | (x == 1))
    if not bool(is_binary.item()):
        raise ValueError(f"{name} 必须是 0/1 张量")


def _resolve_accumulation_dtype(device: torch.device, accumulation_dtype: AccumulationDType) -> torch.dtype:
    if accumulation_dtype == "float16":
        return torch.float16
    if accumulation_dtype == "float32":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def _validate_selector_and_votes(selector: torch.Tensor, votes: torch.Tensor, validate_input: bool) -> tuple[int, int, int]:
    if selector.ndim < 2:
        raise ValueError("selector 必须至少是二维张量，shape 应为 (..., output_rows, cols)")
    if votes.ndim < 2:
        raise ValueError("votes 必须至少是二维张量，shape 应为 (..., vote_rows, cols)")
    if selector.shape[:-2] != votes.shape[:-2]:
        raise ValueError("selector 和 votes 的前缀 batch 维必须一致")

    output_rows, selector_cols = selector.shape[-2:]
    vote_rows, vote_cols = votes.shape[-2:]
    if selector_cols != vote_rows:
        raise ValueError("selector 的列数必须等于 votes 的行数")
    if selector_cols == 0 or vote_rows == 0 or vote_cols == 0:
        raise ValueError("selector 和 votes 的行列维不能为空")

    if validate_input:
        _validate_binary_tensor(selector, "selector")
        _validate_binary_tensor(votes, "votes")

    return output_rows, vote_rows, vote_cols


def _validate_topk_indices_and_votes(topk_indices: torch.Tensor, votes: torch.Tensor, validate_input: bool) -> tuple[int, int, int]:
    if topk_indices.ndim < 2:
        raise ValueError("topk_indices 必须至少是二维张量，shape 应为 (..., output_rows, k)")
    if votes.ndim < 2:
        raise ValueError("votes 必须至少是二维张量，shape 应为 (..., vote_rows, vote_cols)")
    if topk_indices.shape[:-2] != votes.shape[:-2]:
        raise ValueError("topk_indices 和 votes 的前缀 batch 维必须一致")

    output_rows, topk = topk_indices.shape[-2:]
    vote_rows, vote_cols = votes.shape[-2:]
    if topk == 0:
        raise ValueError("topk_indices 的 k 维不能为空")
    if vote_rows == 0 or vote_cols == 0:
        raise ValueError("votes 的行列维不能为空")

    if validate_input:
        _validate_binary_tensor(votes, "votes")

    return output_rows, vote_rows, vote_cols


class SelectedColumnMajorityCudaMatmul(nn.Module):
    """用 batched matmul 计算 selector-mask majority，避免 5 维 selected_votes 展开。"""

    def __init__(self, accumulation_dtype: AccumulationDType = "auto", validate_input: bool = False):
        super().__init__()
        self.accumulation_dtype = accumulation_dtype
        self.validate_input = validate_input

    def forward(self, selector: torch.Tensor, votes: torch.Tensor) -> torch.Tensor:
        output_rows, vote_rows, vote_cols = _validate_selector_and_votes(selector, votes, self.validate_input)

        selector_bool = selector.to(torch.bool)
        votes_bool = votes.to(torch.bool)
        selected_count = selector_bool.sum(dim=-1, keepdim=True, dtype=torch.int32)
        if torch.any(selected_count == 0):
            raise ValueError("selector 的每一行至少要有一个 1，才能执行多数投票")
        first_count = selected_count.reshape(-1)[0]
        if torch.any(selected_count != first_count):
            raise ValueError("selector 的每一行 1 的个数必须一致（固定 k）")

        prefix_shape = selector_bool.shape[:-2]
        batch_count = math.prod(prefix_shape) if prefix_shape else 1
        matmul_dtype = _resolve_accumulation_dtype(selector.device, self.accumulation_dtype)

        selector_3d = selector_bool.reshape(batch_count, output_rows, vote_rows).to(matmul_dtype)
        votes_3d = votes_bool.reshape(batch_count, vote_rows, vote_cols).to(matmul_dtype)
        ones_count = torch.bmm(selector_3d, votes_3d).reshape(*prefix_shape, output_rows, vote_cols)

        threshold = ((selected_count + 1) // 2).to(matmul_dtype)
        return ones_count >= threshold


class TopKIndicesSelectedColumnMajorityCuda(nn.Module):
    """利用 top-k provenance indices 直接 gather 选中行，再用位打包 + LUT 做多数投票。"""

    def __init__(self, validate_input: bool = False):
        super().__init__()
        # self.accumulation_dtype = accumulation_dtype
        self.validate_input = validate_input

    @staticmethod
    def _cache_key(k: int, device: torch.device) -> tuple[int, str]:
        return k, f"{device.type}:{device.index}"

    def _get_majority_lut(self, k: int, device: torch.device) -> torch.Tensor:
        if k > 30:
            raise ValueError("TopKIndicesSelectedColumnMajorityCuda 当前只支持 k <= 30，避免 LUT 规模失控")

        cache_key = self._cache_key(k, device)
        cached = _MAJORITY_LUT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        threshold = (k + 1) // 2
        lut_values = [1 if pattern.bit_count() >= threshold else 0 for pattern in range(1 << k)]
        lut = torch.tensor(lut_values, dtype=torch.bool, device=device)
        _MAJORITY_LUT_CACHE[cache_key] = lut
        return lut

    @staticmethod
    def _pack_vote_patterns(gathered_votes: torch.Tensor) -> torch.Tensor:
        k = gathered_votes.shape[-1]
        packed = torch.zeros(gathered_votes.shape[:-1], dtype=torch.int32, device=gathered_votes.device)
        for bit_offset in range(k):
            bit = gathered_votes[..., bit_offset].to(torch.int32)
            packed = torch.bitwise_or(packed, torch.bitwise_left_shift(bit, bit_offset))
        return packed

    def forward(self, topk_indices: torch.Tensor, votes: torch.Tensor) -> torch.Tensor:
        output_rows, vote_rows, vote_cols = _validate_topk_indices_and_votes(topk_indices, votes, self.validate_input)

        if topk_indices.dtype not in {torch.int32, torch.int64, torch.long}:
            topk_indices = topk_indices.to(torch.long)
        if torch.any(topk_indices < 0) or torch.any(topk_indices >= vote_rows):
            raise ValueError("topk_indices 中存在越界下标")

        votes_bool = votes.to(torch.bool)
        prefix_shape = topk_indices.shape[:-2]
        k = topk_indices.shape[-1]
        batch_count = math.prod(prefix_shape) if prefix_shape else 1

        indices_3d = topk_indices.reshape(batch_count, output_rows, k).to(torch.long)
        votes_3d = votes_bool.reshape(batch_count, vote_rows, vote_cols)
        batch_index = torch.arange(batch_count, device=votes.device).view(batch_count, 1, 1)
        gathered_votes = votes_3d[batch_index, indices_3d].permute(0, 1, 3, 2).contiguous()

        packed_patterns = self._pack_vote_patterns(gathered_votes)
        majority_lut = self._get_majority_lut(k, votes.device)
        majority = majority_lut[packed_patterns.to(torch.long)]
        return majority.reshape(*prefix_shape, output_rows, vote_cols)


class SelectorMaskSelectedColumnMajority(nn.Module):
    """接收 selector-mask + votes，内部转 topk_indices 后复用 TopKIndicesSelectedColumnMajorityCuda。"""

    def __init__(
        self,
        validate_input: bool = False,
        train_temperature: float = 0.125,
        fast_vote: bool = False,
        surrogate_mode: str = "count",
        k_root_degree: int = 2,
    ):
        super().__init__()
        self.validate_input = validate_input
        self.train_temperature = float(train_temperature)
        self.fast_vote = bool(fast_vote)
        if surrogate_mode not in ("count", "fraction"):
            raise ValueError("surrogate_mode must be 'count' or 'fraction'")
        self.surrogate_mode = surrogate_mode
        if int(k_root_degree) <= 0:
            raise ValueError("k_root_degree must be a positive integer")
        self.k_root_degree = int(k_root_degree)
        if self.train_temperature <= 0:
            raise ValueError("train_temperature 必须大于 0")
        self.index_majority = TopKIndicesSelectedColumnMajorityCuda(
            # accumulation_dtype=accumulation_dtype,
            validate_input=validate_input,
        )
        self.matmul_majority = SelectedColumnMajorityCudaMatmul(
            validate_input=validate_input,
        )

    @staticmethod
    def _build_ste_selector(selector: torch.Tensor, selector_bool: torch.Tensor) -> torch.Tensor:
        hard_selector = selector_bool.to(selector.dtype if selector.is_floating_point() else torch.float32)
        if not selector.is_floating_point():
            return hard_selector
        # 前向保持 hard mask，反向对 selector 走恒等梯度（straight-through）。
        return selector + (hard_selector - selector).detach()

    @staticmethod
    def _selector_mask_to_indices(selector_bool: torch.Tensor, k: int) -> torch.Tensor:
        flat_selector = selector_bool.reshape(-1, selector_bool.shape[-1])
        nz = torch.nonzero(flat_selector, as_tuple=False)
        row_count = flat_selector.shape[0]

        if nz.shape[0] != row_count * k:
            raise ValueError("selector 的每一行必须恰好有固定 k 个 1")

        # nonzero 在二维输入上按行优先返回，第二列就是每行被选中的列下标。
        col_indices = nz[:, 1]
        topk_indices = col_indices.reshape(row_count, k)
        return topk_indices.reshape(*selector_bool.shape[:-1], k).to(torch.long)

    def forward(self, selector: torch.Tensor, votes: torch.Tensor) -> torch.Tensor:
        output_rows, vote_rows, _ = _validate_selector_and_votes(selector, votes, self.validate_input)

        selector_bool = selector.to(torch.bool)
        votes_bool = votes.to(torch.bool)

        selected_count = selector_bool.sum(dim=-1)
        if torch.any(selected_count == 0):
            raise ValueError("selector 的每一行至少要有一个 1，才能执行多数投票")

        # 这个包装器要求每行选择数量固定，否则无法用统一的 k 维 topk_indices 表示。
        first_count = selected_count.reshape(-1)[0]
        if torch.any(selected_count != first_count):
            raise ValueError("selector 的每一行 1 的个数必须一致（固定 k）")

        if self.fast_vote:
            hard_majority = self.matmul_majority(selector_bool, votes_bool)
        else:
            k = int(first_count.item())
            topk_indices = self._selector_mask_to_indices(selector_bool, k)
            topk_indices = topk_indices.reshape(*selector.shape[:-2], output_rows, k)
            hard_majority = self.index_majority(topk_indices, votes_bool)

        # 评估态始终走纯硬推理。
        if not self.training:
            return hard_majority

        if not torch.is_grad_enabled() or not selector.requires_grad:
            return hard_majority

        prefix_shape = selector_bool.shape[:-2]
        batch_count = math.prod(prefix_shape) if prefix_shape else 1
        vote_cols = votes.shape[-1]

        selector_ste = self._build_ste_selector(selector, selector_bool)
        selector_ste_3d = selector_ste.reshape(batch_count, output_rows, vote_rows).to(torch.float32)
        votes_3d = votes.to(torch.float32).reshape(batch_count, vote_rows, vote_cols)
        soft_ones_count = torch.bmm(selector_ste_3d, votes_3d).reshape(*prefix_shape, output_rows, vote_cols)

        selected_count_float = selected_count.to(torch.float32).reshape(*prefix_shape, output_rows, 1)
        if self.surrogate_mode == "fraction":
            # 这与先转成 fraction 再除以 (tau / sqrt(k)) 数学等价：
            # ((count / k) - ((k + 1) / (2k))) / (tau / k^(1 - 1/n))
            # = (count - ((k + 1) / 2)) / (tau * k^(1/n))
            # 直接在 count 空间写更直观，也更方便分析 k 对 logit 的影响。
            threshold = (selected_count_float + 1.0) / 2.0
            root_scale = torch.pow(selected_count_float, 1.0 / float(self.k_root_degree))
            effective_temperature = self.train_temperature * root_scale
            soft_majority = torch.sigmoid((soft_ones_count - threshold) / effective_temperature)
        else:
            # 兼容旧行为：在计数空间上做硬前向 + soft 反向代理。
            threshold = (selected_count_float + 1.0) / 2.0
            soft_majority = torch.sigmoid((soft_ones_count - threshold) / self.train_temperature)

        hard_majority_float = hard_majority.to(soft_majority.dtype)
        return hard_majority_float + (soft_majority - soft_majority.detach())


# def run_simple_demo_5x5_5x6() -> dict[str, torch.Tensor | None]:
#     """最小示例：对比 soft_train=True/False 在 train/eval 下的前向与梯度行为。"""

#     selector_base = torch.tensor(
#         [
#             [1, 1, 1, 0, 0],
#             [0, 1, 1, 1, 0],
#             [0, 0, 1, 1, 1],
#             [0, 1, 0, 1, 1],
#             [1, 0, 0, 1, 1],
#         ],
#         dtype=torch.float32,
#     )

#     votes = torch.tensor(
#         [
#             [1, 0, 1, 0, 1, 0],
#             [1, 1, 0, 0, 1, 0],
#             [0, 1, 1, 0, 0, 1],
#             [0, 0, 1, 1, 0, 1],
#             [1, 0, 0, 1, 1, 1],
#         ],
#         dtype=torch.bool,
#     )

#     print("selector shape:", tuple(selector_base.shape))
#     print("votes shape:", tuple(votes.shape))

#     def _run_case(case_name: str, use_soft_train: bool, train_mode: bool, temperature: float):
#         selector = selector_base.clone().requires_grad_(True)
#         module = SelectorMaskSelectedColumnMajority(
#             validate_input=True,
#             use_soft_train=use_soft_train,
#             train_temperature=temperature,
#         )
#         module.train(mode=train_mode)

#         output = module(selector, votes)
#         is_binary = bool(torch.all((output == 0) | (output == 1)).item())

#         loss_value = None
#         if train_mode and output.is_floating_point() and output.requires_grad:
#             loss = output.float().mean()
#             loss.backward()
#             loss_value = float(loss.detach().item())

#         grad = selector.grad.detach().clone() if selector.grad is not None else None
#         grad_nonzero = bool(grad is not None and grad.abs().sum().item() > 0)

#         print(f"\n=== {case_name} ===")
#         print(f"use_soft_train={use_soft_train}, train_mode={train_mode}, temperature={temperature}")
#         print("output shape:", tuple(output.shape))
#         print("output dtype:", output.dtype)
#         print("output binary:", is_binary)
#         print("forward output:")
#         print(output)
#         if loss_value is not None:
#             print("loss:", loss_value)
#         print("selector.grad is None:", grad is None)
#         print("selector.grad nonzero:", grad_nonzero)

#         return output.detach(), grad

#     outputs: dict[str, torch.Tensor | None] = {}

#     outputs["soft_train_train_output"], outputs["soft_train_train_grad"] = _run_case(
#         case_name="soft_train=True + train",
#         use_soft_train=True,
#         train_mode=True,
#         temperature=0.2,
#     )
#     outputs["hard_train_output"], outputs["hard_train_grad"] = _run_case(
#         case_name="soft_train=False + train",
#         use_soft_train=False,
#         train_mode=True,
#         temperature=0.2,
#     )
#     outputs["soft_train_eval_output"], outputs["soft_train_eval_grad"] = _run_case(
#         case_name="soft_train=True + eval",
#         use_soft_train=True,
#         train_mode=False,
#         temperature=0.2,
#     )
#     outputs["hard_eval_output"], outputs["hard_eval_grad"] = _run_case(
#         case_name="soft_train=False + eval",
#         use_soft_train=False,
#         train_mode=False,
#         temperature=0.2,
#     )

#     soft_eval = outputs["soft_train_eval_output"]
#     hard_eval = outputs["hard_eval_output"]
#     eval_same = bool(torch.equal(soft_eval, hard_eval))
#     print("\n=== Eval Consistency ===")
#     print("soft_train=True eval equals soft_train=False eval:", eval_same)

#     return outputs


# __all__ = [
#     "AccumulationDType",
#     "SelectedColumnMajorityCudaMatmul",
#     "TopKIndicesSelectedColumnMajorityCuda",
#     "SelectorMaskSelectedColumnMajority",
#     "run_simple_demo_5x5_5x6",
# ]


# if __name__ == "__main__":
#     run_simple_demo_5x5_5x6()
