from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict

import torch
import torch.nn as nn


_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


try:
    from src.packed_xnor_topk import PackedXnorTopK
    from src.selected_column_majority_cuda import SelectorMaskSelectedColumnMajority
except ModuleNotFoundError:
    from packed_xnor_topk import PackedXnorTopK
    from selected_column_majority_cuda import SelectorMaskSelectedColumnMajority


class PackedXnorTopKSelectorMajority(nn.Module):
    """总装模块：q/k -> PackedXnorTopK，再与 v 做 selector-mask majority。"""

    def __init__(
        self,
        k: int,
        validate_input: bool = False,
        use_thermometer_encoding: bool = False,
        n_thresholds: int = 8,
        v_n_thresholds: int | None = None,
        encoding_scale: float = 10.0,
        apply_sigmoid_before_encoding: bool = True,
        decode_output: bool = False,
        boundary_surrogate_temperature: float = 1.0,
        topk_impl: str = "winner-tree",
        topk_forward_mode: str = "topk",
        topk_surrogate_mode: str = "kth",
        topk_surrogate_proxy_source: str = "encoded",
        topk_kth_detach_value: bool = True,
        topk_kth_use_midpoint_theta: bool = False,
        topk_kth_normalize_soft_mask: bool = True,
        majority_train_temperature: float = 0.125,
        fast_vote: bool = False,
        majority_surrogate_mode: str = "count",
        majority_k_root_degree: int = 2,
        thermometer_decode_use_weights: bool = True,
    ):
        super().__init__()
        if k <= 0:
            raise ValueError("k 必须大于 0")
        if n_thresholds <= 0:
            raise ValueError("n_thresholds 必须大于 0")
        if v_n_thresholds is not None and v_n_thresholds <= 0:
            raise ValueError("v_n_thresholds 必须大于 0")
        if encoding_scale <= 0:
            raise ValueError("encoding_scale 必须大于 0")
        if decode_output and not use_thermometer_encoding:
            raise ValueError("decode_output=True 时必须启用 use_thermometer_encoding")

        if topk_surrogate_proxy_source not in ("encoded", "soft-thermometer"):
            raise ValueError("topk_surrogate_proxy_source must be 'encoded' or 'soft-thermometer'")
        if topk_surrogate_proxy_source == "soft-thermometer" and not use_thermometer_encoding:
            raise ValueError("topk_surrogate_proxy_source='soft-thermometer' requires use_thermometer_encoding=True")

        self.k = k
        self.validate_input = validate_input
        self.use_thermometer_encoding = use_thermometer_encoding
        self.n_thresholds = n_thresholds
        self.v_n_thresholds = int(v_n_thresholds) if v_n_thresholds is not None else n_thresholds
        self.encoding_scale = encoding_scale
        self.apply_sigmoid_before_encoding = apply_sigmoid_before_encoding
        self.decode_output = decode_output
        self.topk_surrogate_proxy_source = topk_surrogate_proxy_source
        self.boundary_surrogate_temperature = boundary_surrogate_temperature
        self.majority_train_temperature = majority_train_temperature
        self.fast_vote = bool(fast_vote)
        self.majority_surrogate_mode = majority_surrogate_mode
        self.majority_k_root_degree = int(majority_k_root_degree)
        self.thermometer_decode_use_weights = bool(thermometer_decode_use_weights)

        self.packed_topk = PackedXnorTopK(
            boundary_surrogate_temperature=boundary_surrogate_temperature,
            topk_impl=topk_impl,
            topk_forward_mode=topk_forward_mode,
            topk_surrogate_mode=topk_surrogate_mode,
            topk_kth_detach_value=topk_kth_detach_value,
            topk_kth_use_midpoint_theta=topk_kth_use_midpoint_theta,
            topk_kth_normalize_soft_mask=topk_kth_normalize_soft_mask,
        )
        self.selector_majority = SelectorMaskSelectedColumnMajority(
            validate_input=validate_input,
            train_temperature=majority_train_temperature,
            fast_vote=fast_vote,
            surrogate_mode=majority_surrogate_mode,
            k_root_degree=majority_k_root_degree,
        )

        qk_threshold_positions = torch.linspace(
            1.0 / (n_thresholds + 1),
            n_thresholds / (n_thresholds + 1),
            n_thresholds,
            dtype=torch.float32,
        )
        v_threshold_positions = torch.linspace(
            1.0 / (self.v_n_thresholds + 1),
            self.v_n_thresholds / (self.v_n_thresholds + 1),
            self.v_n_thresholds,
            dtype=torch.float32,
        )
        self.register_buffer("_qk_threshold_positions", qk_threshold_positions, persistent=False)
        self.register_buffer("_v_threshold_positions", v_threshold_positions, persistent=False)

        if decode_output and self.thermometer_decode_use_weights:
            self.decode_logits = nn.Parameter(torch.zeros(self.v_n_thresholds, dtype=torch.float32))
        else:
            self.register_parameter("decode_logits", None)

    def _thermometer_encode_hard_ste(
        self,
        x: torch.Tensor,
        n_thresholds: int,
        threshold_positions: torch.Tensor,
    ) -> torch.Tensor:
        x_input = torch.sigmoid(x) if self.apply_sigmoid_before_encoding else x
        x_expanded = x_input.unsqueeze(-1)

        threshold_positions = threshold_positions.to(device=x.device, dtype=x.dtype)
        view_shape = [1] * x_expanded.ndim
        view_shape[-1] = n_thresholds
        thresholds = threshold_positions.view(*view_shape)

        soft_bits = torch.sigmoid((x_expanded - thresholds) * self.encoding_scale)
        hard_bits = (soft_bits >= 0.5).to(x.dtype)
        # 前向使用硬 0/1，反向通过 soft_bits 传梯度。
        ste_bits = hard_bits + (soft_bits - soft_bits.detach())
        return ste_bits.reshape(*x.shape[:-1], x.shape[-1] * n_thresholds)

    def _thermometer_encode_soft(
        self,
        x: torch.Tensor,
        n_thresholds: int,
        threshold_positions: torch.Tensor,
    ) -> torch.Tensor:
        x_input = torch.sigmoid(x) if self.apply_sigmoid_before_encoding else x
        x_expanded = x_input.unsqueeze(-1)

        threshold_positions = threshold_positions.to(device=x.device, dtype=x.dtype)
        view_shape = [1] * x_expanded.ndim
        view_shape[-1] = n_thresholds
        thresholds = threshold_positions.view(*view_shape)

        soft_bits = torch.sigmoid((x_expanded - thresholds) * self.encoding_scale)
        return soft_bits.reshape(*x.shape[:-1], x.shape[-1] * n_thresholds)

    def _thermometer_decode(self, encoded: torch.Tensor) -> torch.Tensor:
        encoded_width = encoded.shape[-1]
        if encoded_width % self.v_n_thresholds != 0:
            raise ValueError("待解码张量最后一维必须能被 v_n_thresholds 整除")

        base_width = encoded_width // self.v_n_thresholds
        encoded_reshaped = encoded.reshape(*encoded.shape[:-1], base_width, self.v_n_thresholds)
        if self.thermometer_decode_use_weights and self.decode_logits is not None:
            weights = torch.softmax(self.decode_logits, dim=0).to(encoded_reshaped.dtype)
            view_shape = [1] * encoded_reshaped.ndim
            view_shape[-1] = self.v_n_thresholds
            return (encoded_reshaped * weights.view(*view_shape)).sum(dim=-1)
        return encoded_reshaped.mean(dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError("q、k、v 必须具有相同形状")
        if q.ndim < 2:
            raise ValueError("q、k、v 必须至少是二维张量，shape 应为 (..., rows, cols)")

        proxy_scores = None
        if self.use_thermometer_encoding:
            q_logic = self._thermometer_encode_hard_ste(q, self.n_thresholds, self._qk_threshold_positions)
            k_logic = self._thermometer_encode_hard_ste(k, self.n_thresholds, self._qk_threshold_positions)
            if self.topk_surrogate_proxy_source == "soft-thermometer":
                q_proxy = self._thermometer_encode_soft(q, self.n_thresholds, self._qk_threshold_positions)
                k_proxy = self._thermometer_encode_soft(k, self.n_thresholds, self._qk_threshold_positions)
                proxy_scores = self.packed_topk._xnor_similarity_proxy(q_proxy, k_proxy)
            v_logic = self._thermometer_encode_hard_ste(v, self.v_n_thresholds, self._v_threshold_positions)
        else:
            q_logic = q
            k_logic = k
            v_logic = v

        topk_result = self.packed_topk(
            q_logic,
            k_logic,
            topk=self.k,
            return_selector_mask=True,
            return_topk_packed_scores=False,
            proxy_scores=proxy_scores,
        )
        selector_mask = topk_result["selector_mask"]
        if selector_mask is None:
            raise RuntimeError("PackedXnorTopK 未返回 selector_mask")

        encoded_output = self.selector_majority(selector_mask, v_logic)
        if self.use_thermometer_encoding and self.decode_output:
            return self._thermometer_decode(encoded_output.to(torch.float32))
        return encoded_output


class AttentionLogic(nn.Module):
    """按 ViT baseline 的 Attention 结构实现，核心注意力由 PackedXnorTopKSelectorMajority 计算。"""

    def __init__(
        self,
        embed_dim: int = 192,
        num_heads: int = 3,
        k: int = 15,
        validate_input: bool = False,
        use_thermometer_encoding: bool = False,
        n_thresholds: int = 8,
        v_n_thresholds: int | None = None,
        encoding_scale: float = 10.0,
        apply_sigmoid_before_encoding: bool = True,
        decode_output: bool = False,
        boundary_surrogate_temperature: float = 1.0,
        topk_impl: str = "winner-tree",
        topk_forward_mode: str = "topk",
        topk_surrogate_mode: str = "kth",
        topk_surrogate_proxy_source: str = "encoded",
        topk_kth_detach_value: bool = True,
        topk_kth_use_midpoint_theta: bool = False,
        topk_kth_normalize_soft_mask: bool = True,
        majority_train_temperature: float = 0.125,
        fast_vote: bool = False,
        majority_surrogate_mode: str = "count",
        majority_k_root_degree: int = 2,
        thermometer_decode_use_weights: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.logic_attention = PackedXnorTopKSelectorMajority(
            k=k,
            validate_input=validate_input,
            use_thermometer_encoding=use_thermometer_encoding,
            n_thresholds=n_thresholds,
            v_n_thresholds=v_n_thresholds,
            encoding_scale=encoding_scale,
            apply_sigmoid_before_encoding=apply_sigmoid_before_encoding,
            decode_output=decode_output,
            boundary_surrogate_temperature=boundary_surrogate_temperature,
            topk_impl=topk_impl,
            topk_forward_mode=topk_forward_mode,
            topk_surrogate_mode=topk_surrogate_mode,
            topk_surrogate_proxy_source=topk_surrogate_proxy_source,
            topk_kth_detach_value=topk_kth_detach_value,
            topk_kth_use_midpoint_theta=topk_kth_use_midpoint_theta,
            topk_kth_normalize_soft_mask=topk_kth_normalize_soft_mask,
            majority_train_temperature=majority_train_temperature,
            fast_vote=fast_vote,
            majority_surrogate_mode=majority_surrogate_mode,
            majority_k_root_degree=majority_k_root_degree,
            thermometer_decode_use_weights=thermometer_decode_use_weights,
        )
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward_with_intermediates(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("AttentionLogic 的输入必须是三维张量 [batch_size, num_tokens, embed_dim]")
        if x.shape[-1] != self.embed_dim:
            raise ValueError("输入最后一维必须等于 embed_dim")

        batch_size, num_tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        head_logic_output = self.logic_attention(q, k, v).to(torch.float32)
        fused_heads = head_logic_output.transpose(1, 2).reshape(batch_size, num_tokens, channels)
        output = self.proj(fused_heads)

        return {
            "q": q,
            "k": k,
            "v": v,
            "head_logic_output": head_logic_output,
            "fused_heads": fused_heads,
            "output": output,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_intermediates(x)["output"]


# def run_attention_logic_train_eval_soft_cases_demo(
#     batch_size: int = 2,
#     num_tokens: int = 5,
#     embed_dim: int = 12,
#     num_heads: int = 3,
#     k: int = 3,
# ) -> Dict[str, object]:
#     """测试 AttentionLogic 在四种模式下的前向与梯度连通性。"""
#     if embed_dim % num_heads != 0:
#         raise ValueError("embed_dim 必须能被 num_heads 整除")

#     torch.manual_seed(0)
#     x_base = torch.randn(batch_size, num_tokens, embed_dim, dtype=torch.float32)

#     cases = [
#         ("soft_train=True + train", True, True),
#         ("soft_train=False + train", False, True),
#         ("soft_train=True + eval", True, False),
#         ("soft_train=False + eval", False, False),
#     ]

#     results: Dict[str, object] = {}
#     print("=== AttentionLogic 4-Case Check ===")
#     print(
#         f"shape=({batch_size}, {num_tokens}, {embed_dim}), "
#         f"num_heads={num_heads}, k={k}"
#     )

#     for case_name, use_soft_train, train_mode in cases:
#         model = AttentionLogic(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             k=k,
#             use_soft_train=use_soft_train,
#             majority_train_temperature=0.2,
#         )
#         model.train(mode=train_mode)

#         x = x_base.clone().requires_grad_(train_mode)

#         forward_ok = True
#         forward_error = ""
#         output = None
#         try:
#             output = model(x)
#         except RuntimeError as exc:
#             forward_ok = False
#             forward_error = str(exc)

#         backward_ok = None
#         backward_error = ""
#         input_grad_ok = None
#         qkv_grad_ok = None
#         proj_grad_ok = None

#         if train_mode and forward_ok:
#             loss = output.to(torch.float32).mean()
#             backward_ok = True
#             try:
#                 loss.backward()
#             except RuntimeError as exc:
#                 backward_ok = False
#                 backward_error = str(exc)

#             input_grad_ok = x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum().item() > 0
#             qkv_grad_ok = (
#                 model.qkv.weight.grad is not None
#                 and torch.isfinite(model.qkv.weight.grad).all()
#                 and model.qkv.weight.grad.abs().sum().item() > 0
#             )
#             proj_grad_ok = (
#                 model.proj.weight.grad is not None
#                 and torch.isfinite(model.proj.weight.grad).all()
#                 and model.proj.weight.grad.abs().sum().item() > 0
#             )

#         print(f"\n[{case_name}]")
#         print(f"forward_ok={forward_ok}")
#         if not forward_ok:
#             print(f"forward_error={forward_error}")
#         if forward_ok and output is not None:
#             print(f"output_shape={tuple(output.shape)}, output_dtype={output.dtype}")

#         if train_mode:
#             print(f"backward_ok={backward_ok}")
#             if backward_ok is False:
#                 print(f"backward_error={backward_error}")
#             print(f"input_grad_ok={bool(input_grad_ok)}")
#             print(f"qkv_grad_ok={bool(qkv_grad_ok)}")
#             print(f"proj_grad_ok={bool(proj_grad_ok)}")
#             print(f"train_has_grad={bool(input_grad_ok or qkv_grad_ok or proj_grad_ok)}")
#         else:
#             print("eval_mode: backward check skipped")

#         results[case_name] = {
#             "forward_ok": forward_ok,
#             "forward_error": forward_error,
#             "backward_ok": backward_ok,
#             "backward_error": backward_error,
#             "input_grad_ok": input_grad_ok,
#             "qkv_grad_ok": qkv_grad_ok,
#             "proj_grad_ok": proj_grad_ok,
#         }

#     return results


# __all__ = [
#     "AttentionLogic",
#     "PackedXnorTopKSelectorMajority",
#     "PackedXnorTopK",
#     "SelectorMaskSelectedColumnMajority",
#     "run_attention_logic_train_eval_soft_cases_demo",
# ]


# if __name__ == "__main__":
#     run_attention_logic_train_eval_soft_cases_demo()
