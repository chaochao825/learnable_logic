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


_WINNER_TREE_CACHE: Dict[str, Dict[tuple[int, str], object]] = {
    "replay_parent_slots": {},
    "replay_left_child_slots": {},
    "replay_right_child_slots": {},
}


def _cache_tensor(name: str, key: tuple[int, str], build_fn) -> object:
    cached = _WINNER_TREE_CACHE[name].get(key)
    if cached is not None:
        return cached
    cached = build_fn()
    _WINNER_TREE_CACHE[name][key] = cached
    return cached


def _winner_tree_cache_key(n_logical: int, device: torch.device) -> tuple[int, str]:
    return n_logical, f"{device.type}:{device.index}"


def _build_winner_tree_replay_slots(n_logical: int) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    level_count = n_logical.bit_length()
    replay_parent_slots: list[list[int]] = []
    replay_left_child_slots: list[list[int]] = []
    replay_right_child_slots: list[list[int]] = []

    for leaf_pos in range(n_logical):
        parent_slots_for_leaf = []
        left_slots_for_leaf = []
        right_slots_for_leaf = []
        for level_idx in range(1, level_count):
            parent_slot = leaf_pos >> level_idx
            left_child_slot = parent_slot << 1
            right_child_slot = left_child_slot + 1
            parent_slots_for_leaf.append(parent_slot)
            left_slots_for_leaf.append(left_child_slot)
            right_slots_for_leaf.append(right_child_slot)

        replay_parent_slots.append(parent_slots_for_leaf)
        replay_left_child_slots.append(left_slots_for_leaf)
        replay_right_child_slots.append(right_slots_for_leaf)

    return replay_parent_slots, replay_left_child_slots, replay_right_child_slots


def _get_winner_tree_replay_parent_slots(n_logical: int, device: torch.device) -> torch.Tensor:
    cache_key = _winner_tree_cache_key(n_logical, device)
    return _cache_tensor(
        "replay_parent_slots",
        cache_key,
        lambda: torch.tensor(_build_winner_tree_replay_slots(n_logical)[0], dtype=torch.long, device=device),
    )


def _get_winner_tree_replay_left_child_slots(n_logical: int, device: torch.device) -> torch.Tensor:
    cache_key = _winner_tree_cache_key(n_logical, device)
    return _cache_tensor(
        "replay_left_child_slots",
        cache_key,
        lambda: torch.tensor(_build_winner_tree_replay_slots(n_logical)[1], dtype=torch.long, device=device),
    )


def _get_winner_tree_replay_right_child_slots(n_logical: int, device: torch.device) -> torch.Tensor:
    cache_key = _winner_tree_cache_key(n_logical, device)
    return _cache_tensor(
        "replay_right_child_slots",
        cache_key,
        lambda: torch.tensor(_build_winner_tree_replay_slots(n_logical)[2], dtype=torch.long, device=device),
    )


class PackBitsToUint8(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bit_count = x.shape[-1]
        if bit_count == 0:
            shape = list(x.shape)
            shape[-1] = 0
            return torch.empty(shape, dtype=torch.uint8, device=x.device)
        #这需要算出到底有多少个chunk，使用+7的方式，相当于是除8向上取整（因为//是整除向下取整）
        chunk_count = (bit_count + 7) // 8
        padded_bits = chunk_count * 8 #
        if padded_bits != bit_count:
            pad_shape = list(x.shape)
            pad_shape[-1] = padded_bits - bit_count
            pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], dim=-1)

        #这里将x进行打包，首先转换为uint8类型，然后按照最后一个维度的大小进行重塑，新的形状是原来的形状除了最后一个维度变成chunk_count，最后一个维度变成8。也就是说，每8个bit被分成一组，形成一个新的维度。这样做的目的是为了方便后续将每组8个bit打包成一个uint8数值。
        chunks = x.to(torch.uint8).reshape(*x.shape[:-1], chunk_count, 8)
        packed = chunks[..., 0] #这句是什么意思呢？这句代码的意思是从重塑后的张量中取出最后一个维度的第0个元素，作为初始的packed值。因为每组8个bit被分成一组，所以chunks[..., 0]就是每组的第一个bit。接下来通过循环将剩余的7个bit依次左移并与当前的packed值进行按位或运算，最终得到每组8个bit打包成一个uint8数值的结果。
        for offset in range(1, 8):
            packed = torch.bitwise_or(torch.bitwise_left_shift(packed, 1), chunks[..., offset])#将原初的packed值左移一位，然后后面补0，然新的数也就是编号为offset的数被取出来，然后这个数由于是int8，而且只有0和1，所以前面都是0，最后一位是这个数，按位或就可以合并。
        return packed #最终的的张量为[...,chunk_count]的形状，每个元素是一个uint8数值，表示原来8个bit的打包结果。


class UnpackUint8Bits(nn.Module):
    def forward(self, x: torch.Tensor, bit_width: int) -> torch.Tensor:
        if x.dtype != torch.uint8:
            raise ValueError("x 必须是 uint8 packed 张量")
        if bit_width < 0:
            raise ValueError("bit_width 不能小于 0")
        if x.shape[-1] == 0:
            shape = list(x.shape)
            shape[-1] = bit_width
            return torch.empty(shape, dtype=torch.bool, device=x.device)

        bit_offsets = torch.arange(7, -1, -1, device=x.device, dtype=torch.uint8)#创建一个tensor，是76543210，一共8个数
        expanded = x.unsqueeze(-1)#扩展x的维度，变成[...,1]，从[[176, 64]]变为了[[[176],[64]]]
        bits = torch.bitwise_and(torch.bitwise_right_shift(expanded, bit_offsets), 1).to(torch.bool)
        #这个对最后一维进行计算，左移一个list相当于是对最后一维广播，原来如果是按照上面的例子是[1,2,1]，变成[1,2,8]，
        #然后最后一维的每一个元素左移n位，n是list中的对应值，然后再与1进行按位与
        flat_bits = bits.reshape(*x.shape[:-1], x.shape[-1] * 8) #将最后两个维度合并，原来是[[[1],[0]]]变成[[1,0]]
        return flat_bits[..., :bit_width]#裁掉补掉的位数，变回原来的bit_width


class PackedBytePopcount(nn.Module):
    def __init__(self):
        super().__init__()
        lut = torch.tensor([bin(index).count("1") for index in range(256)], dtype=torch.uint8)#构建一个256长度的list，里面每个元素索引编号二进制的1的个数
        self.register_buffer("lut", lut, persistent=False)#这是干啥呢？这是将lut注册为一个buffer，意味着它是模型的一部分，但不会被视为模型的参数，也不会在训练过程中更新。persistent=False表示这个buffer在保存和加载模型时不会被包含在内，这通常用于那些不需要持久化的中间计算结果或常量数据

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.uint8:
            raise ValueError("x 必须是 uint8 packed 张量")
        return self.lut[x.to(torch.long)].to(torch.int16)#由于加法树可能会超出int8，因此要变为int16
        #这里对每一个元素进行索引操作，将每个uint8数值作为索引，从lut中取出对应的1的个数，得到一个新的张量，形状与输入相同，但元素类型是int16，表示每个uint8数值中1的个数。

class LutAdderTree(nn.Module): #
    """用查找表做 pairwise 加法树归约，替代显式 sum(dim=-1)。"""

    def __init__(self):
        super().__init__()
        values = torch.arange(256, dtype=torch.int16)#创建一个256长度的list，里面每个元素是0-255
        add_lut = values.unsqueeze(1) + values.unsqueeze(0)#构建一个256x256的加法查找表，add_lut[i, j] = i + j，其中i和j分别是0-255的整数。这个表可以用来快速计算两个uint8数值的和，而不需要进行实际的加法运算。
        self.register_buffer("add_lut", add_lut, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            raise ValueError("x 至少需要一个维度")
        if x.shape[-1] == 0:
            raise ValueError("最后一个维度不能为空")
        if x.dtype != torch.int16:
            raise ValueError("x 必须是 int16 张量")

        current = x
        while current.shape[-1] > 1: #相当于是对最后一个维度做归一加法，最终最后一个维度只有一个值就是最终有几个1
            if current.shape[-1] & 1:#对最后一维是否为奇数进行检查，按位与，1的其他位都是0，最后一维是1，如果奇数那么其二进制的最后一位也是1
                pad_shape = list(current.shape)
                pad_shape[-1] = 1 #相当于补一个数，就是其余的形状都一样，然后便于拼接
                pad = torch.zeros(pad_shape, dtype=current.dtype, device=current.device)#补0
                current = torch.cat([current, pad], dim=-1)


            left = current[..., 0::2]
            right = current[..., 1::2]
            current = self.add_lut[left.to(torch.long), right.to(torch.long)]
            #由于是循环执行相当于这里对最后一维的奇偶进行加法，然后进行规约求和
        return current[..., 0]#将最后一维去掉，取最后一维唯一一个元素


class PackedTransposedXnorMatrixProduct(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod#这个函数逻辑属于这个类，但它本身不依赖类实例
    def _last_byte_mask(bit_width: int, device: torch.device) -> torch.Tensor | None:
        """
        生成掩码，用于在 bit_width 不是 8 的倍数时，生成掩码，最后一个int8多余补的部分为0，其余为1。
        """
        remainder = bit_width % 8
        if remainder == 0:
            return None
        mask_value = ((1 << remainder) - 1) << (8 - remainder)
        #(1 << remainder) - 1会生成一个二进制数，其中只有低remainder位是1，其余位是0。例如，如果remainder是3，那么(1 << 3) - 1会得到0b111，即十进制的7。接着，这个值被左移(8 - remainder)位，以将有效的bit位移动到字节的高位。例如，如果remainder是3，那么这个值会被左移5位，得到0b11100000，即十进制的224。最终，这个掩码可以用于对最后一个字节进行按位与操作，以保留有效的bit位并清除无效的bit位。
        return torch.tensor(mask_value, dtype=torch.uint8, device=device)

    def forward(self, packed_a: torch.Tensor, packed_b: torch.Tensor, bit_width: int) -> torch.Tensor:
        """
        对 packed q/k 执行矩阵乘法式的行列配对 XNOR。

        输入:
        - packed_a.shape == (..., rows, packed_bytes)
        - packed_b.shape == (..., rows, packed_bytes)

        处理方式:
        - packed_a.unsqueeze(-2) -> (..., rows, 1, packed_bytes)
        - packed_b.unsqueeze(-3) -> (..., 1, rows, packed_bytes)
        - 广播后得到 (..., rows, rows, packed_bytes)

        因此输出的每个矩阵元素 [i, j] 都是 q 的第 i 行与 k 的第 j 行
        做逐位 XNOR 后得到的 packed 01 串，这与原始 attention 逻辑里
        “按矩阵乘法方式用 q 对 k^T 做两两配对”的语义一致。
        """
        if packed_a.ndim < 2 or packed_b.ndim < 2:
            raise ValueError("packed_a 和 packed_b 都必须至少是二维张量")
        if packed_a.shape != packed_b.shape:
            raise ValueError("packed_a 和 packed_b 必须具有相同形状")
        if packed_a.dtype != torch.uint8 or packed_b.dtype != torch.uint8:
            raise ValueError("packed_a 和 packed_b 必须是 uint8")

        expanded_a = packed_a.unsqueeze(-2)
        expanded_b = packed_b.unsqueeze(-3)
        packed_xnor = torch.bitwise_not(torch.bitwise_xor(expanded_a, expanded_b))

        last_mask = self._last_byte_mask(bit_width, packed_xnor.device)
        if last_mask is not None:
            packed_xnor = packed_xnor.clone() #这句话是啥意思？由于 packed_xnor 可能是输入的一个视图，如果直接修改它可能会影响到原始输入张量。通过调用 clone() 方法，创建了一个新的张量，这样后续对 packed_xnor 的修改就不会影响到原始输入张量了。
            packed_xnor[..., -1] = torch.bitwise_and(packed_xnor[..., -1], last_mask) #如果 bit_width 不是 8 的倍数，那么最后一个字节中可能有一些无效的 bit 位。通过与 last_mask 进行按位与操作，可以将这些无效的 bit 位清零，确保最终的 packed_xnor 张量中的每个元素都只包含有效的 bit 位。
        return packed_xnor


class PackedXnorTopK(nn.Module):
    """
    批量 XNOR 矩阵乘法式 q 与 k^T 行列配对，并基于逐位 XNOR 矩阵的每行做 top-k。

    语义说明：
    - 输入 q/k 为 float32 或 bool 的 0/1 张量，shape 为 (..., rows, bit_width)
    - 先按最后一维打包成 uint8
    - 按矩阵乘法式的 q 与 k^T 行列配对做批量 XNOR，得到形状 (..., rows, rows, packed_bytes)
    - 输出矩阵中的每个元素都是一个 packed 01 串，表示某个 q_i 与 k_j 的逐位 XNOR 结果
    - 对结果矩阵的每一行沿“候选 k 行”这一维做 winner-tree top-k，得到 selector mask
    - 用 packed popcount 得到每个 query/key 对的匹配 1 的个数
    - 原始链路里先把 01 串排序，再通过 tournament 比较其大小；对二值串来说，这一步等价于比较 1 的个数，因此这里直接基于 popcount 做 top-k
    - 输出 selector_mask 仍保持与现有链路兼容的 bool / float 形式
    """

    def __init__(
        self,
        boundary_surrogate_temperature: float = 1.0,
        topk_impl: str = "winner-tree",
        topk_forward_mode: str = "topk",
        topk_surrogate_mode: str = "kth",
        topk_kth_detach_value: bool = True,
        topk_kth_use_midpoint_theta: bool = False,
        topk_kth_normalize_soft_mask: bool = True,
    ):
        super().__init__()
        self.pack_bits = PackBitsToUint8()
        self.unpack_bits = UnpackUint8Bits()
        self.packed_xnor = PackedTransposedXnorMatrixProduct()
        self.byte_popcount = PackedBytePopcount()
        self.adder_tree = LutAdderTree()

        self.boundary_surrogate_temperature = float(boundary_surrogate_temperature)
        if topk_impl not in ("winner-tree", "torch-topk"):
            raise ValueError("topk_impl 只能是 'winner-tree' 或 'torch-topk'")
        self.topk_impl = topk_impl
        if topk_forward_mode not in ("topk", "random-k", "fixed-random-k", "dense"):
            raise ValueError(
                "topk_forward_mode must be 'topk', 'random-k', "
                "'fixed-random-k', or 'dense'"
            )
        self.topk_forward_mode = topk_forward_mode
        if topk_surrogate_mode not in ("kth", "random-kth", "soft-rank", "sigmoid-topk", "subset-gibbs"):
            raise ValueError("topk_surrogate_mode 目前只支持 'kth'、'random-kth'、'soft-rank'、'sigmoid-topk' 或 'subset-gibbs'")
        self.topk_surrogate_mode = topk_surrogate_mode
        self.topk_kth_detach_value = bool(topk_kth_detach_value)
        self.topk_kth_use_midpoint_theta = bool(topk_kth_use_midpoint_theta)
        self.topk_kth_normalize_soft_mask = bool(topk_kth_normalize_soft_mask)
        # shape: [row_count, k]，同一个 PackedXnorTopK（即同一个 block 内）固定复用
        self.register_buffer("_fixed_random_topk_indices", torch.empty(0, dtype=torch.int64), persistent=True)

    @staticmethod
    def _validate_inputs(q: torch.Tensor, k: torch.Tensor) -> None:
        if q.ndim < 2 or k.ndim < 2:
            raise ValueError("q 和 k 都必须至少是二维张量")
        if q.shape != k.shape:
            raise ValueError("q 和 k 必须具有相同形状")
        if q.shape[-1] == 0:
            raise ValueError("bit_width 不能为空")

    @staticmethod
    def _binary_float_to_bool(x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.bool:
            return x
        return x.to(torch.bool)

    @staticmethod
    def _xnor_similarity_proxy(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        q_float = q.to(torch.float32)
        k_float = k.to(torch.float32)
        bit_width = q_float.shape[-1]
        pairwise_dot = torch.matmul(q_float, k_float.transpose(-1, -2))
        q_ones = q_float.sum(dim=-1, keepdim=True)
        k_ones = k_float.sum(dim=-1).unsqueeze(-2)
        return float(bit_width) - q_ones - k_ones + 2.0 * pairwise_dot

    @staticmethod
    def _validate_score_bias(score_bias: torch.Tensor, scores: torch.Tensor) -> None:
        """Validate an integer-valued bias that broadcasts over batch only."""
        if score_bias.dtype == torch.bool or score_bias.is_complex():
            raise ValueError("score_bias must have a real integer or floating-point dtype")
        if score_bias.device != scores.device:
            raise ValueError("score_bias and XNOR/popcount scores must be on the same device")
        if score_bias.ndim < 2 or score_bias.shape[-2:] != scores.shape[-2:]:
            raise ValueError("score_bias must preserve the complete query/key score matrix")
        if score_bias.ndim not in (scores.ndim, scores.ndim - 1):
            raise ValueError(
                "score_bias must match the score shape or omit only its leading batch dimension"
            )
        if score_bias.ndim == scores.ndim - 1:
            if score_bias.shape != scores.shape[1:]:
                raise ValueError(
                    "batch-shared score_bias must equal score.shape[1:] exactly"
                )
        else:
            if score_bias.shape[1:] != scores.shape[1:] or score_bias.shape[0] not in (
                1,
                scores.shape[0],
            ):
                raise ValueError(
                    "score_bias may broadcast only a singleton leading batch dimension"
                )
        if score_bias.is_floating_point():
            detached_bias = score_bias.detach()
            values_are_valid = torch.logical_and(
                torch.isfinite(detached_bias),
                detached_bias == torch.round(detached_bias),
            ).all()
            message = "score_bias must contain only finite integer-valued values"
            if detached_bias.device.type == "cuda" and hasattr(torch, "_assert_async"):
                # Preserve strict checking without a host synchronization in
                # every attention block of every training step.
                torch._assert_async(values_are_valid, message)
            elif not bool(values_are_valid.item()):
                raise ValueError(message)

    @staticmethod
    def _add_score_bias(scores: torch.Tensor, score_bias: torch.Tensor) -> torch.Tensor:
        if scores.is_floating_point() or score_bias.is_floating_point():
            dtype = torch.promote_types(scores.dtype, score_bias.dtype)
            return scores.to(dtype) + score_bias.to(dtype)
        dtype = torch.int64 if scores.dtype == torch.int64 or score_bias.dtype == torch.int64 else torch.int32
        return scores.to(dtype) + score_bias.to(dtype)

    @staticmethod
    def hard_topk_mask_kth_boundary_surrogate(
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        temperature: float = 1.0,
        eps: float = 1e-6,
        detach_kth_value: bool = True,
        use_midpoint_theta: bool = False,
        normalize_soft_mask: bool = True,
    ):
        """
        按照第K大边界公式构造 hard-topk 的 surrogate。

        Args:
            hard_mask: [..., N]，bool 或 float
            proxy_scores: [..., N]
            limit: K
            temperature: T_topk
            eps: 稳定项
            detach_kth_value: 是否 StopGrad(theta_k)
            use_midpoint_theta: 是否使用第k大与第k+1大分数的算术平均作为阈值
            normalize_soft_mask: 是否对 soft_mask 做 sum≈k 的归一化
        Returns:
            out_mask: 前向 hard，反向 soft
            soft_mask: 软代理
            theta_k: 第K大边界
        """
        dtype = proxy_scores.dtype
        width = proxy_scores.shape[-1]

        if hard_mask.dtype == torch.bool:
            hard_float = hard_mask.to(dtype)
        else:
            hard_float = hard_mask.to(dtype)

        if limit <= 0:
            zero = torch.zeros_like(proxy_scores)
            theta_k = torch.full_like(proxy_scores[..., :1], float("-inf"))
            return zero, zero, theta_k

        if limit >= width:
            one = torch.ones_like(proxy_scores)
            theta_k = torch.full_like(proxy_scores[..., :1], float("inf"))
            return one, one, theta_k

        safe_temp = max(float(temperature), eps)

        if use_midpoint_theta:
            topk2_vals = torch.topk(proxy_scores, k=limit + 1, dim=-1).values
            kth_val = topk2_vals[..., limit - 1:limit]
            k1th_val = topk2_vals[..., limit:limit + 1]
            theta_k = 0.5 * (kth_val + k1th_val)
        else:
            topk_vals = torch.topk(proxy_scores, k=limit, dim=-1).values
            theta_k = topk_vals[..., -1:]  # [..., 1]

        if detach_kth_value:
            theta_k = theta_k.detach()

        soft_mask = torch.sigmoid((proxy_scores - theta_k) / safe_temp)
        if normalize_soft_mask:
            soft_mask = limit * soft_mask / (soft_mask.sum(dim=-1, keepdim=True) + eps)

        out_mask = hard_float + (soft_mask - soft_mask.detach())
        return out_mask, soft_mask, theta_k

    @staticmethod
    def hard_topk_mask_random_kth_boundary_surrogate(
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        temperature: float = 1.0,
        eps: float = 1e-6,
        detach_kth_value: bool = True,
        candidate_indices: torch.Tensor | None = None,
        use_midpoint_theta: bool = False,
        normalize_soft_mask: bool = True,
    ):
        """
        与 kth surrogate 相同，但 theta_k 不再来自真实 top-k 边界，
        而是先从全部位置里均匀随机采样 k 个，再从这 k 个里随机挑 1 个作为阈值。
        如果给定 candidate_indices，则直接复用这组 k 个候选（例如随机前向实际选中的那组 index），
        避免前向随机一套、反向再随机另一套。

        当 use_midpoint_theta=True 时：
            - 若没有 candidate_indices，则改为随机采样 k+1 个候选，
              并使用这组随机候选里第 k 与第 k+1 个分数的平均值作为阈值；
            - 若给定 candidate_indices（大小为 k），则在其外再均匀随机补 1 个候选，
              再使用这组 k+1 个随机候选里第 k 与第 k+1 个分数的平均值作为阈值。
        """
        dtype = proxy_scores.dtype
        width = proxy_scores.shape[-1]

        if hard_mask.dtype == torch.bool:
            hard_float = hard_mask.to(dtype)
        else:
            hard_float = hard_mask.to(dtype)

        if limit <= 0:
            zero = torch.zeros_like(proxy_scores)
            theta_k = torch.full_like(proxy_scores[..., :1], float("-inf"))
            return zero, zero, theta_k

        if limit >= width:
            one = torch.ones_like(proxy_scores)
            theta_k = torch.full_like(proxy_scores[..., :1], float("inf"))
            return one, one, theta_k

        safe_temp = max(float(temperature), eps)

        if candidate_indices is None:
            random_scores = torch.rand(proxy_scores.shape, device=proxy_scores.device, dtype=torch.float32)
            sample_k = limit + 1 if use_midpoint_theta else limit
            sampled_idx = torch.topk(random_scores, k=sample_k, dim=-1, largest=True, sorted=False).indices
        else:
            sampled_idx = candidate_indices.to(torch.long)
            if sampled_idx.shape != proxy_scores.shape[:-1] + (limit,):
                raise ValueError("candidate_indices must have shape [..., k] matching proxy_scores")

            if use_midpoint_theta:
                random_scores = torch.rand(proxy_scores.shape, device=proxy_scores.device, dtype=torch.float32)
                random_scores = random_scores.scatter(
                    -1,
                    sampled_idx,
                    torch.full_like(sampled_idx, -1, dtype=random_scores.dtype),
                )
                extra_idx = torch.topk(random_scores, k=1, dim=-1, largest=True, sorted=False).indices
                sampled_idx = torch.cat([sampled_idx, extra_idx], dim=-1)
        sampled_scores = torch.gather(proxy_scores, -1, sampled_idx)

        if use_midpoint_theta:
            sampled_top_vals = torch.topk(sampled_scores, k=limit + 1, dim=-1).values
            kth_val = sampled_top_vals[..., limit - 1:limit]
            k1th_val = sampled_top_vals[..., limit:limit + 1]
            theta_k = 0.5 * (kth_val + k1th_val)
        else:
            random_pick_scores = torch.rand(sampled_scores.shape, device=proxy_scores.device, dtype=torch.float32)
            kth_pick = torch.topk(random_pick_scores, k=1, dim=-1, largest=True, sorted=False).indices
            theta_k = torch.gather(sampled_scores, -1, kth_pick)

        if detach_kth_value:
            theta_k = theta_k.detach()

        soft_mask = torch.sigmoid((proxy_scores - theta_k) / safe_temp)
        if normalize_soft_mask:
            soft_mask = limit * soft_mask / (soft_mask.sum(dim=-1, keepdim=True) + eps)

        out_mask = hard_float + (soft_mask - soft_mask.detach())
        return out_mask, soft_mask, theta_k

    @staticmethod
    def hard_topk_mask_soft_rank_surrogate(
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        temperature: float = 1.0,
        eps: float = 1e-6,
    ):
        """
        使用 soft-rank 近似 hard top-k：
        1) 用两两比较的 sigmoid 构造每个位置的 soft rank；
        2) 再把“rank <= k”写成一个平滑门；
        3) 最后归一化到总和约等于 k。

        Args:
            hard_mask: [..., N]，bool 或 float
            proxy_scores: [..., N]
            limit: K
            temperature: 复用 boundary surrogate temperature
            eps: 稳定项
        Returns:
            out_mask: 前向 hard，反向 soft
            soft_mask: 软代理
            soft_rank: 每个位置的软排序（最大值约为 1）
        """
        dtype = proxy_scores.dtype
        width = proxy_scores.shape[-1]

        if hard_mask.dtype == torch.bool:
            hard_float = hard_mask.to(dtype)
        else:
            hard_float = hard_mask.to(dtype)

        if limit <= 0:
            zero = torch.zeros_like(proxy_scores)
            soft_rank = torch.full_like(proxy_scores, float(width))
            return zero, zero, soft_rank

        if limit >= width:
            one = torch.ones_like(proxy_scores)
            soft_rank = torch.ones_like(proxy_scores)
            return one, one, soft_rank

        safe_temp = max(float(temperature), eps)

        score_i = proxy_scores.unsqueeze(-1)  # [..., N, 1]
        score_j = proxy_scores.unsqueeze(-2)  # [..., 1, N]
        gt_prob = torch.sigmoid((score_j - score_i) / safe_temp)  # [..., N, N]
        soft_rank = 0.5 + gt_prob.sum(dim=-1)  # [..., N]

        soft_mask = torch.sigmoid((float(limit) + 0.5 - soft_rank) / safe_temp)
        soft_mask = limit * soft_mask / (soft_mask.sum(dim=-1, keepdim=True) + eps)

        out_mask = hard_float + (soft_mask - soft_mask.detach())
        return out_mask, soft_mask, soft_rank

    @staticmethod
    def hard_topk_mask_sigmoid_topk_surrogate(
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        temperature: float = 1.0,
        eps: float = 1e-6,
        newton_steps: int = 24,
    ):
        """
        用全局平滑阈值构造 hard-topk 的 surrogate。

        令 soft_mask_i = sigmoid((s_i - theta) / tau)，其中 theta 不是第 k 大分数，
        而是通过约束 sum_i soft_mask_i = k 的一维方程求得。这样仍保持“边界附近梯度大”，
        但 theta 会受到全部输入分数影响，而不是只依赖单个 kth 边界值。

        Args:
            hard_mask: [..., N]，bool 或 float
            proxy_scores: [..., N]
            limit: K
            temperature: T_topk
            eps: 稳定项
            newton_steps: 固定 Newton 迭代次数（内部数值求解，不暴露为训练超参数）
        Returns:
            out_mask: 前向 hard，反向 soft
            soft_mask: 软代理
            theta: 由全局约束 sum soft_mask = k 得到的平滑阈值
        """
        dtype = proxy_scores.dtype
        width = proxy_scores.shape[-1]

        if hard_mask.dtype == torch.bool:
            hard_float = hard_mask.to(dtype)
        else:
            hard_float = hard_mask.to(dtype)

        if limit <= 0:
            zero = torch.zeros_like(proxy_scores)
            theta = torch.full_like(proxy_scores[..., :1], float("inf"))
            return zero, zero, theta

        if limit >= width:
            one = torch.ones_like(proxy_scores)
            theta = torch.full_like(proxy_scores[..., :1], float("-inf"))
            return one, one, theta

        safe_temp = max(float(temperature), eps)

        # 用真实 kth 边界作 Newton 初值，之后阈值会被所有输入共同修正。
        theta = torch.topk(proxy_scores, k=limit, dim=-1).values[..., -1:]

        for _ in range(newton_steps):
            soft_mask = torch.sigmoid((proxy_scores - theta) / safe_temp)
            residual = soft_mask.sum(dim=-1, keepdim=True) - float(limit)
            slope = (soft_mask * (1.0 - soft_mask)).sum(dim=-1, keepdim=True)
            theta = theta + safe_temp * residual / (slope + eps)

        soft_mask = torch.sigmoid((proxy_scores - theta) / safe_temp)
        out_mask = hard_float + (soft_mask - soft_mask.detach())
        return out_mask, soft_mask, theta
    def hard_topk_mask_subset_gibbs_surrogate(
        self,
        hard_mask: torch.Tensor,
        proxy_scores: torch.Tensor,
        limit: int,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        """
        Subset-Gibbs TopK surrogate.

        数学定义：
            对所有大小为 k 的子集 A 建立 Gibbs 分布：

                P(A | s) ∝ exp( sum_{i in A} s_i / tau ),  |A| = k

            soft_mask_i = P(i in A)

        等价于：
            F_tau(s) = tau * log sum_{|A|=k} exp( sum_{i in A} s_i / tau )
            soft_mask = grad_s F_tau(s)

        前向：
            使用 hard_mask，保持离散 TopK。

        反向：
            使用 subset-level Gibbs inclusion marginal。

        Args:
            hard_mask: [..., N]，bool 或 float，真实 hard TopK mask
            proxy_scores: [..., N]，TopK 的代理分数
            limit: K
            temperature: tau
            eps: 数值稳定项

        Returns:
            out_mask: 前向 hard，反向 soft
            soft_mask: subset Gibbs inclusion marginal
            smooth_value: 平滑 TopK value，形状 [..., 1]
        """

        dtype = proxy_scores.dtype
        device = proxy_scores.device
        width = proxy_scores.shape[-1]

        hard_float = hard_mask.to(dtype)

        if limit <= 0:
            zero = torch.zeros_like(proxy_scores)
            smooth_value = torch.zeros_like(proxy_scores[..., :1])
            return zero, zero, smooth_value

        if limit >= width:
            one = torch.ones_like(proxy_scores)
            smooth_value = proxy_scores.sum(dim=-1, keepdim=True)
            return one, one, smooth_value

        tau = max(float(temperature), eps)

        # ------------------------------------------------------------
        # 1. 构造 log w_i = s_i / tau
        # ------------------------------------------------------------
        # 减去每一行最大值做数值稳定。
        # 因为每个子集都包含 k 个元素，所以所有子集权重都会共同乘上 exp(-k * shift)，
        # 不会改变 soft_mask。
        logw_raw = proxy_scores / tau
        shift = logw_raw.max(dim=-1, keepdim=True).values
        logw = logw_raw - shift

        # 训练时需要对 log-domain DP 反向传播。直接用 -inf 表示“不可能状态”
        # 在某些 autograd 路径（尤其是 logaddexp / logsumexp 组合）里容易产生 NaN 梯度。
        # 这里改用足够小的有限负数，数学上仍然等价于零权重，只提升数值稳定性。
        neg_large = float(torch.finfo(dtype).min / 2.0)

        # ------------------------------------------------------------
        # 2. prefix DP
        # ------------------------------------------------------------
        # prefix[i][r] = log e_r(w_0, ..., w_{i-1})
        # 其中 e_r 是 r 阶基本对称多项式。
        # shape: [..., limit + 1]
        prefix = []

        dp = torch.full(
            (*logw.shape[:-1], limit + 1),
            neg_large,
            dtype=dtype,
            device=device,
        )
        dp[..., 0] = 0.0
        prefix.append(dp)

        for i in range(width):
            wi = logw[..., i:i + 1]

            shifted = torch.cat(
                [
                    torch.full_like(dp[..., :1], neg_large),
                    dp[..., :-1] + wi,
                ],
                dim=-1,
            )
            dp = torch.logaddexp(dp, shifted)
            prefix.append(dp)

        logZ_shifted = prefix[-1][..., limit:limit + 1]

        # 原始的 logZ：
        # logZ = log sum_A exp(sum_i s_i / tau)
        # 因为前面做了 logw - shift，所以这里要补回 k * shift。
        logZ = logZ_shifted + float(limit) * shift

        smooth_value = tau * logZ

        # ------------------------------------------------------------
        # 3. suffix DP
        # ------------------------------------------------------------
        # suffix[i][r] = log e_r(w_i, ..., w_{N-1})
        suffix = [None] * (width + 1)

        dp = torch.full(
            (*logw.shape[:-1], limit + 1),
            neg_large,
            dtype=dtype,
            device=device,
        )
        dp[..., 0] = 0.0
        suffix[width] = dp

        for i in reversed(range(width)):
            wi = logw[..., i:i + 1]

            shifted = torch.cat(
                [
                    torch.full_like(dp[..., :1], neg_large),
                    dp[..., :-1] + wi,
                ],
                dim=-1,
            )
            dp = torch.logaddexp(dp, shifted)
            suffix[i] = dp

        # ------------------------------------------------------------
        # 4. 计算 inclusion marginal
        # ------------------------------------------------------------
        # soft_mask_i = P(i in A)
        #
        # m_i = w_i * e_{k-1}(w_{-i}) / e_k(w)
        #
        # 其中：
        # e_{k-1}(w_{-i})
        # = sum_{a=0}^{k-1}
        #       e_a(w_0, ..., w_{i-1})
        #       e_{k-1-a}(w_{i+1}, ..., w_{N-1})
        # ------------------------------------------------------------
        prefix_stack = torch.stack(prefix[:-1], dim=-2)   # [..., N, limit + 1]
        suffix_stack = torch.stack(suffix[1:], dim=-2)    # [..., N, limit + 1]

        subset_terms = prefix_stack[..., :limit] + torch.flip(
            suffix_stack[..., :limit],
            dims=[-1],
        )
        log_e_without = torch.logsumexp(subset_terms, dim=-1)  # [..., N]

        # 注意这里用的是 shifted 后的 logw 和 shifted 后的 logZ。
        # shift 会在分子分母中抵消，所以直接用 logZ_shifted 即可。
        log_m = logw + log_e_without - logZ_shifted
        soft_mask = torch.exp(log_m)

        # 理论上 sum soft_mask = k。
        # 这里做一次极轻量修正，避免 log-domain DP 在低精度下产生微小误差。
        soft_mask = soft_mask * (
            float(limit) / (soft_mask.sum(dim=-1, keepdim=True) + eps)
        )

        # ------------------------------------------------------------
        # 5. Straight-through estimator
        # ------------------------------------------------------------
        # 前向：
        #   out_mask = hard_float
        #
        # 反向：
        #   d out_mask / d score = d soft_mask / d score
        # ------------------------------------------------------------
        out_mask = hard_float + (soft_mask - soft_mask.detach())

        return out_mask, soft_mask, smooth_value
    @staticmethod
    def _build_selector_mask(indices: torch.Tensor, row_count: int) -> torch.Tensor:
        row_ids = torch.arange(row_count, device=indices.device, dtype=indices.dtype)
        view_shape = [1] * indices.ndim # 创建一个长度为 indices.ndim 的列表，列表中的每个元素都为 1 indices.ndim是什么？是 indices 的维度数量，例如如果 indices 是一个形状为 (2, 3) 的张量，那么 indices.ndim 就是 2。这个 view_shape 列表的作用是为了后续将 row_ids 视图化成与 indices 兼容的形状，以便进行比较操作。通过将 view_shape 的最后一个元素设置为 row_count，可以确保 row_ids 的形状与 indices 的前几个维度兼容，而最后一个维度则对应于 row_count。
        view_shape[-1] = row_count # 将 view_shape 的最后一个元素设置为 row_count，这样就可以将 row_ids 视图化成与 indices 兼容的形状，便于后续的比较操作。最终 row_ids 的形状将是 [1, 1, ..., row_count]，其中有 indices.ndim - 1 个 1 和一个 row_count。
        row_ids = row_ids.view(*view_shape)
        return torch.any(indices.unsqueeze(-1) == row_ids, dim=-2)

    @staticmethod
    def _select_left(score_a: torch.Tensor, score_b: torch.Tensor, index_a: torch.Tensor, index_b: torch.Tensor) -> torch.Tensor:
        """
        输入是两组候选的分数和索引
        先比 score，分数大的赢。
        如果分数相等，再比 index，索引大的赢（这是 tie-break 规则）。
        输出 True 表示选左边 A，False 表示选右边 B。
        """
        a_gt_b = score_a > score_b
        #逐元素比较A的分数是否大于B的分数，如果大于的话，那么a_gt_b对应位置就是True，否则就是False
        a_eq_b = score_a == score_b
        #逐元素比较A的分数是否等于B的分数，如果相等的话，那么a_eq_b对应位置就是True，否则就是False
        idx_a_gt_b = index_a > index_b
        #逐元素比较A的索引是否大于B的索引，如果大于的话，那么idx_a_gt_b对应位置就是True，否则就是False
        return torch.logical_or(a_gt_b, torch.logical_and(a_eq_b, idx_a_gt_b))
        #select_left = a_gt_b | (a_eq_b & idx_a_gt_b)

    # def _select_pair(
    #     self,
    #     score_a: torch.Tensor,
    #     score_b: torch.Tensor,
    #     index_a: torch.Tensor,
    #     index_b: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """
    #     给定两组候选 (score_a, index_a) 和 (score_b, index_b)，
    #     按照 _select_left 的规则，分出“高的一组”和“低的一组”，并把分数和索引都一起返回
    #     """
    #     select_left = self._select_left(score_a, score_b, index_a, index_b)#先按照上面的规则得到掩码
    #     high_scores = torch.where(select_left, score_a, score_b)#如果这个位置select_left为True，那么就返回score_a，否则返回score_b
    #     low_scores = torch.where(select_left, score_b, score_a)#如果这个位置select_left为True，那么就返回score_b，否则返回score_a
    #     high_indices = torch.where(select_left, index_a, index_b)#如果这个位置select_left为True，那么就返回index_a，否则返回index_b
    #     low_indices = torch.where(select_left, index_b, index_a)#如果这个位置select_left为True，那么就返回index_b，否则返回index_a
    #     return high_scores, low_scores, high_indices, low_indices

    @staticmethod
    def _pad_last_dim(x: torch.Tensor, target_len: int, pad_value: int | float) -> torch.Tensor:
        current_len = x.shape[-1]
        if current_len >= target_len:
            return x

        pad_shape = list(x.shape)
        pad_shape[-1] = target_len - current_len #x需要补到的长度
        pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
        #用pad填充到pad_shape指定的形状，pad_shape的最后一个维度是target_len - current_len，也就是需要补的长度，其他维度和x一样。pad_value是用来填充的值，dtype和device与x相同。
        return torch.cat([x, pad], dim=-1)

    def _winner_tree_topk(self, scores: torch.Tensor, indices: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
        item_count = scores.shape[-1]
        if limit <= 0:
            return scores[..., :0], indices[..., :0]
        if item_count <= 1:
            return scores[..., :limit], indices[..., :limit]

        n_logical = 1 << (item_count - 1).bit_length() #n_logical = 补到不小于 item_count 的最小 2 的幂
        min_score: int | float
        if scores.is_floating_point():
            min_score = float(torch.finfo(scores.dtype).min)
        else:
            min_score = int(torch.iinfo(scores.dtype).min) #获取这个数据类型的最小值，用于填充永远不会用到的位置
        invalid_index = -1 #永远用不到的位置的id是-1

        work_scores = self._pad_last_dim(scores, n_logical, min_score).clone()
        work_indices = self._pad_last_dim(indices, n_logical, invalid_index).clone()
        #score和index都pad到指定的长度
        prefix_shape = scores.shape[:-1] #并行处理所有batch维度
        base_positions = torch.arange(n_logical, device=scores.device, dtype=torch.long) #这代表每个叶子在最后一维里的位置编号
        if prefix_shape:
            view_shape = [1] * len(prefix_shape) + [n_logical]#创建一个形状为 [1, 1, ..., n_logical] 的视图形状，其中有 len(prefix_shape) 个 1 和一个 n_logical。这个 view_shape 的作用是为了将 base_positions 视图化成与 scores 的前几个维度兼容的形状，以便进行广播操作。通过将 view_shape 的前面部分设置为 1，可以确保 base_positions 在与 scores 进行操作时会自动广播到正确的形状。
            level_positions = base_positions.view(*view_shape).expand(*prefix_shape, n_logical)
            #扩展维度到 [batch_dim1, batch_dim2, ..., n_logical]
            #具体讲讲每一个函数的作用：首先，base_positions 是一个形状为 [n_logical] 的张量，包含了从 0 到 n_logical-1 的整数。通过调用 view(*view_shape)，我们将 base_positions 的形状变为 [1, 1, ..., n_logical]，其中有 len(prefix_shape) 个 1 和一个 n_logical。接下来，通过调用 expand(*prefix_shape, n_logical)，我们将这个视图扩展到与 scores 的前几个维度兼容的形状，即 [batch_dim1, batch_dim2, ..., n_logical]，其中 batch_dim1、batch_dim2 等是 scores 的前几个维度的大小。这样，level_positions 就是一个与 scores 兼容的张量，其中最后一维包含了每个叶子在最后一维里的位置编号，而前面的维度则与 scores 的批次维度相匹配。
        else:
            level_positions = base_positions

        tree_levels = [level_positions] #第 0 层是叶子层，存的是位置编号
        current = level_positions
        current_width = n_logical
        #current 指向当前层位置，current_width 是当前层宽度
        while current_width > 1: #向上归并直到树顶
            '''
            按奇偶把当前层位置分成左右两组，用这些位置去取分数和索引取出左右两边真实值
            用 _select_left 判断每对谁赢，生成“上一层”的赢家位置，把这一层保存到 tree_levels，记录宽度减半
            '''
            left_pos = current[..., 0::2]
            right_pos = current[..., 1::2]
            #奇偶切分当前层的位置，得到左右子树的位置
            left_scores = torch.gather(work_scores, -1, left_pos)
            right_scores = torch.gather(work_scores, -1, right_pos)
            #根据位置从 work_scores 中 gather 出左右子树的分数
            left_indices = torch.gather(work_indices, -1, left_pos)
            right_indices = torch.gather(work_indices, -1, right_pos)
            #根据位置从 work_indices 中 gather 出左右子树的索引

            select_left = self._select_left(left_scores, right_scores, left_indices, right_indices)
            #左树和右树进行比较，得到一个布尔掩码，表示对于每一对左右子树，是否选择左子树
            current = torch.where(select_left, left_pos, right_pos)
            #根据 select_left 的结果，选择左子树的位置或右子树的位置，得到当前层的胜者位置
            tree_levels.append(current)#这个列表每一个元素是tensor，存储了每一层的胜者位置，从叶子层到树顶层。通过调用 tree_levels.append(current)，我们将当前层的胜者位置添加到 tree_levels 列表中，以便后续在 top-k 选择过程中使用。最终，tree_levels 列表中的最后一个元素将包含树顶层的胜者位置，即整个输入中的最大值的位置。
            #append是啥？append是列表的方法，用于在列表末尾添加一个元素。这里的 tree_levels 是一个列表，存储了每一层的胜者位置。通过调用 tree_levels.append(current)，我们将当前层的胜者位置添加到 tree_levels 列表中，以便后续在 top-k 选择过程中使用。
            #那这样操作结束后的tree_level形状是什么？每一层的形状都是 [batch_dim1, batch_dim2, ..., current_width]，其中 current_width 是当前层的宽度，初始为 n_logical，每向上归并一次就会减半，直到最后一层的宽度为 1。每个元素表示该位置的胜者在原始输入中的位置编号。
            current_width //= 2

        selected_scores = []
        selected_indices = []
        level_count = len(tree_levels)
        replay_parent_slots_lut = _get_winner_tree_replay_parent_slots(n_logical, scores.device)
        #告诉你这一层要把新赢家写回到父层的哪个槽位
        replay_left_child_slots_lut = _get_winner_tree_replay_left_child_slots(n_logical, scores.device)
        #告诉你这一层要去子层的哪个左孩子槽位取候选
        replay_right_child_slots_lut = _get_winner_tree_replay_right_child_slots(n_logical, scores.device)
        #告诉你这一层要去子层的哪个右孩子槽位取候选
        for _ in range(limit):
            winner_pos = tree_levels[-1]# 取当前冠军在叶子层的原始位置编号
            winner_score = torch.gather(work_scores, -1, winner_pos)
            winner_index = torch.gather(work_indices, -1, winner_pos)
            #取原始的冠军分数和索引值
            selected_scores.append(winner_score)
            selected_indices.append(winner_index)
            ##把这一轮冠军的分数和索引存起来，循环结束后会拼成 top-k 输出。
            work_scores = work_scores.scatter(-1, winner_pos, torch.full_like(winner_score, min_score))
            work_indices = work_indices.scatter(-1, winner_pos, torch.full_like(winner_index, invalid_index))
            #把这次冠军在 work_scores 里的位置改成极小值 min_score，同时把对应 index 改成无效值 invalid_index
            replay_parent_slots = replay_parent_slots_lut[winner_pos]
            replay_left_child_slots = replay_left_child_slots_lut[winner_pos]
            replay_right_child_slots = replay_right_child_slots_lut[winner_pos]
            #得到该冠军从叶子到根每一层：要更新哪个父槽位 parent_slot，该父节点对应哪两个孩子槽位 left_child_slot、right_child_slot
            for level_idx in range(1, level_count):
                #内层循环从第 1 层到根层，逐层回放更新
                child_level = tree_levels[level_idx - 1]
                parent_slot = replay_parent_slots[..., level_idx - 1]
                left_child_slot = replay_left_child_slots[..., level_idx - 1]
                right_child_slot = replay_right_child_slots[..., level_idx - 1]

                left_pos = torch.gather(child_level, -1, left_child_slot)
                right_pos = torch.gather(child_level, -1, right_child_slot)
                left_scores = torch.gather(work_scores, -1, left_pos)
                right_scores = torch.gather(work_scores, -1, right_pos)
                left_indices = torch.gather(work_indices, -1, left_pos)
                right_indices = torch.gather(work_indices, -1, right_pos)

                select_left = self._select_left(left_scores, right_scores, left_indices, right_indices)
                new_winner = torch.where(select_left, left_pos, right_pos)
                tree_levels[level_idx] = tree_levels[level_idx].scatter(-1, parent_slot, new_winner)

        return torch.cat(selected_scores, dim=-1), torch.cat(selected_indices, dim=-1)

    @staticmethod
    def _torch_topk_indices(scores: torch.Tensor, limit: int) -> torch.Tensor:
        if limit <= 0:
            return torch.empty(scores.shape[:-1] + (0,), dtype=torch.int64, device=scores.device)
        return torch.topk(scores, k=limit, dim=-1, largest=True, sorted=True).indices.to(torch.int64)

    @staticmethod
    def _random_k_indices(scores: torch.Tensor, limit: int) -> torch.Tensor:
        if limit <= 0:
            return torch.empty(scores.shape[:-1] + (0,), dtype=torch.int64, device=scores.device)
        random_scores = torch.rand(scores.shape, device=scores.device, dtype=torch.float32)
        return torch.topk(random_scores, k=limit, dim=-1, largest=True, sorted=False).indices.to(torch.int64)

    def _fixed_random_k_indices(self, scores: torch.Tensor, limit: int) -> torch.Tensor:
        row_count = scores.shape[-1]
        if limit <= 0:
            return torch.empty(scores.shape[:-1] + (0,), dtype=torch.int64, device=scores.device)

        cached = self._fixed_random_topk_indices
        if (
            cached.numel() == 0
            or cached.device != scores.device
            or cached.shape != (row_count, limit)
        ):
            random_scores = torch.rand((row_count, row_count), device=scores.device, dtype=torch.float32)
            cached = torch.topk(random_scores, k=limit, dim=-1, largest=True, sorted=False).indices.to(torch.int64)
            self._fixed_random_topk_indices = cached

        view_shape = [1] * (scores.ndim - 2) + [row_count, limit]
        return cached.view(*view_shape).expand(*scores.shape[:-1], limit)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        topk: int,
        return_selector_mask: bool = True,
        return_topk_packed_scores: bool = True,
        proxy_scores: torch.Tensor | None = None,
        score_bias: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(q, k)

        bit_width = q.shape[-1]
        q_bits = self._binary_float_to_bool(q)
        k_bits = self._binary_float_to_bool(k)

        packed_q = self.pack_bits(q_bits)
        packed_k = self.pack_bits(k_bits)
        packed_xnor_scores = self.packed_xnor(packed_q, packed_k, bit_width)

        byte_popcount = self.byte_popcount(packed_xnor_scores)
        xnor_popcount = self.adder_tree(byte_popcount)
        if score_bias is not None:
            self._validate_score_bias(score_bias, xnor_popcount)

        # Random/static-k variants are causal controls for score-based routing
        # and deliberately ignore positional bias in both hard and surrogate
        # paths.  Dense has no selector, but keeps the complete (optionally
        # biased) score matrix so a weighted-V readout can use every key.
        applied_score_bias = (
            score_bias if self.topk_forward_mode in ("topk", "dense") else None
        )
        routing_scores = (
            xnor_popcount
            if applied_score_bias is None
            else self._add_score_bias(xnor_popcount, applied_score_bias)
        )
        row_count = q.shape[-2]
        limit = row_count if self.topk_forward_mode == "dense" else min(topk, row_count)
        topk_indices: torch.Tensor
        selector_mask = None

        if self.topk_forward_mode == "dense":
            base_indices = torch.arange(row_count, device=q.device, dtype=torch.int64)
            view_shape = [1] * routing_scores.ndim
            view_shape[-1] = row_count
            topk_indices = base_indices.view(*view_shape).expand_as(routing_scores)
        elif self.topk_forward_mode == "random-k":
            topk_indices = self._random_k_indices(routing_scores, limit)
        elif self.topk_forward_mode == "fixed-random-k":
            topk_indices = self._fixed_random_k_indices(routing_scores, limit)
        elif self.topk_impl == "torch-topk":
            topk_indices = self._torch_topk_indices(routing_scores, limit)
        else:
            base_indices = torch.arange(row_count, device=q.device, dtype=torch.int64)
            view_shape = [1] * routing_scores.ndim
            view_shape[-1] = row_count
            base_indices = base_indices.view(*view_shape).expand_as(routing_scores)

            _, topk_indices = self._winner_tree_topk(routing_scores, base_indices, limit)

        if return_selector_mask:
            selector_mask_hard = self._build_selector_mask(topk_indices, q.shape[-2])

            score_bias_needs_grad = bool(
                applied_score_bias is not None
                and applied_score_bias.is_floating_point()
                and applied_score_bias.requires_grad
            )
            use_surrogate = self.training and (
                ((q.requires_grad or k.requires_grad) and (q.is_floating_point() or k.is_floating_point()))
                or score_bias_needs_grad
            )
            if use_surrogate:
                if proxy_scores is None:
                    proxy_scores = self._xnor_similarity_proxy(q, k)
                elif proxy_scores.shape != xnor_popcount.shape:
                    raise ValueError("proxy_scores must have the same shape as xnor_popcount")
                if applied_score_bias is not None:
                    proxy_scores = self._add_score_bias(proxy_scores, applied_score_bias)
                if self.topk_surrogate_mode == "random-kth":
                    selector_mask, _, _ = self.hard_topk_mask_random_kth_boundary_surrogate(
                        selector_mask_hard,
                        proxy_scores,
                        limit,
                        temperature=self.boundary_surrogate_temperature,
                        detach_kth_value=self.topk_kth_detach_value,
                        candidate_indices=topk_indices if self.topk_forward_mode != "topk" else None,
                        use_midpoint_theta=self.topk_kth_use_midpoint_theta,
                        normalize_soft_mask=self.topk_kth_normalize_soft_mask,
                    )
                elif self.topk_surrogate_mode == "soft-rank":
                    selector_mask, _, _ = self.hard_topk_mask_soft_rank_surrogate(
                        selector_mask_hard,
                        proxy_scores,
                        limit,
                        temperature=self.boundary_surrogate_temperature,
                    )
                elif self.topk_surrogate_mode == "sigmoid-topk":
                    selector_mask, _, _ = self.hard_topk_mask_sigmoid_topk_surrogate(
                        selector_mask_hard,
                        proxy_scores,
                        limit,
                        temperature=self.boundary_surrogate_temperature,
                    )
                elif self.topk_surrogate_mode == "subset-gibbs":
                    selector_mask, _, _ = self.hard_topk_mask_subset_gibbs_surrogate(
                        selector_mask_hard,
                        proxy_scores,
                        limit,
                        temperature=self.boundary_surrogate_temperature,
                    )
                else:
                    selector_mask, _, _ = self.hard_topk_mask_kth_boundary_surrogate(
                        selector_mask_hard,
                        proxy_scores,
                        limit,
                        temperature=self.boundary_surrogate_temperature,
                        detach_kth_value=self.topk_kth_detach_value,
                        use_midpoint_theta=self.topk_kth_use_midpoint_theta,
                        normalize_soft_mask=self.topk_kth_normalize_soft_mask,
                    )
            else:
                selector_mask = selector_mask_hard

        gathered_packed_scores = None
        if return_topk_packed_scores:
            gathered_packed_scores = torch.gather(
                packed_xnor_scores,
                -2,
                topk_indices.unsqueeze(-1).expand(*topk_indices.shape, packed_xnor_scores.shape[-1]),
            )

        return {
            "packed_q": packed_q,
            "packed_k": packed_k,
            "packed_xnor_scores": packed_xnor_scores,
            "byte_popcount": byte_popcount,
            "xnor_popcount": xnor_popcount,
            "routing_scores": routing_scores,
            "topk_indices": topk_indices,
            "selector_mask": selector_mask,
            "topk_packed_xnor_scores": gathered_packed_scores,
        }


def estimate_tensor_bytes(tensors: Dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values() if isinstance(tensor, torch.Tensor))


# def test_packed_xnor_topk_5x6_top3() -> Dict[str, torch.Tensor]:
#     """
#     固定两个 5x6 的 0/1 输入，topk=3，打印前向输出与反向梯度，便于快速手工核对。
#     """
#     module = PackedXnorTopK()
#     module.train()

#     q = torch.tensor(
#         [
#             [1, 0, 1, 0, 1, 0],
#             [0, 1, 1, 0, 0, 1],
#             [1, 1, 0, 0, 1, 0],
#             [0, 0, 1, 1, 0, 1],
#             [1, 0, 0, 1, 1, 0],
#         ],
#         dtype=torch.float32,
#         requires_grad=True,
#     )
#     k = torch.tensor(
#         [
#             [1, 0, 1, 1, 0, 0],
#             [0, 1, 0, 1, 1, 0],
#             [1, 1, 1, 0, 0, 1],
#             [0, 0, 1, 0, 1, 1],
#             [1, 0, 0, 1, 0, 1],
#         ],
#         dtype=torch.float32,
#         requires_grad=True,
#     )

#     outputs = module(q, k, topk=3, return_selector_mask=True, return_topk_packed_scores=True)
#     selector_mask = outputs["selector_mask"]

#     weight = torch.tensor(
#         [
#             [0.10, 0.20, 0.30, 0.40, 0.50],
#             [0.60, 0.70, 0.80, 0.90, 1.00],
#             [1.10, 1.20, 1.30, 1.40, 1.50],
#             [1.60, 1.70, 1.80, 1.90, 2.00],
#             [2.10, 2.20, 2.30, 2.40, 2.50],
#         ],
#         dtype=selector_mask.dtype,
#         device=selector_mask.device,
#     )

#     loss = (selector_mask * weight).sum()

#     backward_ok = True
#     backward_error = ""
#     try:
#         loss.backward()
#     except RuntimeError as exc:
#         backward_ok = False
#         backward_error = str(exc)

#     q_grad_ok = q.grad is not None and torch.isfinite(q.grad).all() and q.grad.abs().sum() > 0
#     k_grad_ok = k.grad is not None and torch.isfinite(k.grad).all() and k.grad.abs().sum() > 0
#     grad_broken = (not backward_ok) or (q.grad is None) or (k.grad is None)

#     print("=== Forward ===")
#     print("q:")
#     print(q.detach())
#     print("k:")
#     print(k.detach())
#     print("xnor_popcount:")
#     print(outputs["xnor_popcount"].detach())
#     print("topk_indices:")
#     print(outputs["topk_indices"].detach())
#     print("selector_mask:")
#     print(selector_mask.detach())
#     print("topk_packed_xnor_scores:")
#     print(outputs["topk_packed_xnor_scores"].detach())
#     print("loss:", float(loss.detach().item()))

#     print("=== Backward ===")
#     print("surrogate_mode=boundary-surrogate")
#     print(f"selector_mask.requires_grad={selector_mask.requires_grad}")
#     print(f"loss.requires_grad={loss.requires_grad}")
#     print(f"backward_ok={backward_ok}")
#     if not backward_ok:
#         print(f"backward_error={backward_error}")
#     print(f"q_grad_ok={bool(q_grad_ok)}")
#     print(f"k_grad_ok={bool(k_grad_ok)}")
#     print(f"grad_broken={bool(grad_broken)}")
#     print("q.grad:")
#     print(q.grad)
#     print("k.grad:")
#     print(k.grad)

#     return {
#         "q": q.detach(),
#         "k": k.detach(),
#         "xnor_popcount": outputs["xnor_popcount"].detach(),
#         "topk_indices": outputs["topk_indices"].detach(),
#         "selector_mask": selector_mask.detach(),
#         "topk_packed_xnor_scores": outputs["topk_packed_xnor_scores"].detach(),
#         "loss": loss.detach(),
#         "q_grad": q.grad.detach() if q.grad is not None else None,
#         "k_grad": k.grad.detach() if k.grad is not None else None,
#     }



# if __name__ == "__main__":
#     test_outputs = test_packed_xnor_topk_5x6_top3()
#     print("=== Estimated output bytes ===")
#     print(estimate_tensor_bytes(test_outputs))
