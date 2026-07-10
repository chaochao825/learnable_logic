"""
输入级参数化（IWP）逻辑层和前馈网络实现。

本模块实现了：
- LogicLayerIWP: 带温度控制和软评估的逻辑层包装器
- LogicFFN_IWP: 多层逻辑层堆叠的前馈网络，支持温度计编码
"""

import math

import torch
import torch.nn as nn

from local_difflogic.difflogic import IWPLogicLayerCudaFunction, LogicLayerIWP as BaseLogicLayerIWP  # type: ignore
from local_difflogic.functional import GradFactor, bin_gate  # type: ignore


class _BinarizeSTE(torch.autograd.Function):
    """
    硬门控的直通估计器（Straight-through Estimator）。

    前向传播：在 0.5 处阈值化（二值化）。
    反向传播：梯度不变地传递。
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播：将输入二值化（>= 0.5 为 1，否则为 0）。

        Args:
            ctx: 上下文对象
            input: 输入张量

        Returns:
            torch.Tensor: 二值化后的张量
        """
        return (input >= 0.5).to(input.dtype) #input >= 0.5为真返回1，<0.5为假返回0。

    @staticmethod # 静态方法，不需要实例化即可调用。
    def backward(ctx, grad_output: torch.Tensor):
        """
        反向传播：直接传递梯度。

        Args:
            ctx: 上下文对象
            grad_output: 输出梯度

        Returns:
            torch.Tensor: 输入梯度（与输出梯度相同）
        """
        return grad_output


class LogicLayerIWP(BaseLogicLayerIWP):
    """
    输入级参数化（IWP, Input-wise Parametrization）逻辑层。

    这是对官方 LogicLayerIWP 的包装器，在基础逻辑层的基础上增加了温度控制和软硬评估切换功能。

    **核心特性**：
    1. **IWP 参数化**：每个逻辑门只需要 4 个参数（而不是传统的 16 个），大大减少了参数量
    2. **温度控制**：通过温度参数控制逻辑门的软硬程度，支持温度退火训练策略
    3. **软硬评估切换**：
       - 训练时：使用软评估（连续概率值），保证梯度可以反向传播
       - 推理时：使用硬评估（二值化），实现真正的二进制逻辑门网络

    **工作原理**：
    - 输入：二值化的张量（通过温度计编码或其他方式得到）
    - 权重处理：`weights / temperature` → `act_fn` → 得到概率权重 `w`
    - 逻辑门运算：使用 CUDA 加速的逻辑门计算（IWP 模式）
    - 输出：逻辑门的输出结果

    **与父类的区别**：
    - 父类 `BaseLogicLayerIWP`：基础的 IWP 逻辑层实现
    - 本类：添加了温度控制和软硬评估模式切换，更适合训练和推理的灵活切换
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        grad_factor: float = 1.0,
        residual_fraction: float = 0.0,
        residual_freeze: bool = True,
        weight_init_choice: str = "ri",
        weight_init_sigma: float = 0.5,
        soft_eval: bool = True,
        connections: str = "unique",
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        device: torch.device | None = None,
    ):
        """
        初始化 IWP 逻辑层。

        Args:
            in_dim: 输入维度，即输入张量的最后一个维度大小
            out_dim: 输出维度，即输出神经元的数量（每个神经元对应一个逻辑门）
            grad_factor: 梯度因子，用于深度模型的梯度缩放（默认 1.0，不缩放）
            residual_fraction: 残差连接比例，范围 [0, 1]（0 表示无残差连接）
                - 残差连接会固定部分权重为恒等映射，有助于梯度流动
            residual_freeze: 残差门是否固定为直通 A 并关闭梯度；False 时仅改连接，权重可训练
            weight_init_choice: 权重初始化方式（"ri" 软直通 A / "gauss" 高斯随机）
            weight_init_sigma: weight_init 尺度（ri 时 ±sigma 送入激活；gauss 时标准差；SIN01 下 0.5 更可塑）
            soft_eval: 是否使用软评估模式
                - True: 使用连续概率值（可微，适合训练）
                - False: 使用二值化权重（硬评估，适合推理）
            connections: 连接模式，决定输入如何连接到逻辑门
                - "unique": 唯一连接模式，确保每个输入都被使用
                - "random": 随机连接模式
            act_fn: 激活函数类型，用于将权重 logits 转换为概率
                - "SIN01": 正弦激活函数（默认）
                - "sigmoid": Sigmoid 激活函数
                - "sigmoid-st": Sigmoid 直通估计器
                - "sin-st": 正弦直通估计器
                - "linear": 线性激活
            shift_init_enable: 是否执行重尾初始化（shift_init）
            shift_init_type: 重尾类型（ri / and-or / and-or-ri / uniform）
            init_shift: 初始化偏移量，用于权重初始化（默认 1.2）
            init_shift_direction: 初始化偏移方向，4位二进制字符串（如 "0101"）
                用于控制每个逻辑门参数的初始化方向
            device: 计算设备（CUDA 或 CPU），如果为 None 则自动选择（优先 CUDA）

        Note:
            - IWP 模式下，每个逻辑门只需要 4 个参数（而不是传统的 16 个）
            - 权重形状为 `[out_dim, 4]`，其中 4 对应 IWP 的 4 个参数
            - 如果使用 CUDA，会自动使用 CUDA 加速的逻辑门计算
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_str = "cuda" if device.type == "cuda" else "cpu"
        res_fraction = max(float(residual_fraction), 0.0)
        implementation = "cuda" if device.type == "cuda" else "python"
        super().__init__(
            in_dim=in_dim,
            out_dim=out_dim,
            device=device_str,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
            res_connect_fraction=res_fraction,
            residual_freeze=residual_freeze,
            act_fn=act_fn,
            shift_init_enable=shift_init_enable,
            shift_init_type=shift_init_type,
            init_shift=init_shift if init_shift > 0 else 0.0,
            init_shift_direction=init_shift_direction,
            weight_init_choice=weight_init_choice,
            sigma=weight_init_sigma,
            run_op_on_iwp=False,
        )
        self.runtime_device = device
        self.temperature = 1.0
        self._soft_eval = soft_eval

    def set_temperature(self, temperature: float):
        """
        设置温度参数（用于控制逻辑门的软硬程度）。

        温度参数用于控制权重从软（连续概率）到硬（二值化）的过渡：
        - 温度高（如 1.0）：权重更软，输出更平滑，适合训练初期
        - 温度低（如 0.1）：权重更硬，输出更接近二值化，适合训练后期和推理

        温度退火策略：训练过程中逐渐降低温度，从软到硬平滑过渡。

        Args:
            temperature: 温度值，必须 > 0
                - 温度越高：`weights / temperature` 越小，sigmoid 输出更平滑（接近 0.5）
                - 温度越低：`weights / temperature` 越大，sigmoid 输出更极端（接近 0 或 1）

        Note:
            - 温度值会被限制在 [1e-4, +∞) 范围内，避免除零错误
            - 在训练循环中，通常通过 `get_temperature()` 函数计算当前步的温度值
        """
        self.temperature = max(float(temperature), 1e-4)

    def harden(self):
        """
        硬化逻辑层（将权重完全二值化，用于推理）。

        将软权重（连续值）转换为硬权重（二值化），使模型在推理时完全使用二进制逻辑门。
        这是一个不可逆的操作，会永久修改权重值。

        **硬化过程**：
        1. 计算当前权重的概率值：`probs = act_fn(weights / temperature)`
        2. 二值化：`binary = (probs >= 0.5)` → 0 或 1
        3. 将权重设置为极值：`weights = binary * 20.0 - 10.0`
           - 如果 binary=1，权重设为 10.0（经过 sigmoid 后接近 1）
           - 如果 binary=0，权重设为 -10.0（经过 sigmoid 后接近 0）

        **使用场景**：
        - 推理前：将训练好的软权重转换为硬权重
        - Hard finetune 阶段：在训练后期进行硬权重微调

        Note:
            - 此操作在 `torch.no_grad()` 上下文中执行，不会影响梯度计算
            - 硬化后的权重几乎不可再训练（因为 sigmoid(±10) ≈ 0 或 1，梯度接近 0）
            - 通常在整个模型的所有逻辑层上统一调用 `harden_model()`
        不同 act_fn 下的硬化值选择：
        - sigmoid / sigmoid-st:
            - 使用对称的 ±C（这里保持 ±10），再经过 sigmoid(±10/T) 后仍然远离 0.5，阈值稳定。
        - SIN01 / sin-st:
            - 为避免在前向中再次经过 SIN01(weights / temperature) 时翻转，
              将 binary=1 / 0 映射为 ±(π/2)*T，使得 weights / T = ±π/2，
              从而 SIN01(±π/2) ∈ {1,0}，再阈值不会改变门类型。
        - linear:
            - 线性激活前向为 clamp(x, 0, 1)。这里直接使用 0/1 作为硬化值，
              在当前温度范围内（T≈[0.6, 1.0]）满足阈值稳定。

        其他未知 act_fn 保持原先的 ±10 行为以兼容旧用法。
        """
        self._soft_eval = False
        with torch.no_grad():
            # 1) 先根据当前 act_fn 与 temperature 计算 binary（软权重 → 0/1）
            temp = max(float(self.temperature), 1e-4)
            logits = self.weights / temp
            probs = self.act_fn(logits)
            binary = (probs >= 0.5).to(self.weights.dtype)

            # 2) 根据 act_fn 选择自洽的硬化值，使得前向再次经过
            #    act_fn(weights / temperature) + 阈值时不会翻转门类型。
            act_fn_str = getattr(self, "act_fn_str", None)

            # 默认回退常数幅度（与历史实现一致）
            pos_val = 10.0
            neg_val = -10.0

            if act_fn_str in ("sigmoid", "sigmoid-st", None):
                # 传统 sigmoid 系列：±C 在 sigmoid(·/T) 下不会穿越 0.5 阈值。
                w_hard = binary * pos_val + (1.0 - binary) * neg_val
            elif act_fn_str in ("SIN01", "sin-st"):
                # SIN01 / sin-st: 选择 ±(π/2)*T，使 weights / T = ±π/2，
                # SIN01(±π/2) ∈ {1,0}，阈值稳定。
                scale = (math.pi / 2.0) * temp
                # binary∈{0,1} → (2*binary-1)∈{-1,+1}
                w_hard = (2.0 * binary - 1.0) * scale
            elif act_fn_str == "linear":
                # 线性直通：直接使用 0/1，前向 clamp(weights/T,0,1) 再阈值即可。
                w_hard = binary
            else:
                # 未知激活：保持旧行为 ±10 以兼容。
                w_hard = binary * pos_val + (1.0 - binary) * neg_val

            self.weights.data.copy_(w_hard.to(self.weights.dtype))

    def set_soft_eval(self, soft: bool):
        """
        设置软评估模式标志。

        控制前向传播时是否使用软评估（连续概率值）还是硬评估（二值化权重）。
        注意：这只是一个标志，不会改变权重值本身。要真正硬化权重，需要调用 `harden()`。

        Args:
            soft: 软评估模式标志
                - True: 使用软评估，权重通过 `act_fn(weights / temperature)` 得到连续概率值
                  适合训练，梯度可以正常反向传播
                - False: 使用硬评估，权重通过 `_BinarizeSTE(probs)` 二值化为 0/1
                  适合推理，但训练时如果 `self.training=True`，仍会使用软评估

        Note:
            - 在 `_forward_cuda_with_temperature()` 中，实际判断条件是：
              `if self._soft_eval or self.training:` → 使用软评估
            - 这意味着训练时（`self.training=True`）总是使用软评估，无论 `_soft_eval` 的值
            - 只有在推理时（`self.training=False`）且 `_soft_eval=False` 时，才会使用硬评估
        """
        self._soft_eval = bool(soft)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行逻辑门网络的计算。

        根据实现方式（CUDA 或 Python）选择相应的前向传播方法。
        这是 PyTorch 模块的标准前向传播接口。

        Args:
            x: 输入张量，形状为 `[batch_size, in_dim]` 或 `[..., in_dim]`
               通常是经过温度计编码后的二值化张量，或直接归一化的连续值

        Returns:
            torch.Tensor: 输出张量，形状为 `[batch_size, out_dim]` 或 `[..., out_dim]`
                每个输出维度对应一个逻辑门的输出

        Note:
            - 如果 `grad_factor != 1.0`，会对输入应用梯度缩放（用于深度模型的梯度控制）
            - CUDA 实现：使用 `_forward_cuda_with_temperature()`，性能更好
            - Python 实现：使用 `forward_python()`，作为 CUDA 不可用时的备选
        """
        if self.grad_factor != 1.0:
            x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == "cuda":
            return self._forward_cuda_with_temperature(x)
        return self.forward_python(x)

    def _forward_cuda_with_temperature(self, x: torch.Tensor) -> torch.Tensor:
        """
        CUDA 加速的前向传播实现（带温度控制）。

        这是性能优化的 CUDA 实现，使用编译后的 CUDA 内核进行逻辑门计算。

        **计算流程**：
        1. 输入转置：`x` [B, in_dim] → `x_t` [in_dim, B]（CUDA 内核要求的格式）
        2. 权重处理：
           - `logits = weights / temperature`（温度缩放）
           - `probs = act_fn(logits)`（激活函数，得到概率值）
           - 根据 `_soft_eval` 和 `training` 决定使用软评估还是硬评估
        3. CUDA 内核计算：调用 `IWPLogicLayerCudaFunction` 执行逻辑门运算
        4. 输出转置：`y` [out_dim, B] → `y` [B, out_dim]

        Args:
            x: 输入张量，形状为 `[batch_size, in_dim]`
               训练时必须位于 CUDA 设备上

        Returns:
            torch.Tensor: 输出张量，形状为 `[batch_size, out_dim]`
                每个输出维度对应一个逻辑门的计算结果

        Note:
            - 训练时：使用 `IWPLogicLayerCudaFunction.apply()`，支持梯度反向传播
            - 推理时：在 `torch.no_grad()` 上下文中执行，不计算梯度
            - 如果 `_soft_eval=True` 或 `training=True`，使用连续概率权重（软评估）
            - 如果 `_soft_eval=False` 且 `training=False`，使用二值化权重（硬评估）
        """
        if self.training:
            assert x.device.type == "cuda", x.device
        assert x.ndim == 2, x.ndim

        x_t = x.transpose(0, 1).contiguous()
        assert x_t.shape[0] == self.in_dim, (x_t.shape, self.in_dim)

        logits = self.weights / self.temperature
        probs = self.act_fn(logits)
        if self._soft_eval or self.training:
            w = probs.to(x.dtype)
        else:
            w = _BinarizeSTE.apply(probs).to(x.dtype)

        a, b = self.indices
        args = (x_t, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y)

        if self.training:
            y = IWPLogicLayerCudaFunction.apply(*args).transpose(0, 1)
        else:
            with torch.no_grad():
                y = IWPLogicLayerCudaFunction.apply(*args).transpose(0, 1)
        return y

    def forward_python(self, x: torch.Tensor) -> torch.Tensor:
        """
        Python 实现的前向传播（带温度控制）。

        这是纯 Python 实现，作为 CUDA 不可用时的备选方案。
        使用 Python 实现的逻辑门运算，性能较 CUDA 版本慢，但可以在 CPU 上运行。

        **计算流程**：
        1. 提取输入对：根据 `indices` 获取每对输入 `(a, b)`
        2. 权重处理：与 CUDA 版本相同，通过温度缩放和激活函数得到概率权重
        3. Python 逻辑门计算：使用 `bin_gate()` 函数执行逻辑门运算
        4. 返回输出

        Args:
            x: 输入张量，形状为 `[..., in_dim]`（支持任意前导维度）

        Returns:
            torch.Tensor: 输出张量，形状为 `[..., out_dim]`
                保持输入的前导维度不变，只改变最后一个维度

        Note:
            - 此实现使用 `bin_gate()` 函数进行逻辑门计算（在 `functional.py` 中定义）
            - 性能较 CUDA 版本慢，主要用于调试或 CPU 环境
            - 支持任意维度的输入（不仅限于 2D），更灵活但更慢
        """
        assert x.shape[-1] == self.in_dim, (x.shape[-1], self.in_dim)
        a = x[..., self.indices[0]]
        b = x[..., self.indices[1]]
        logits = self.weights / self.temperature
        probs = self.act_fn(logits)
        if self._soft_eval or self.training:
            weights = probs
        else:
            weights = _BinarizeSTE.apply(probs)
        return bin_gate(a, b, weights)


class LearnableConnLightLogicLayer(nn.Module):
    """T-Net-style candidate-pool connectivity wrapped around the existing IWP gate."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layer_id: int,
        candidate_k: int = 64,
        grad_factor: float = 1.0,
        residual_fraction: float = 0.0,
        residual_freeze: bool = True,
        weight_init_choice: str = "ri",
        weight_init_sigma: float = 0.5,
        soft_eval: bool = True,
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        seed: int | None = None,
        cross_block_source_count: int = 0,
        cross_block_candidate_frac: float = 0.0,
    ):
        super().__init__()
        if layer_id < 1:
            raise ValueError("layer_id must be >= 1")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_id = int(layer_id)
        self.candidate_k = int(candidate_k)
        self.tau_conn = 2.0
        self.beta_skip = 0.0
        self.cross_block_source_count = max(int(cross_block_source_count), 0)
        self.cross_block_candidate_frac = max(0.0, min(1.0, float(cross_block_candidate_frac)))
        self.register_buffer("connections_fixed_flag", torch.zeros((), dtype=torch.bool))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gate = LogicLayerIWP(
            in_dim=2 * out_dim,
            out_dim=out_dim,
            grad_factor=grad_factor,
            residual_fraction=residual_fraction,
            residual_freeze=residual_freeze,
            weight_init_choice=weight_init_choice,
            weight_init_sigma=weight_init_sigma,
            soft_eval=soft_eval,
            connections="unique",
            act_fn=act_fn,
            shift_init_enable=shift_init_enable,
            shift_init_type=shift_init_type,
            init_shift=init_shift,
            init_shift_direction=init_shift_direction,
            device=device,
        )
        strict_a, strict_b = make_strict_indices(out_dim, self.gate.indices[0].device)
        _overwrite_indices(self.gate, strict_a, strict_b)

        g = None
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(seed)
        cand_l0, cand_n0 = self._build_candidates(g)
        cand_l1, cand_n1 = self._build_candidates(g)
        self.register_buffer("candidate_layers0", cand_l0)
        self.register_buffer("candidate_nodes0", cand_n0)
        self.register_buffer("candidate_layers1", cand_l1)
        self.register_buffer("candidate_nodes1", cand_n1)
        self.conn_logits0 = nn.Parameter(torch.zeros(out_dim, self.candidate_k))
        self.conn_logits1 = nn.Parameter(torch.zeros(out_dim, self.candidate_k))
        self.register_buffer("selected_layers0", torch.zeros(out_dim, dtype=torch.long))
        self.register_buffer("selected_nodes0", torch.zeros(out_dim, dtype=torch.long))
        self.register_buffer("selected_layers1", torch.zeros(out_dim, dtype=torch.long))
        self.register_buffer("selected_nodes1", torch.zeros(out_dim, dtype=torch.long))

    def _randint(self, high: int, size: tuple[int, ...], generator: torch.Generator | None) -> torch.Tensor:
        if generator is None:
            return torch.randint(high, size, dtype=torch.long)
        return torch.randint(high, size, generator=generator, dtype=torch.long)

    def _build_candidates(self, generator: torch.Generator | None) -> tuple[torch.Tensor, torch.Tensor]:
        cross_count = 0
        if self.cross_block_source_count > 0 and self.cross_block_candidate_frac > 0.0:
            cross_count = int(round(self.candidate_k * self.cross_block_candidate_frac))
            cross_count = max(0, min(self.candidate_k, cross_count))
        local_k = self.candidate_k - cross_count
        counts = {
            "prev": int(round(local_k * 0.60)),
            "prev2": int(round(local_k * 0.25)),
            "older": int(round(local_k * 0.10)),
        }
        counts["input"] = local_k - sum(counts.values())
        counts["cross"] = cross_count
        sources = {
            "prev": [self.layer_id - 1],
            "prev2": [self.layer_id - 2] if self.layer_id - 2 >= 0 else [],
            "older": list(range(1, max(self.layer_id - 2, 1))),
            "input": [0],
            # Negative ids index the external history appended after local history:
            # -1 = immediate previous Transformer block, -2 = two blocks back, etc.
            "cross": [-(i + 1) for i in range(self.cross_block_source_count)],
        }
        available_prev = sources["prev"] if sources["prev"] else [0]
        for key in list(counts):
            if not sources[key]:
                counts["prev"] += counts[key]
                counts[key] = 0
        layers = torch.empty(self.out_dim, self.candidate_k, dtype=torch.long)
        nodes = torch.empty(self.out_dim, self.candidate_k, dtype=torch.long)
        offset = 0
        for key in ("prev", "prev2", "older", "input", "cross"):
            count = counts[key]
            if count <= 0:
                continue
            src_layers = sources[key] or available_prev
            src_idx = self._randint(len(src_layers), (self.out_dim, count), generator)
            src_tensor = torch.tensor(src_layers, dtype=torch.long)
            layers[:, offset : offset + count] = src_tensor[src_idx]
            nodes[:, offset : offset + count] = self._randint(self.in_dim, (self.out_dim, count), generator)
            offset += count
        if offset != self.candidate_k:
            layers[:, offset:] = self.layer_id - 1
            nodes[:, offset:] = self._randint(self.in_dim, (self.out_dim, self.candidate_k - offset), generator)
        if generator is None:
            perm = torch.rand(self.out_dim, self.candidate_k).argsort(dim=1)
        else:
            perm = torch.rand(self.out_dim, self.candidate_k, generator=generator).argsort(dim=1)
        return torch.gather(layers, 1, perm), torch.gather(nodes, 1, perm)

    def set_connection_schedule(self, tau_conn: float, beta_skip: float, freeze: bool = False) -> None:
        self.tau_conn = max(float(tau_conn), 1e-4)
        self.beta_skip = float(beta_skip)
        self.conn_logits0.requires_grad_(not freeze)
        self.conn_logits1.requires_grad_(not freeze)

    def _is_connections_fixed(self) -> bool:
        return bool(self.connections_fixed_flag.item())

    def _scores(self, logits: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        penalty = self._distance_penalty(layers, dtype=logits.dtype, device=logits.device)
        return logits - float(self.beta_skip) * penalty

    def _distance_penalty(
        self,
        layers: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        layers = layers.to(device)
        penalty = self.layer_id - layers
        cross_mask = layers < 0
        if cross_mask.any():
            cross_order = (-layers).clamp_min(1)
            cross_penalty = self.layer_id + cross_order - 1
            penalty = torch.where(cross_mask, cross_penalty, penalty)
        return penalty.to(dtype=dtype, device=device)

    def _gather(self, history: list[torch.Tensor], layers: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        batch = history[0].shape[0]
        values = torch.empty(batch, self.out_dim, self.candidate_k, device=history[0].device, dtype=history[0].dtype)
        layers = layers.to(history[0].device)
        nodes = nodes.to(history[0].device)
        for source_layer in torch.unique(layers).tolist():
            source = history[int(source_layer)]
            mask = layers == int(source_layer)
            safe_nodes = nodes.masked_fill(~mask, 0).reshape(-1)
            gathered = source.index_select(1, safe_nodes).view(batch, self.out_dim, self.candidate_k)
            values = torch.where(mask.unsqueeze(0), gathered, values)
        return values

    def _soft_port(
        self,
        history: list[torch.Tensor],
        logits: torch.Tensor,
        layers: torch.Tensor,
        nodes: torch.Tensor,
    ) -> torch.Tensor:
        device = history[0].device
        dtype = history[0].dtype
        layers = layers.to(device)
        nodes = nodes.to(device)
        weights = torch.softmax(self._scores(logits, layers) / self.tau_conn, dim=-1).to(dtype)
        out = torch.zeros(history[0].shape[0], self.out_dim, device=device, dtype=dtype)
        for source_layer in torch.unique(layers).tolist():
            source = history[int(source_layer)]
            mask = layers == int(source_layer)
            routing = torch.zeros(self.out_dim, source.shape[1], device=device, dtype=dtype)
            routing.scatter_add_(1, nodes.masked_fill(~mask, 0), weights.masked_fill(~mask, 0.0))
            out = out + source.matmul(routing.t())
        return out

    def _fixed_port(self, history: list[torch.Tensor], layers: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        out = torch.empty(history[0].shape[0], self.out_dim, device=history[0].device, dtype=history[0].dtype)
        layers = layers.to(history[0].device)
        nodes = nodes.to(history[0].device)
        for source_layer in torch.unique(layers).tolist():
            mask = layers == int(source_layer)
            out[:, mask] = history[int(source_layer)].index_select(1, nodes[mask])
        return out

    def forward_from_history(self, history: list[torch.Tensor]) -> torch.Tensor:
        if self._is_connections_fixed():
            u0 = self._fixed_port(history, self.selected_layers0, self.selected_nodes0)
            u1 = self._fixed_port(history, self.selected_layers1, self.selected_nodes1)
        else:
            u0 = self._soft_port(history, self.conn_logits0, self.candidate_layers0, self.candidate_nodes0)
            u1 = self._soft_port(history, self.conn_logits1, self.candidate_layers1, self.candidate_nodes1)
        gate_input = torch.empty(
            history[0].shape[0],
            2 * self.out_dim,
            device=history[0].device,
            dtype=history[0].dtype,
        )
        gate_input[:, 0::2] = u0
        gate_input[:, 1::2] = u1
        return self.gate(gate_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from_history([x])

    def selected_sources(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._is_connections_fixed():
            return self.selected_layers0, self.selected_nodes0, self.selected_layers1, self.selected_nodes1
        with torch.no_grad():
            idx0 = self._scores(self.conn_logits0, self.candidate_layers0).argmax(dim=-1)
            idx1 = self._scores(self.conn_logits1, self.candidate_layers1).argmax(dim=-1)
            row = torch.arange(self.out_dim, device=idx0.device)
            return (
                self.candidate_layers0[row, idx0],
                self.candidate_nodes0[row, idx0],
                self.candidate_layers1[row, idx1],
                self.candidate_nodes1[row, idx1],
            )

    def discretize_connections(self, beta_skip: float | None = None) -> None:
        old_beta = self.beta_skip
        if beta_skip is not None:
            self.beta_skip = float(beta_skip)
        with torch.no_grad():
            l0, n0, l1, n1 = self.selected_sources()
            self.selected_layers0.copy_(l0.to(self.selected_layers0.device))
            self.selected_nodes0.copy_(n0.to(self.selected_nodes0.device))
            self.selected_layers1.copy_(l1.to(self.selected_layers1.device))
            self.selected_nodes1.copy_(n1.to(self.selected_nodes1.device))
        self.connections_fixed_flag.fill_(True)
        self.conn_logits0.requires_grad_(False)
        self.conn_logits1.requires_grad_(False)
        self.beta_skip = old_beta

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "connections_fixed_flag"
        if key not in state_dict:
            state_dict[key] = torch.zeros_like(self.connections_fixed_flag)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def connection_metrics(self) -> dict[str, float]:
        with torch.no_grad():
            l0, _n0, l1, _n1 = self.selected_sources()
            selected_layers = torch.cat([l0.detach().cpu(), l1.detach().cpu()])
            distances = self._distance_penalty(selected_layers, dtype=torch.float32, device=selected_layers.device).cpu()
            input_mask = selected_layers == 0
            cross_mask = selected_layers < 0
            hidden = ~(input_mask | cross_mask)
            total = max(int(selected_layers.numel()), 1)
            if self._is_connections_fixed():
                entropy_value = 0.0
            else:
                probs0 = torch.softmax(self._scores(self.conn_logits0, self.candidate_layers0) / self.tau_conn, dim=-1)
                probs1 = torch.softmax(self._scores(self.conn_logits1, self.candidate_layers1) / self.tau_conn, dim=-1)
                entropy = -0.5 * (
                    (probs0 * probs0.clamp_min(1e-12).log()).sum(dim=-1).mean()
                    + (probs1 * probs1.clamp_min(1e-12).log()).sum(dim=-1).mean()
                )
                entropy_value = float(entropy.item())
            return {
                "conn_avg_skip_distance": float(distances.mean().item()),
                "conn_pct_l1": float(((distances == 1) & hidden).sum().item() / total),
                "conn_pct_l2": float(((distances == 2) & hidden).sum().item() / total),
                "conn_pct_older": float(((distances >= 3) & hidden).sum().item() / total),
                "conn_pct_input": float(input_mask.sum().item() / total),
                "conn_pct_cross_block": float(cross_mask.sum().item() / total),
                "conn_entropy": entropy_value,
            }

    def gate_dead_ratio(self) -> float:
        weights = self.gate.weights
        temp = max(float(getattr(self.gate, "temperature", 1.0)), 1e-4)
        with torch.no_grad():
            bits = self.gate.act_fn(weights / temp) >= 0.5
            dead = (bits.amin(dim=1) == bits.amax(dim=1)).sum().item()
        return float(dead / max(int(bits.shape[0]), 1))

    def set_temperature(self, temperature: float) -> None:
        self.gate.set_temperature(temperature)

    def set_soft_eval(self, soft: bool) -> None:
        self.gate.set_soft_eval(soft)

    def harden(self) -> None:
        self.discretize_connections()
        self.gate.harden()


class LogicFFN_IWP(nn.Module):
    """堆叠多个 LogicLayerIWP 层以替代 ViT 块中的 FFN。

    支持温度计编码（Thermometer Encoding）：
    - 训练时：可微分编码，软权重
    - 推理时：硬编码，硬权重，完全二值化
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 3,
        hidden_multiplier: float = 1.0,
        n_thresholds: int = 31,  # 温度编码的阈值数量
        use_thermometer: bool = True,  # 是否使用温度编码
        encoding_temperature: float = 10.0,  # 编码时的温度参数（训练时可微，推理时硬）
        grad_factor: float = 1.0,
        residual_init: bool = True,
        weight_init_choice: str = "ri",
        weight_init_sigma: float = 0.5,
        residual_connect_fraction: float | None = None,
        soft_eval: bool = True,
        connections: str = "unique",
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        connectivity: str = "fixed",
        learnable_conn_k: int = 64,
        learnable_conn_use_skip_bias: bool = True,
        cross_block_source_count: int = 0,
        cross_block_candidate_frac: float = 0.0,
    ):
        super().__init__()
        assert num_layers >= 1
        if connectivity not in {"fixed", "learnable"}:
            raise ValueError(f"Unknown connectivity: {connectivity}")
        self.embed_dim = embed_dim
        self.n_thresholds = n_thresholds
        self.use_thermometer = use_thermometer
        self.encoding_temperature = encoding_temperature
        self._train_hard_thermometer = False
        self.connectivity = connectivity
        self.learnable_conn_use_skip_bias = bool(learnable_conn_use_skip_bias)
        self.cross_block_source_count = max(int(cross_block_source_count), 0)
        self.cross_block_candidate_frac = max(0.0, min(1.0, float(cross_block_candidate_frac)))
        if hidden_multiplier <= 0:
            raise ValueError("hidden_multiplier must be positive")
        rounded_hidden_multiplier = int(round(float(hidden_multiplier)))
        if abs(float(hidden_multiplier) - float(rounded_hidden_multiplier)) > 1e-6:
            raise ValueError("hidden_multiplier must be an integer value for grouped logic reduction")
        self.hidden_multiplier = max(1, rounded_hidden_multiplier)
        if connectivity == "learnable" and self.hidden_multiplier != 1:
            raise ValueError("hidden_multiplier > 1 is currently supported only with fixed connectivity")

        # 如果使用温度编码，逻辑层的输入输出维度需要扩展
        if use_thermometer:
            logic_dim = embed_dim * n_thresholds
        else:
            logic_dim = embed_dim
        self.logic_dim = logic_dim
        self.hidden_logic_dim = logic_dim * self.hidden_multiplier

        self.layers = nn.ModuleList()
        for idx in range(num_layers):
            in_dim = logic_dim if idx == 0 else self.hidden_logic_dim
            out_dim = self.hidden_logic_dim
            if residual_connect_fraction is not None:
                fraction = max(0.0, min(1.0, float(residual_connect_fraction)))
            elif residual_init and idx < num_layers - 1:
                fraction = (idx + 1) / max(1, num_layers) # 残差连接比例，从0到1，用于控制残差连接的强度。
            else:
                fraction = 0.0
            if in_dim != out_dim:
                fraction = 0.0
            if connectivity == "learnable":
                layer = LearnableConnLightLogicLayer(
                    in_dim,
                    out_dim,
                    layer_id=idx + 1,
                    candidate_k=learnable_conn_k,
                    grad_factor=grad_factor,
                    residual_fraction=fraction,
                    residual_freeze=residual_init,
                    weight_init_choice=weight_init_choice,
                    weight_init_sigma=weight_init_sigma,
                    soft_eval=soft_eval,
                    act_fn=act_fn,
                    shift_init_enable=shift_init_enable,
                    shift_init_type=shift_init_type,
                    init_shift=init_shift,
                    init_shift_direction=init_shift_direction,
                    cross_block_source_count=self.cross_block_source_count,
                    cross_block_candidate_frac=self.cross_block_candidate_frac,
                )
            else:
                layer = LogicLayerIWP(
                    in_dim,  # 使用扩展后的维度
                    out_dim,
                    grad_factor=grad_factor,
                    residual_fraction=fraction,
                    residual_freeze=residual_init,
                    weight_init_choice=weight_init_choice,
                    weight_init_sigma=weight_init_sigma,
                    soft_eval=soft_eval,
                    connections=connections,
                    act_fn=act_fn,
                    shift_init_enable=shift_init_enable,
                    shift_init_type=shift_init_type,
                    init_shift=init_shift,
                    init_shift_direction=init_shift_direction,
                )
            self.layers.append(layer)
        self.dropout = nn.Dropout(0.1)

        # 解码权重（可学习，用于将 n_thresholds 个二进制位解码回 FP32）
        if use_thermometer:
            # 初始化为均匀权重（可以学习）
            self.decode_weights = nn.Parameter(
                torch.ones(1, n_thresholds) / n_thresholds
            )

    def reduce_hidden_logic(self, h: torch.Tensor, hard: bool = False) -> torch.Tensor:
        if self.hidden_multiplier == 1:
            return h
        if self.use_thermometer:
            grouped = h.view(h.shape[0], self.embed_dim, self.n_thresholds, self.hidden_multiplier)
        else:
            grouped = h.view(h.shape[0], self.embed_dim, self.hidden_multiplier)
        h = grouped.mean(dim=-1)
        if hard:
            h = (h >= 0.5).float()
        return h.reshape(h.shape[0], -1)

    def thermometer_encode(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor: #false代表训练时使用软编码，true代表推理时使用硬编码。
        """
        温度计编码：将 FP32 值编码为多个二进制位

        Args:
            x: [B*N, embed_dim] FP32 值（应该在 [0,1] 范围内）
            hard: 是否使用硬编码（推理时）

        Returns:
            [B*N, embed_dim * n_thresholds] 二进制编码
        """
        # 确保输入在 [0,1] 范围内
        x_clamped = torch.clamp(x, 0.0, 1.0)

        # 计算阈值位置：1/(n+1), 2/(n+1), ..., n/(n+1)
        thresholds = torch.linspace(
            1.0 / (self.n_thresholds + 1),
            self.n_thresholds / (self.n_thresholds + 1),
            self.n_thresholds,
            device=x.device,
            dtype=x.dtype
        )  # [n_thresholds]

        # 扩展维度以便广播比较
        # x_clamped: [B*N, embed_dim, 1]
        # thresholds: [1, 1, n_thresholds]
        x_expanded = x_clamped.unsqueeze(-1)  # [B*N, embed_dim, 1]
        thresholds_expanded = thresholds.unsqueeze(0).unsqueeze(0)  # [1, 1, n_thresholds]

        if hard:
            # 推理时：硬编码（完全二值化，不可微）
            binary = (x_expanded > thresholds_expanded).float()  # [B*N, embed_dim, n_thresholds]
        else:
            # 训练时：可微分松弛（使用温度控制的 sigmoid）
            # 温度越高，越接近硬编码，但始终可微
            diff = (x_expanded - thresholds_expanded) * self.encoding_temperature
            binary = torch.sigmoid(diff)  # [B*N, embed_dim, n_thresholds]

        # 重塑为 [B*N, embed_dim * n_thresholds]
        binary_flat = binary.view(binary.shape[0], -1) #保持第一维不变，将后面维度展平。
        return binary_flat

    def thermometer_decode(self, binary: torch.Tensor) -> torch.Tensor:
        """
        温度计解码：将多个二进制位解码回 FP32 值

        Args:
            binary: [B*N, embed_dim * n_thresholds] 二进制编码

        Returns:
            [B*N, embed_dim] FP32 值
        """
        # 重塑为 [B*N, embed_dim, n_thresholds]
        binary_reshaped = binary.view(binary.shape[0], self.embed_dim, self.n_thresholds)

        # 使用可学习的权重进行加权求和
        # decode_weights: [1, n_thresholds]
        weights = torch.softmax(self.decode_weights, dim=-1)  # [1, n_thresholds]，可学好后推理存LUT5

        # 加权求和：sum(binary * weights) -> [B*N, embed_dim]
        x = (binary_reshaped * weights).sum(dim=-1)  # [B*N, embed_dim]

        return x

    def _forward_impl(
        self,
        x: torch.Tensor,
        external_history: list[torch.Tensor] | None = None,
        return_logic_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。

        Args:
            x: 输入张量 [batch_size, num_patches, embed_dim]

        Returns:
            torch.Tensor: 输出张量 [batch_size, num_patches, embed_dim]
        """
        external_history = list(external_history or [])
        b, n, d = x.shape
        h = x.view(b * n, d)

        if self.use_thermometer:
            # 1. 归一化到 [0, 1]（确保温度编码的语义正确）
            h = torch.sigmoid(h)

            # 2. 温度编码：FP32 -> 二进制
            # 推理时硬编码，训练时可微编码
            hard = (not self.training) or self._train_hard_thermometer
            h_binary = self.thermometer_encode(h, hard=hard)  # [B*N, embed_dim * n_thresholds]

            # 3. 逻辑层处理（在二值空间）
            history = [h_binary]
            for layer in self.layers:
                if hasattr(layer, "forward_from_history"):
                    h_binary = layer.forward_from_history(history + external_history)
                else:
                    h_binary = layer(h_binary)
                h_binary = self.dropout(h_binary)
                # 推理时保持硬二值化（确保输出是0或1）
                if hard:
                    h_binary = (h_binary >= 0.5).float()
                history.append(h_binary)
            h_binary = self.reduce_hidden_logic(h_binary, hard=hard)

            # 4. 解码：二进制 -> FP32
            logic_state = h_binary
            h = self.thermometer_decode(h_binary)  # [B*N, embed_dim]
        else:
            # 原始方式：直接归一化后送入逻辑层
            h = torch.sigmoid(h)
            history = [h]
            for layer in self.layers:
                if hasattr(layer, "forward_from_history"):
                    h = layer.forward_from_history(history + external_history)
                else:
                    h = layer(h)
                h = self.dropout(h)
                history.append(h)
            h = self.reduce_hidden_logic(h, hard=False)
            logic_state = h

        out = h.view(b, n, d)
        if return_logic_state:
            return out, logic_state
        return out

    def forward_with_logic_history(
        self,
        x: torch.Tensor,
        external_history: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, logic_state = self._forward_impl(
            x,
            external_history=external_history,
            return_logic_state=True,
        )
        return out, logic_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def set_temperature(self, temperature: float):
        """
        设置所有逻辑层的温度参数。

        Args:
            temperature: 温度值
        """
        for layer in self.layers:
            layer.set_temperature(temperature)

    def set_train_hard_thermometer(self, enabled: bool):
        self._train_hard_thermometer = bool(enabled)

    def set_connection_schedule(self, tau_conn: float, beta_skip: float, freeze: bool = False) -> None:
        beta = float(beta_skip) if self.learnable_conn_use_skip_bias else 0.0
        for layer in self.layers:
            if hasattr(layer, "set_connection_schedule"):
                layer.set_connection_schedule(tau_conn=tau_conn, beta_skip=beta, freeze=freeze)

    def discretize_connections(self, beta_skip: float | None = None) -> None:
        beta = beta_skip if self.learnable_conn_use_skip_bias else 0.0
        for layer in self.layers:
            if hasattr(layer, "discretize_connections"):
                layer.discretize_connections(beta_skip=beta)

    def connection_stats(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        total_ports = 0
        node_depths: list[torch.Tensor] = [torch.zeros(self.layers[0].in_dim, dtype=torch.float32)]
        fanout_keys: list[torch.Tensor] = []
        dead_sum = 0.0
        dead_count = 0
        base_dim = int(self.layers[0].in_dim)

        def layer_dead_ratio(layer: nn.Module) -> float:
            weights = getattr(layer, "weights", None)
            if weights is None:
                return 0.0
            temp = max(float(getattr(layer, "temperature", 1.0)), 1e-4)
            act_fn = getattr(layer, "act_fn", None)
            if act_fn is None:
                return 0.0
            with torch.no_grad():
                bits = act_fn(weights / temp) >= 0.5
                dead = (bits.amin(dim=1) == bits.amax(dim=1)).sum().item()
            return float(dead / max(int(bits.shape[0]), 1))

        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, "connection_metrics"):
                metrics = layer.connection_metrics()
                l0, n0, l1, n1 = layer.selected_sources()
                dead_ratio = layer.gate_dead_ratio()
            else:
                prev_layer = layer_idx
                indices = getattr(layer, "indices", None)
                if indices is None:
                    continue
                l0 = torch.full((layer.out_dim,), prev_layer, dtype=torch.long, device=indices[0].device)
                l1 = torch.full((layer.out_dim,), prev_layer, dtype=torch.long, device=indices[1].device)
                n0 = indices[0]
                n1 = indices[1]
                selected_layers = torch.cat([l0.detach().cpu(), l1.detach().cpu()])
                distances = (layer_idx + 1 - selected_layers).float()
                input_mask = selected_layers == 0
                cross_mask = selected_layers < 0
                hidden = ~(input_mask | cross_mask)
                total = max(int(selected_layers.numel()), 1)
                metrics = {
                    "conn_avg_skip_distance": float(distances.mean().item()),
                    "conn_pct_l1": float(((distances == 1) & hidden).sum().item() / total),
                    "conn_pct_l2": float(((distances == 2) & hidden).sum().item() / total),
                    "conn_pct_older": float(((distances >= 3) & hidden).sum().item() / total),
                    "conn_pct_input": float(input_mask.sum().item() / total),
                    "conn_pct_cross_block": float(cross_mask.sum().item() / total),
                    "conn_entropy": 0.0,
                }
                dead_ratio = layer_dead_ratio(layer)
            ports = int(layer.out_dim * 2)
            total_ports += ports
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * ports

            l0_cpu, n0_cpu = l0.detach().cpu(), n0.detach().cpu()
            l1_cpu, n1_cpu = l1.detach().cpu(), n1.detach().cpu()
            depth0 = torch.empty(layer.out_dim, dtype=torch.float32)
            depth1 = torch.empty(layer.out_dim, dtype=torch.float32)
            for source_layer in torch.unique(torch.cat([l0_cpu, l1_cpu])).tolist():
                mask0 = l0_cpu == int(source_layer)
                mask1 = l1_cpu == int(source_layer)
                if int(source_layer) < 0:
                    if mask0.any():
                        depth0[mask0] = 0.0
                    if mask1.any():
                        depth1[mask1] = 0.0
                elif mask0.any():
                    depth0[mask0] = node_depths[int(source_layer)].index_select(0, n0_cpu[mask0])
                if int(source_layer) >= 0 and mask1.any():
                    depth1[mask1] = node_depths[int(source_layer)].index_select(0, n1_cpu[mask1])
            node_depths.append(torch.maximum(depth0, depth1) + 1.0)
            fanout_l0 = torch.where(l0_cpu >= 0, l0_cpu, len(self.layers) + (-l0_cpu - 1))
            fanout_l1 = torch.where(l1_cpu >= 0, l1_cpu, len(self.layers) + (-l1_cpu - 1))
            fanout_keys.append(fanout_l0 * base_dim + n0_cpu)
            fanout_keys.append(fanout_l1 * base_dim + n1_cpu)
            dead_sum += dead_ratio * layer.out_dim
            dead_count += layer.out_dim

        stats = {key: value / max(total_ports, 1) for key, value in totals.items()}
        all_keys = torch.cat(fanout_keys) if fanout_keys else torch.empty(0, dtype=torch.long)
        if all_keys.numel() > 0:
            external_count = max(int(getattr(self, "cross_block_source_count", 0)), 0)
            minlength = max(len(node_depths) - 1 + external_count, 1) * base_dim
            fanout_all = torch.bincount(all_keys, minlength=minlength).float()
            fanout_used = fanout_all[fanout_all > 0]
            stats["conn_fanout_max"] = float(fanout_used.max().item()) if fanout_used.numel() else 0.0
            stats["conn_fanout_p95"] = float(torch.quantile(fanout_used, 0.95).item()) if fanout_used.numel() else 0.0
            stats["conn_fanout_p95_all"] = float(torch.quantile(fanout_all, 0.95).item())
            stats["conn_zero_fanout_ratio"] = float((fanout_all == 0).float().mean().item())
            stats["conn_source_used_ratio"] = float((fanout_all > 0).float().mean().item())
        stats["conn_dead_gate_ratio"] = float(dead_sum / max(dead_count, 1))
        stats["conn_effective_depth"] = float(max((d.max().item() for d in node_depths), default=0.0))
        if node_depths:
            out_depth = node_depths[-1]
            stats["conn_avg_output_depth"] = float(out_depth.mean().item())
            stats["conn_output_depth_p95"] = float(torch.quantile(out_depth, 0.95).item())
        return stats

    def harden(self):
        """
        硬化所有逻辑层（推理时使用）。

        将所有逻辑层从软评估模式切换到硬评估模式。
        """
        self.discretize_connections()
        for layer in self.layers:
            layer.harden()


def make_strict_indices(out_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成严格相邻索引 (0,1), (2,3), (4,5), ...，使 Gate[k] 严格连接 Input[2k] 与 Input[2k+1]。
    无随机打乱。
    """
    idx_a = torch.arange(0, out_dim * 2, 2, device=device, dtype=torch.int64)
    idx_b = torch.arange(1, out_dim * 2, 2, device=device, dtype=torch.int64)
    return idx_a, idx_b


def _recompute_given_x_indices_of_y(layer: nn.Module, in_dim: int, out_dim: int, device: torch.device) -> None:
    """用 layer.indices 重算 given_x_indices_of_y_start / given_x_indices_of_y，供 CUDA 反向使用。"""
    a = layer.indices[0].cpu()
    b = layer.indices[1].cpu()
    given_x_indices_of_y = [[] for _ in range(in_dim)]
    for y in range(out_dim):
        given_x_indices_of_y[int(a[y].item())].append(y)
        given_x_indices_of_y[int(b[y].item())].append(y)
    starts = [0]
    for g in given_x_indices_of_y:
        starts.append(starts[-1] + len(g))
    flat = [item for sublist in given_x_indices_of_y for item in sublist]
    layer.given_x_indices_of_y_start = torch.tensor(starts, device=device, dtype=torch.int64)
    layer.given_x_indices_of_y = torch.tensor(flat, dtype=torch.int64, device=device)


def _overwrite_indices(layer: nn.Module, new_a: torch.Tensor, new_b: torch.Tensor) -> None:
    """
    用 new_a/new_b 覆盖 LogicLayer 的 indices，并在需要时为 CUDA 反向重算 given_x_indices_of_y_*。
    """
    with torch.no_grad():
        a = new_a.long().to(layer.indices[0].device)
        b = new_b.long().to(layer.indices[1].device)
        layer.indices = (a, b)

    # 如果存在 CUDA 逆索引结构，则按当前 indices 重建
    if hasattr(layer, "given_x_indices_of_y_start") and hasattr(layer, "given_x_indices_of_y"):
        in_dim = layer.in_dim  # type: ignore[attr-defined]
        out_dim = layer.out_dim  # type: ignore[attr-defined]
        _recompute_given_x_indices_of_y(layer, in_dim, out_dim, layer.indices[0].device)

#现在训练实际的使用情况是没有使用这个层，而是直接使用FusedVectorizedForestLayer层。保留备用。
class VectorizedStrictForestLayer(nn.Module):
    """
    单层向量化、树内严格固定的森林。
    树内连接严格 (0,1),(2,3),...；唯一随机处为顶叶子采样的 leaf_indices。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tree_depth: int = 3,
        grad_factor: float = 1.0,
        residual_init: bool = True,
        weight_init_choice: str = "ri",
        soft_eval: bool = True,
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        seed: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.tree_depth = tree_depth
        leaves_per_tree = 2**tree_depth
        total_leaves = out_dim * leaves_per_tree

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        leaf_idx = torch.randint(0, in_dim, (total_leaves,), generator=g, dtype=torch.int64)
        self.register_buffer("leaf_indices", leaf_idx)

        self.layers = nn.ModuleList()
        current_dim = total_leaves
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for depth_step in range(tree_depth):
            out_d = current_dim // 2
            res_frac = (depth_step + 1) / tree_depth if residual_init else 0.0
            layer = LogicLayerIWP(
                in_dim=current_dim,
                out_dim=out_d,
                grad_factor=grad_factor,
                residual_fraction=res_frac,
                residual_freeze=residual_init,
                weight_init_choice=weight_init_choice,
                soft_eval=soft_eval,
                connections="unique",
                act_fn=act_fn,
                shift_init_enable=shift_init_enable,
                shift_init_type=shift_init_type,
                init_shift=init_shift,
                init_shift_direction=init_shift_direction,
                device=device,
            )
            with torch.no_grad():
                strict_a, strict_b = make_strict_indices(out_d, layer.indices[0].device)
                layer.indices[0].copy_(strict_a)
                layer.indices[1].copy_(strict_b)
                if hasattr(layer, "given_x_indices_of_y_start") and hasattr(layer, "given_x_indices_of_y"):
                    _recompute_given_x_indices_of_y(layer, current_dim, out_d, layer.indices[0].device)
            self.layers.append(layer)
            current_dim = out_d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x[:, self.leaf_indices]
        for layer in self.layers:
            h = layer(h)
        return h

    def set_temperature(self, temperature: float) -> None:
        for layer in self.layers:
            layer.set_temperature(temperature)

    def harden(self) -> None:
        for layer in self.layers:
            layer.harden()


class FusedVectorizedForestLayer(nn.Module):
    """
    融合索引版的向量化随机森林层。

    - 保持 leaf_indices 的随机采样不变；
    - 第一层直接从原始输入 x 中按 leaf_indices 取数（无 Python 展开）；
    - 后续层与严格版类似，执行 (0,1),(2,3)... 的二叉规约。
    """

    def __init__(
        self,
        in_dim: int,
        num_trees: int,
        tree_depth: int = 3,
        grad_factor: float = 1.0,
        residual_init: bool = True,
        weight_init_choice: str = "ri",
        weight_init_sigma: float = 0.5,
        residual_connect_fraction: float | None = None,
        soft_eval: bool = True,
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        seed: int | None = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_trees = num_trees
        self.tree_depth = tree_depth

        leaves_per_tree = 2**tree_depth
        total_leaves = num_trees * leaves_per_tree

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed)
        leaf_idx = torch.randint(0, in_dim, (total_leaves,), generator=g, dtype=torch.int64)
        self.register_buffer("leaf_indices", leaf_idx)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers: list[nn.Module] = []

        # 第一层：融合“采样 + 一级规约”
        out_dim_l1 = num_trees * (2 ** (tree_depth - 1))
        if residual_connect_fraction is not None:
            res_frac_l1 = max(0.0, min(1.0, float(residual_connect_fraction)))
        else:
            # 原始期望的残差比例（例如 1/3），用于对齐旧实现的行为
            desired_fraction = (1.0 / tree_depth) if residual_init else 0.0
            if residual_init and desired_fraction > 0.0:
                # 计算按期望比例需要的残差连接数量
                desired_count = int(torch.ceil(torch.tensor(float(out_dim_l1) * desired_fraction)).item())
                if desired_count > in_dim:
                    # 若超过输入维度上限，则将比例夹到安全范围内（略小于 in_dim / out_dim_l1）
                    safe_res_fraction = max(0.0, (float(in_dim) - 1e-5) / float(out_dim_l1))
                else:
                    safe_res_fraction = desired_fraction
            else:
                safe_res_fraction = 0.0
            res_frac_l1 = safe_res_fraction
        layer1 = LogicLayerIWP(
            in_dim=in_dim,
            out_dim=out_dim_l1,
            grad_factor=grad_factor,
            residual_fraction=res_frac_l1,
            residual_freeze=residual_init,
            weight_init_choice=weight_init_choice,
            weight_init_sigma=weight_init_sigma,
            soft_eval=soft_eval,
            connections="random",
            act_fn=act_fn,
            shift_init_enable=shift_init_enable,
            shift_init_type=shift_init_type,
            init_shift=init_shift,
            init_shift_direction=init_shift_direction,
            device=device,
        )

        # 将 leaf_indices 融合进第一层的 indices
        leaves = leaf_idx.view(num_trees, leaves_per_tree)  # [T, 2^depth]
        new_a = leaves[:, 0::2].reshape(-1)  # 每棵树 (0,2,4,6)
        new_b = leaves[:, 1::2].reshape(-1)  # 每棵树 (1,3,5,7)
        _overwrite_indices(layer1, new_a, new_b)
        layers.append(layer1)

        current_dim = out_dim_l1

        # 后续层：严格二叉规约 (0,1),(2,3),...
        for depth_step in range(1, tree_depth):
            out_d = current_dim // 2
            if residual_connect_fraction is not None:
                res_frac = max(0.0, min(1.0, float(residual_connect_fraction)))
            else:
                res_frac = ((depth_step + 1) / tree_depth) if residual_init else 0.0
            layer = LogicLayerIWP(
                in_dim=current_dim,
                out_dim=out_d,
                grad_factor=grad_factor,
                residual_fraction=res_frac,
                residual_freeze=residual_init,
                weight_init_choice=weight_init_choice,
                weight_init_sigma=weight_init_sigma,
                soft_eval=soft_eval,
                connections="unique",
                act_fn=act_fn,
                shift_init_enable=shift_init_enable,
                shift_init_type=shift_init_type,
                init_shift=init_shift,
                init_shift_direction=init_shift_direction,
                device=device,
            )
            strict_a, strict_b = make_strict_indices(out_d, layer.indices[0].device)
            _overwrite_indices(layer, strict_a, strict_b)

            layers.append(layer)
            current_dim = out_d

        assert current_dim == num_trees, (current_dim, num_trees)

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def set_temperature(self, temperature: float) -> None:
        for layer in self.layers:
            layer.set_temperature(temperature)

    def harden(self) -> None:
        for layer in self.layers:
            layer.harden()


class RandomForestLogicFFN_IWP(nn.Module):
    """
    向量化随机森林版逻辑 FFN（可配置层数、等宽）：
    - 输入/输出: [B, N, embed_dim]
    - 逻辑维度 logic_dim = embed_dim * n_thresholds（或 embed_dim），每层森林 in/out 均为 logic_dim
    - 数据流: Input -> [Thermometer] -> Bits -> [Forest 1] -> ... -> [Forest N] -> Bits -> [Decoder] -> Output
    """

    def __init__(
        self,
        embed_dim: int,
        n_thresholds: int = 31,
        use_thermometer: bool = True,
        encoding_temperature: float = 10.0,
        grad_factor: float = 1.0,
        residual_init: bool = True,
        weight_init_choice: str = "ri",
        weight_init_sigma: float = 0.5,
        residual_connect_fraction: float | None = None,
        soft_eval: bool = True,
        act_fn: str = "SIN01",
        shift_init_enable: bool = True,
        shift_init_type: str = "ri",
        init_shift: float = 1.2,
        init_shift_direction: str = "0101",
        num_forest_layers: int = 2,
        seed: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_thresholds = n_thresholds
        self.use_thermometer = use_thermometer
        self.encoding_temperature = encoding_temperature
        self._train_hard_thermometer = False

        logic_dim = embed_dim * n_thresholds if use_thermometer else embed_dim
        self.logic_dim = logic_dim
        self.num_forest_layers = num_forest_layers

        self.forest_layers = nn.ModuleList()
        for i in range(num_forest_layers):
            layer_seed = (seed + i) if seed is not None else None
            self.forest_layers.append(
                FusedVectorizedForestLayer(
                    in_dim=logic_dim,
                    num_trees=logic_dim,
                    tree_depth=3,
                    grad_factor=grad_factor,
                    residual_init=residual_init,
                    weight_init_choice=weight_init_choice,
                    weight_init_sigma=weight_init_sigma,
                    residual_connect_fraction=residual_connect_fraction,
                    soft_eval=soft_eval,
                    act_fn=act_fn,
                    shift_init_enable=shift_init_enable,
                    shift_init_type=shift_init_type,
                    init_shift=init_shift,
                    init_shift_direction=init_shift_direction,
                    seed=layer_seed,
                )
            )

        self.dropout = nn.Dropout(0.1)

        if use_thermometer:
            self.decode_weights = nn.Parameter(
                torch.ones(1, n_thresholds) / n_thresholds
            )

    def thermometer_encode(self, x: torch.Tensor, hard: bool = False) -> torch.Tensor:
        x_clamped = torch.clamp(x, 0.0, 1.0)
        thresholds = torch.linspace(
            1.0 / (self.n_thresholds + 1),
            self.n_thresholds / (self.n_thresholds + 1),
            self.n_thresholds,
            device=x.device,
            dtype=x.dtype,
        )
        x_expanded = x_clamped.unsqueeze(-1)
        thresholds_expanded = thresholds.unsqueeze(0).unsqueeze(0)
        if hard:
            binary = (x_expanded > thresholds_expanded).float()
        else:
            diff = (x_expanded - thresholds_expanded) * self.encoding_temperature
            binary = torch.sigmoid(diff)
        return binary.view(binary.shape[0], -1)

    def thermometer_decode(self, binary: torch.Tensor) -> torch.Tensor:
        binary_reshaped = binary.view(binary.shape[0], self.embed_dim, self.n_thresholds)
        weights = torch.softmax(self.decode_weights, dim=-1)
        x = (binary_reshaped * weights).sum(dim=-1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = x.view(b * n, d)
        hard = (not self.training) or self._train_hard_thermometer

        if self.use_thermometer:
            h = torch.sigmoid(h)
            z = self.thermometer_encode(h, hard=hard)
        else:
            z = torch.sigmoid(h)

        for forest in self.forest_layers:
            z = forest(z)
            z = self.dropout(z)
            if hard:
                z = (z >= 0.5).float()

        if self.use_thermometer:
            h_out = self.thermometer_decode(z)
        else:
            h_out = z

        return h_out.view(b, n, d)

    def set_temperature(self, temperature: float) -> None:
        for forest in self.forest_layers:
            forest.set_temperature(temperature)

    def set_train_hard_thermometer(self, enabled: bool) -> None:
        self._train_hard_thermometer = bool(enabled)

    def harden(self) -> None:
        for forest in self.forest_layers:
            forest.harden()
