"""
可微分逻辑门网络的实现

本模块实现了可微分逻辑门网络的核心组件，包括标准逻辑层（LogicLayer）
和输入级参数化的逻辑层（LogicLayerIWP），以及用于分类的分组求和模块（GroupSum）。
"""
import torch
import math
import difflogic_cuda, difflogic_cuda_iwp
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor, SinSkipGrad, SigmoidStraightThrough, LinearStraightThrough, bin_gate, weight_init
import itertools
import time

########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    可微分逻辑门网络的核心模块。提供可微分逻辑门层。

    该类实现了逻辑门网络的基本功能，支持CUDA加速和Python实现两种方式。
    逻辑层将输入映射到输出，每个输出神经元通过二元逻辑运算（AND, OR, XOR等）组合两个输入。
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            weight_init_choice='gauss',
            sigma = 1.0,
            res_connect_fraction=0.0,
            residual_freeze=True,
    ):
        """
        初始化逻辑层

        参数:
            in_dim: 输入维度
            out_dim: 输出维度
            device: 计算设备（选项：'cuda' / 'cpu'）
            grad_factor: 梯度因子。对于深层模型（>6层），应增加此值（例如2）以避免梯度消失
            implementation: 实现方式（选项：'cuda' / 'python'）。CUDA实现比Python实现快约100倍
            connections: 初始化逻辑门网络连接的方法（'random' 或 'unique'）
            weight_init_choice: 权重初始化方法（'gauss' 或 'ri'）
            sigma: 权重初始化方差
            res_connect_fraction: 残差连接的比例（0.0表示无残差连接）
            residual_freeze: 若 True，残差连接对应的门权重固定为直通A并关闭梯度；若 False，仅改连接，权重可训练
        """
        super().__init__()
        self.weight_init_choice = weight_init_choice
        self.weights = torch.nn.parameter.Parameter(weight_init(
            (out_dim, 16),
            choice=self.weight_init_choice,
            sigma=sigma,
            device=device
        ))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        # CUDA实现是快速实现。顾名思义，CUDA实现仅在device='cuda'时可用。
        # Python实现存在有两个原因：
        # 1. 提供易于理解的可微分逻辑门网络实现
        # 2. 提供CPU上的可微分逻辑门网络实现
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

        # 梯度掩码（用于避免记录人工固定权重的梯度）
        # False = 固定, True = 可学习
        mask = torch.ones_like(self.weights, dtype=torch.bool)
        self.register_buffer('grad_mask', mask)

        # 残差连接（这会调整self.indices，需要在计算逆连接之前调用！）
        self.res_connect_fraction = res_connect_fraction
        self.residual_freeze = residual_freeze
        self.num_res_connections = int(math.ceil(res_connect_fraction*self.out_dim))
        if self.num_res_connections > 0:
            assert self.in_dim >= self.num_res_connections
            self.setup_res_connections()

        # 逆连接索引（用于提高CUDA实现的反向传播效率）
        if self.implementation == 'cuda':
            # 定义额外的索引以提高CUDA实现反向传播的效率
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)

    def pass_dropout(self, affected_input_indices):
        """
        传递dropout：根据受影响的输入索引，确定哪些输出索引受到影响

        参数:
            affected_input_indices: 受dropout影响的输入索引

        返回:
            受影响的输出索引
        """
        if isinstance(affected_input_indices, list):
            affected_input_indices = torch.tensor(affected_input_indices, dtype=torch.long, device=self.device)
        starts = self.given_x_indices_of_y_start[affected_input_indices]
        ends = self.given_x_indices_of_y_start[affected_input_indices + 1]
        counts = ends - starts

        total = counts.sum()

        # Create a flat index array of all affected outputs
        # Step 1: repeat start indices according to how many outputs each input connects to
        repeated_starts = torch.repeat_interleave(starts, counts)

        # Step 2: create relative offsets within each connection block
        relative_offsets = torch.arange(total, device=self.device) - \
                        torch.repeat_interleave(counts.cumsum(0) - counts, counts)

        # Step 3: compute final indices into self.given_x_indices_of_y
        affected_output_indices = self.given_x_indices_of_y[repeated_starts + relative_offsets]
        affected_output_indices = torch.unique(affected_output_indices)

        return affected_output_indices

    def remove_dropout_mask(self):
        """移除dropout掩码"""
        if hasattr(self, "dropout_mask"):
            del self.dropout_mask

    def setup_res_connections(self):
        """
        设置残差连接：前 num_res_connections 个输出的连接改为 a[i]=i（恒等）。
        若 residual_freeze 为 True，再将这些门的权重固定为直通 A 并关闭梯度。
        """
        # 只改连接：输出 i 的输入 A = 输入 i
        a, b = self.indices
        a[0:self.num_res_connections] = torch.arange(self.num_res_connections, device=self.device).long()
        self.indices = a, b

        # 仅当 residual_freeze 时固定权重并关梯度（残差初始化）
        if self.residual_freeze:
            self.grad_mask[0:self.num_res_connections, :] = False
            self.fix_res_connect_gates()

    def fix_res_connect_gates(self):
        """
        固定残差连接的逻辑门权重

        将前num_res_connections个神经元的权重设置为实现恒等映射（选择输入A）
        """
        # 将逻辑门固定为前馈输入A
        with torch.no_grad():
            self.weights[0:self.num_res_connections, :] = -10
            self.weights[0:self.num_res_connections, 3] = 10

    def reset_fixed_weights(self):
        """重置固定权重（在每次前向传播后调用）"""
        self.fix_res_connect_gates()
        self.remove_dropout_mask()

    def apply_grad_mask(self):
        """在反向传播后应用梯度掩码"""
        if self.w_00 is not None and self.weights.grad is not None:
            self.weights.grad *= self.grad_mask

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量

        返回:
            输出张量
        """
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            print(self.indices[0].dtype, self.indices[1].dtype)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            x = LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                x = LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)

        return x

    def get_connections(self, connections, device='cuda'):
        """
        获取输入到输出的连接索引

        参数:
            connections: 连接方法（'random' 或 'unique'）
            device: 计算设备

        返回:
            (a, b): 两个输入索引张量，形状为(out_dim,)
        """
        assert self.out_dim * 2 >= self.in_dim, '神经元数量 ({}) 必须不小于输入数量 ({}) 的一半，否则无法使用或考虑所有输入。'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    def extra_repr(self):
        """返回模块的字符串表示"""
        return '{}, {}{}{}{}'.format(self.in_dim, self.out_dim,
            ', train' if self.training else 'eval',
            f', weights_init={self.weight_init_choice}',
            f', num_resconnections={self.num_res_connections}' if self.num_res_connections > 0 else '',
        )

class LogicLayerIWP(LogicLayer):
    """
    输入级参数化（Input-wise Parametrization）的逻辑层

    可微分逻辑门网络的核心模块。提供使用输入级参数化的可微分逻辑门层。
    IWP使用4个参数（而非16个）来表示每个逻辑门，从而减少参数数量并提高训练稳定性。
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            weight_init_choice='gauss',
            sigma = 1.0,
            act_fn = torch.sigmoid,
            init_shift=0.0,
            init_shift_direction=None,
            shift_init_enable=True,
            shift_init_type=None,
            res_connect_fraction=0.0,
            residual_freeze=True,
            skip_grad=False,
            random_outage=None,
            random_outage_prob=0.0,
            run_op_on_iwp=False,
            run_original_cuda=False,
    ):
        """
        初始化IWP逻辑层

        参数:
            in_dim: 输入维度
            out_dim: 输出维度
            device: 计算设备
            grad_factor: 梯度因子
            implementation: 实现方式（'cuda' 或 'python'）
            connections: 连接方法（'random' 或 'unique'）
            weight_init_choice: 权重初始化方法
            sigma: 权重初始化方差
            act_fn: 激活函数类型（'SIN01', 'sigmoid', 'sigmoid-st', 'sin-st', 'linear'）
            init_shift: 初始化偏移量
            init_shift_direction: 初始化偏移方向（4位字符串，如'0101'）
            shift_init_enable: 是否执行重尾初始化（shift_init）；False 时跳过
            shift_init_type: 重尾类型（'ri'/'and-or'/'and-or-ri'/'uniform'）；None 时沿用 weight_init_choice
            res_connect_fraction: 残差连接比例
            residual_freeze: 残差门是否固定为直通A并关闭梯度（False=仅改连接，权重可训练）
            skip_grad: 是否跳过梯度计算
            random_outage: 随机中断类型（'const0', 'const1', 'const0.5', 'uniform', 'bernoulli'）
            random_outage_prob: 随机中断概率
            run_op_on_iwp: 是否在IWP上运行原始参数化
            run_original_cuda: 是否运行原始CUDA实现
        """
        self.act_fn_str = act_fn
        super().__init__(in_dim, out_dim,
            device=device,
            weight_init_choice=weight_init_choice,
            sigma=sigma,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
            res_connect_fraction=res_connect_fraction,
            residual_freeze=residual_freeze,
            )

        # 二元输出估计器
        self.act_fn = None
        if self.act_fn_str == "SIN01":
            self.act_fn = self.sin_act
        elif self.act_fn_str == 'sigmoid':
            self.act_fn = torch.sigmoid
        elif self.act_fn_str == 'sigmoid-st':
            self.act_fn = self.sigmoid_st
        elif self.act_fn_str == 'sin-st':
            self.act_fn = self.sin_st
        elif self.act_fn_str == 'linear':
            self.act_fn = self.linear_act

        self.run_op_on_iwp = run_op_on_iwp
        # 输入级参数化，每个神经元只需要4个参数（而非16个）
        if not self.run_op_on_iwp:
            self.weights = torch.nn.parameter.Parameter(weight_init(
                (out_dim, 4),
                choice=self.weight_init_choice,
                sigma=sigma,
                device=device
            ))

            # 重新初始化梯度掩码
            mask = torch.ones_like(self.weights, dtype=torch.bool)
            self.register_buffer('grad_mask', mask)

            # 重尾初始化：仅当 shift_init_enable 且 init_shift != 0 时执行
            self.init_shift = init_shift
            self.init_shift_direction = init_shift_direction
            if shift_init_enable and self.init_shift != 0:
                assert init_shift_direction is not None, init_shift_direction
                init_type = shift_init_type if shift_init_type is not None else self.weight_init_choice
                self.shift_init(init_type, self.init_shift, self.init_shift_direction)

            # 残差连接：IWP 覆盖权重后，仅当 residual_freeze 时再固定残差门为直通 A 并关梯度
            if self.num_res_connections > 0 and self.residual_freeze:
                self.grad_mask[0:self.num_res_connections, :] = False
                self.fix_res_connect_gates()

        # 跳过梯度
        self.skip_grad = skip_grad

        # Dropout（随机中断）
        self.random_outage=random_outage
        self.random_outage_prob=random_outage_prob

    def sin_act(self, x):
        """
        正弦激活函数：将输入映射到[0,1]区间
        """
        return 0.5 + 0.5 * torch.sin(x)

    def sin_st(self, x):
        """正弦直通激活函数（使用符号梯度）"""
        return SinSkipGrad.apply(x)

    def linear_act(self, x):
        """线性直通激活函数"""
        return LinearStraightThrough.apply(x)

    def sigmoid_st(self, x):
        """Sigmoid直通激活函数（使用直通梯度）"""
        return SigmoidStraightThrough.apply(x)

    def fix_res_connect_gates(self):
        """
        重写：固定残差连接的逻辑门权重（适配IWP参数化）
        将逻辑门固定为前馈输入A
        """
        with torch.no_grad():
            if self.act_fn_str == "SIN01":
                self.weights[0:self.num_res_connections, 0] = -torch.pi/2
                self.weights[0:self.num_res_connections, 1] = -torch.pi/2
                self.weights[0:self.num_res_connections, 2] = torch.pi/2
                self.weights[0:self.num_res_connections, 3] = torch.pi/2
            elif self.act_fn_str == 'sigmoid':
                self.weights[0:self.num_res_connections, 0] = -10
                self.weights[0:self.num_res_connections, 1] = -10
                self.weights[0:self.num_res_connections, 2] = 10
                self.weights[0:self.num_res_connections, 3] = 10
            else:
                raise NotImplementedError(self.act_fn_str)

    def shift_init(self, init, shift, shift_combination):
        """
        IWP的重尾初始化

        参数:
            init: 初始化方法（'ri', 'and-or', 'and-or-ri', 'uniform'）
            shift: 偏移量
            shift_combination: 4位字符串，指定每个参数的偏移方向（'0'表示减，'1'表示加）
        """
        with torch.no_grad():
            M = self.weights.shape[0]
            if init is None or init == 'ri':
                for i in range(4):
                    if shift_combination[i] == '0':
                        self.weights[..., i] -= shift
                    else:
                        self.weights[..., i] += shift
            else:
                if init == 'ri':
                    K = 1
                    sign_patterns = torch.tensor([
                        [-1, -1, 1, 1], # A
                    ], dtype=self.weights.dtype)
                if init == 'and-or':
                    K = 2
                    sign_patterns = torch.tensor([
                        [-1, 1, 1, 1], # OR
                        [-1, -1, -1, 1], # AND
                    ], dtype=self.weights.dtype)
                elif init == 'and-or-ri':
                    K = 4
                    sign_patterns = torch.tensor([
                        [-1, 1, 1, 1], # OR
                        [-1, -1, -1, 1], # AND
                        [-1, -1, 1, 1], # A
                        [-1, 1, -1, 1], # B
                    ], dtype=self.weights.dtype)
                elif init == 'uniform':
                    K = 16
                    bool_outputs = torch.tensor(list(itertools.product([0, 1], repeat=4)), dtype=torch.float32)
                    sign_patterns = 2 * bool_outputs - 1
                else:
                    raise NotImplementedError(init)
                probs = np.full(K, 1 / K)
                choices = np.random.choice(K, size=M, p=probs)
                signs_tensor = sign_patterns.to(self.device, dtype=self.weights.dtype)[choices]
                self.weights += shift * signs_tensor


    def extra_repr(self):
        """返回模块的字符串表示"""
        if self.run_op_on_iwp:
            return super().extra_repr()
        return super().extra_repr() + '{}{}{}{}'.format(
            f', init_shift={self.init_shift} ({self.init_shift_direction})' if self.init_shift != 0 else '',
            f', act_fn={self.act_fn_str}',
            f', skip_grad' if self.skip_grad else '',
            f', random_outage={self.random_outage} (prob={self.random_outage_prob})' if self.random_outage is not None else '',
        )

    def forward_python(self, x):
        """
        重写：Python实现的前向传播（使用IWP参数化）
        """
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            # NOTE: torch.int64 == torch.long.
            # print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            # print(self.indices[0].dtype, self.indices[1].dtype)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_gate(a, b, self.act_fn(self.weights))
        else:
            weights = (self.act_fn(self.weights) >= 0.5).to(x.dtype)
            x = bin_gate(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.run_op_on_iwp:
            if self.training:
                w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
                x = LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
            else:
                w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
                with torch.no_grad():
                    x = LogicLayerCudaFunction.apply(
                        x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                    ).transpose(0, 1)
        else:
            if self.training:
                w = self.act_fn(self.weights).to(x.dtype)
                x = IWPLogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
            else:
                with torch.no_grad():
                    w = torch.round(self.act_fn(self.weights)).to(x.dtype)
                    x = IWPLogicLayerCudaFunction.apply(
                        x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                    ).transpose(0, 1)

        return x

########################################################################################################################


class GroupSum(torch.nn.Module):
    """
    分组求和模块

    将逻辑层的输出分组并求和，生成分类logits。支持多个分组、可选的偏置和权重，
    以及温度缩放（softmax temperature）。
    """
    def __init__(self, k: int, tau: float = 1., device='cuda', n_splits: int = 1, for_each_split: bool = False,
        use_bias: bool = False,
        use_weights: bool = False,
        gs_flip: int = 0,
        in_dim=None):
        """
        初始化分组求和模块

        参数:
            k: 期望的实值输出数量，例如类别数
            tau: softmax温度参数。求和后的输出除以tau
            device: 计算设备
            n_splits: 分组数量
            for_each_split: 是否为每个分组使用独立求和
            use_bias: 是否在输出logits中添加偏置
            use_weights: 是否为每个激活使用可学习权重
            gs_flip: 翻转掩码的周期（0表示不翻转）
            in_dim: 输入维度（输入张量的最后一个维度）
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device
        self.n_splits = n_splits
        self.for_each_split = for_each_split
        self.use_bias = use_bias
        self.use_weights = use_weights
        self.in_dim = in_dim
        self.gs_flip = gs_flip

        if self.use_bias:
            if self.for_each_split:
                # Bias shape: (n_splits, k) for individual splits
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.n_splits, self.k, device=device))
            else:
                # Bias shape: (k,) for summed across splits
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.k, device=device))
        else:
            self.register_parameter('bias', None)
        if self.use_weights:
            if self.in_dim is None:
                raise ValueError("in_dim must be provided when use_weights=True")
            self.weights = torch.nn.parameter.Parameter(torch.ones(self.in_dim, device=device))
        else:
            self.register_parameter('weights', None)

        if self.gs_flip > 0:
            assert in_dim is not None
            flip_mask = ((torch.arange(in_dim, device=device)) % self.gs_flip == 0)
            self.register_buffer('flip_mask', flip_mask)

    def forward(self, x):
        """
        执行分组求和操作的前向计算

        参数:
            x: 输入张量

        返回:
            分组求和操作后的输出张量
        """
        if self.use_weights and self.weights is not None:
            assert x.shape[-1] == self.weights.shape[-1], f"x.shape: {x.shape}, self.weights.shape: {self.weights.shape}"
            x = x * self.weights

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        assert x.shape[-1] % self.n_splits == 0, (x.shape, self.n_splits)

        if self.gs_flip > 0:
            x = self.flip(x)

        if self.for_each_split:
            logits = x.reshape(*x.shape[:-1], self.n_splits, self.k, x.shape[-1] // (self.k * self.n_splits)).sum(-1) / self.tau
        else:
            assert x.shape[-1] % (self.k*self.n_splits) == 0, (x.shape, self.k, self.n_splits)
            logits = x.reshape(*x.shape[:-1], self.n_splits, self.k, x.shape[-1] // (self.k * self.n_splits)).sum(-1).sum(-2) / self.tau

        if self.use_bias and self.bias is not None:
            assert logits.shape[-1] == self.bias.shape[-1], f"logits.shape: {logits.shape}, self.bias.shape: {self.bias.shape}"
            logits = logits + self.bias

        # print(f"GS Logits: {logits}")
        return logits

    def flip(self, x):
        """
        翻转每第k个输入（0变1，1变0）

        参数:
            x: 输入张量

        返回:
            翻转后的张量
        """
        assert x.shape[-1] == self.flip_mask.shape[-1], (x.shape, self.flip_mask.shape)

        return x * (~self.flip_mask) + (1-x)*self.flip_mask


    def extra_repr(self):
        return 'k={}, tau={}, device={}, {}{}{}{}{}'.format(
            self.k, self.tau, self.device,
            'ns={}, '.format(self.n_splits) if self.n_splits > 1 else '',
            'for_each_split, ' if self.for_each_split else '',
            'bias, ' if self.use_bias else 'no_bias, ',
            'weights' if self.use_weights else 'no_weights',
            ', gs_flip={}'.format(self.gs_flip) if self.gs_flip != 0 else '',
        )

########################################################################################################################

class LogicLayerCudaFunction(torch.autograd.Function):
    """
    标准逻辑层的CUDA加速前向和反向传播函数
    """
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        """
        CUDA前向传播

        参数:
            x: 输入张量
            a, b: 输入索引
            w: 权重（16个逻辑门的概率分布）
            given_x_indices_of_y_start: 逆连接索引的起始位置
            given_x_indices_of_y: 逆连接索引
        """
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        """
        CUDA反向传播

        参数:
            grad_y: 输出梯度

        返回:
            输入梯度和权重梯度
        """
        cls = LogicLayerCudaFunction

        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            start = time.perf_counter()

        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            end = time.perf_counter()
            duration_ns = 1000000 *(end - start)
            cls.backward_count += 1
            cls.backward_time += duration_ns * cls.timing_measurements_factor
            cls.backward_times.append(duration_ns)
            # print(f"Backward ns: {duration_ns}")

        return grad_x, None, None, grad_w, None, None, None


class IWPLogicLayerCudaFunction(torch.autograd.Function):
    """
    IWP逻辑层的CUDA加速前向和反向传播函数
    """
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        """
        CUDA前向传播（IWP版本）

        参数:
            x: 输入张量
            a, b: 输入索引
            w: 权重（4个概率值，对应4种输入组合）
            given_x_indices_of_y_start: 逆连接索引的起始位置
            given_x_indices_of_y: 逆连接索引
        """
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda_iwp.iwp_forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        """
        CUDA反向传播（IWP版本）

        参数:
            grad_y: 输出梯度

        返回:
            输入梯度和权重梯度
        """
        cls = IWPLogicLayerCudaFunction

        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            start = time.perf_counter()

        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda_iwp.iwp_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda_iwp.iwp_backward_w(x, a, b, grad_y)
        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            end = time.perf_counter()
            duration_ns = 1000000 *(end - start)
            cls.backward_count += 1
            cls.backward_time += duration_ns * cls.timing_measurements_factor
            cls.backward_times.append(duration_ns)
            # print(f"Backward ns: {duration_ns}")

        return grad_x, None, None, grad_w, None, None, None



########################################################################################################################
