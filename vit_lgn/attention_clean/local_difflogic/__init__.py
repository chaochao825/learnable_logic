"""
difflogic 模块：可微分逻辑门网络的实现

本模块提供了可微分逻辑门网络的核心组件，包括：
- LogicLayer: 标准逻辑层，实现可微分逻辑门网络
- LogicLayerIWP: 输入级参数化（Input-wise Parametrization）的逻辑层
- GroupSum: 分组求和模块，用于将逻辑层输出聚合为分类结果
- LogicLayerCudaFunction: CUDA加速的前向和反向传播函数
- IWPLogicLayerCudaFunction: IWP版本的CUDA加速前向和反向传播函数
"""
from .difflogic import LogicLayer, GroupSum, LogicLayerIWP, LogicLayerCudaFunction, IWPLogicLayerCudaFunction
