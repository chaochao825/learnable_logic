"""
数据加载和预处理管道。

本模块实现了：
- CIFAR-10/100 数据集的加载和预处理
- 数据增强（随机翻转、随机裁剪）
- 温度编码变换（支持多种阈值数量）
- 数据集缓存机制
"""

import math
import random
from pathlib import Path
from typing import Callable, Optional

import torch
import torchvision
from torch.utils.data import Dataset, Subset
from torchvision.transforms import v2

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

# 项目根目录和数据目录路径
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"


class CachedDataset(Dataset):
    """
    读取缓存的 torch 张量并应用可选数据增强的数据集类。

    用于从磁盘加载预处理后的数据，避免每次训练时重复预处理。
    """

    def __init__(self, data_dir: Path, transform: Optional[Callable] = None):
        """
        初始化缓存数据集。

        Args:
            data_dir: 缓存数据目录（包含 .pt 文件）
            transform: 可选的数据变换函数
        """
        self.data_dir = data_dir
        self.transform = transform if transform is not None else (lambda x: x)
        self.files = sorted([p for p in data_dir.glob("*.pt")])

    def __len__(self) -> int:
        """
        返回数据集大小。

        Returns:
            int: 数据集中的样本数量
        """
        return len(self.files)

    def __getitem__(self, idx: int):
        """
        获取指定索引的数据样本。

        Args:
            idx: 样本索引

        Returns:
            tuple: (图像张量, 标签)
        """
        data = torch.load(self.files[idx], map_location="cpu")
        img, label = data["img"], data["label"]
        img = self.transform(img)
        return img, label


def preprocess_dataset(
    dataset_class,
    data_dir: Path,
    save_dir: Path,
    num_samples_required: Optional[int],
    transform: Callable,
):
    """
    使用确定性变换将基础数据集物化到磁盘。

    将数据集预处理后保存为 .pt 文件，以便后续快速加载。

    Args:
        dataset_class: 数据集类（如 CIFAR10）
        data_dir: 原始数据目录
        save_dir: 保存预处理数据的目录
        num_samples_required: 需要的样本数量（如果为 None，使用全部数据）
        transform: 数据变换函数
    """

    save_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [p for p in save_dir.glob("*.pt")]
    base_dataset = None

    if num_samples_required is None:
        base_dataset = dataset_class(root=data_dir, train=True, transform=transform, download=True)
        num_samples_required = len(base_dataset)

    if len(existing_files) >= num_samples_required:
        return

    if base_dataset is None:
        base_dataset = dataset_class(root=data_dir, train=True, transform=transform, download=True)

    idx_counter = len(existing_files)
    total_generated = len(existing_files)
    while total_generated < num_samples_required:
        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            save_path = save_dir / f"{idx_counter:06d}.pt"
            torch.save({"img": img, "label": label}, save_path)
            idx_counter += 1
            total_generated += 1
            if total_generated >= num_samples_required:
                break


def seed_worker(worker_id: int):
    """
    为数据加载器的工作进程设置随机种子。

    Args:
        worker_id: 工作进程 ID
    """
    worker_seed = torch.initial_seed() % 2**32
    if np is not None:
        np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataset_class(dataset_name: str):
    """
    根据数据集名称获取对应的数据集类。

    Args:
        dataset_name: 数据集名称（"cifar-10" 或 "cifar-100"）

    Returns:
        Dataset 类

    Raises:
        ValueError: 如果数据集名称不支持
    """
    if dataset_name == "cifar-10":
        return torchvision.datasets.CIFAR10
    if dataset_name == "cifar-100":
        return torchvision.datasets.CIFAR100
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def temperature_encoding(n_thresholds: int):
    """
    创建温度编码变换函数。

    将输入值编码为多个二进制位（温度计编码）。

    Args:
        n_thresholds: 阈值数量

    Returns:
        Callable: 温度编码变换函数
    """
    return lambda x: torch.cat([(x > (i + 1) / (n_thresholds + 1)).float() for i in range(n_thresholds)], dim=0)


def get_transforms(args):
    """
    根据命令行参数获取数据变换列表。

    Args:
        args: 命令行参数对象

    Returns:
        tuple: (基础变换列表, 数据增强变换列表)

    Raises:
        ValueError: 如果编码方式不支持
    """
    transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    augment_transform_list: list = []
    if args.augment:
        augment_transform_list = [
            # 基础空间增强
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomCrop(size=32, padding=4),
            # 颜色和对比度增强
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
            v2.RandomGrayscale(p=0.1),  # 随机灰度化（10%概率）
            v2.RandomAutocontrast(p=0.1),  # 随机自动对比度（10%概率）
            # 以下增强已经证明无用
            # v2.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # 随机擦除（后续添加，效果不佳）
            # v2.RandAugment(num_ops=2, magnitude=9),  # RandAugment（后续添加）
            # v2.RandomRotation(degrees=15),  # 随机旋转（效果为负，已禁用）
        ]

    encoding_resolutions = [1, 3, 7, 15, 23, 31]
    encoding_map = {"real-input": lambda x: x}
    encoding_map.update({f"{n}-thresholds": temperature_encoding(n) for n in encoding_resolutions})

    if args.data_encoding not in encoding_map:
        raise ValueError(f"Unknown encoding: {args.data_encoding}")

    threshold_transform = encoding_map[args.data_encoding]
    transform_list.append(v2.Lambda(threshold_transform))

    return transform_list, augment_transform_list


def load_dataset(args):
    """
    加载数据集并创建数据加载器。

    Args:
        args: 命令行参数对象

    Returns:
        tuple: (训练数据加载器, 验证数据加载器, 测试数据加载器, 训练集评估用加载器_无增强, 训练批次数, 变换函数)
    """
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_class = get_dataset_class(args.dataset)

    transform_list, augment_transform_list = get_transforms(args)
    transforms_eval = v2.Compose(transform_list)
    transforms_train = v2.Compose(augment_transform_list + transform_list)

    cifar_data_dir = DATA_ROOT / args.dataset

    full_train_aug = dataset_class(cifar_data_dir, train=True, transform=transforms_train, download=True)
    full_train_eval = dataset_class(cifar_data_dir, train=True, transform=transforms_eval, download=True)

    total_train = len(full_train_aug)
    valid_ratio = max(0.0, min(1.0, args.valid_set_size))
    n_valid = int(total_train * valid_ratio)
    n_valid = min(n_valid, total_train)
    n_train = total_train - n_valid

    generator = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(total_train, generator=generator)
    train_indices = perm[:n_train]
    valid_indices = perm[n_train:]

    train_set = Subset(full_train_aug, train_indices)
    train_set_eval = Subset(full_train_eval, train_indices)
    valid_set = Subset(full_train_eval, valid_indices) if n_valid > 0 else None

    test_set = dataset_class(cifar_data_dir, train=False, transform=transforms_eval, download=True)

    num_workers = getattr(args, "num_workers", 4)
    rng = torch.Generator().manual_seed(args.seed)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=rng,
    )

    train_eval_loader = torch.utils.data.DataLoader(
        train_set_eval,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=num_workers,
    )

    valid_loader = None
    if valid_set is not None and len(valid_set) > 0:
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    num_train_batches = len(train_set) // args.batch_size
    transform = lambda x: x
    return train_loader, valid_loader, test_loader, train_eval_loader, num_train_batches, transform


def num_channels_of_dataset(args):
    """
    根据数据集和编码方式计算输入通道数。

    Args:
        args: 命令行参数对象

    Returns:
        int: 输入通道数

    Raises:
        ValueError: 如果数据集不支持
    """
    if "cifar" not in args.dataset:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    base_channels = 3
    if args.data_encoding == "real-input":
        return base_channels
    n_thresholds = int(args.data_encoding.split("-")[0])
    return base_channels * n_thresholds


def class_count_of_dataset(args):
    """
    根据数据集名称获取类别数量。

    Args:
        args: 命令行参数对象

    Returns:
        int: 类别数量

    Raises:
        ValueError: 如果数据集不支持
    """
    if args.dataset == "cifar-10":
        return 10
    if args.dataset == "cifar-100":
        return 100
    raise ValueError(f"Unknown dataset: {args.dataset}")
