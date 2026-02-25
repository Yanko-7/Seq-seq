import torch
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MemmapTokenDataset(Dataset):
    """
    极致性能的 Dataset：直接将硬盘上的二进制 Token 映射到虚拟内存。
    依靠操作系统的 Page Cache 机制，实现零拷贝（Zero-copy）级别的数据读取。
    """

    def __init__(self, memmap_path: str | Path, block_size: int):
        super().__init__()
        self.block_size = block_size

        # 假设你在 pack_dataset.py 中保存的是 uint16 格式以节省硬盘
        # mode='r' 表示只读，保证多进程下的安全
        self.data = np.memmap(memmap_path, dtype=np.uint16, mode="r")
        self.total_tokens = len(self.data)

        # 虚拟的 Epoch 长度：在海量数据预训练中，通常不需要严格遍历一遍
        # 这里定义每个 Epoch 采样 100,000 个 Batch
        self.epoch_length = 100_000

    def __len__(self):
        return self.epoch_length

    def __getitem__(self, _):
        # 1. 随机跳跃读取（Random Seek），避免顺序读取时的 Epoch 边界问题
        # block_size + 1 是因为我们需要同时切出输入(x)和目标(y)
        start_idx = torch.randint(
            0, self.total_tokens - self.block_size - 1, (1,)
        ).item()

        # 2. 从内存映射中切片 (极其快速的 O(1) 操作)
        chunk = self.data[start_idx : start_idx + self.block_size + 1]

        # 3. 转换为 PyTorch 张量，并转为 int64 (Embedding 层只认 int64)
        chunk_tensor = torch.from_numpy(chunk.astype(np.int64))

        # 4. 错位构造自回归的 X 和 Y
        x = chunk_tensor[:-1]  # 取前 N 个
        y = chunk_tensor[1:]  # 取后 N 个 (右移一位的预测目标)

        return x, y


class FastTokenDataModule(L.LightningDataModule):
    """
    Lightning 的标准数据管家，负责管理多进程 DataLoader
    """

    def __init__(
        self,
        data_dir: str,
        block_size: int = 2048,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # Setup 会在每张 GPU (每个进程) 上被调用一次
        # 我们在这里指向之前跑好的离线 .bin / .memmap 文件
        train_path = self.data_dir / "train_tokens.bin"
        val_path = self.data_dir / "val_tokens.bin"

        if stage == "fit" or stage is None:
            self.train_dataset = MemmapTokenDataset(train_path, self.block_size)
            self.val_dataset = MemmapTokenDataset(val_path, self.block_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,  # 极其重要：将数据锁在页锁定内存中，加速 CPU 到 GPU 的传输
            shuffle=False,  # 因为在 Dataset 里已经是随机取样了，这里就不需要 shuffle 了
            drop_last=True,  # 丢弃不完整的 batch，防止硬件对齐惩罚
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
        )
