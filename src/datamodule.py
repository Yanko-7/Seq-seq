from tqdm import tqdm

import torch
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import random
import litdata as ld
from src.tokenizer import BRepTokenType
from itertools import chain


def next_multiple_of_n(v: float | int, *, n: int) -> int:
    return ((int(v) + n - 1) // n) * n


class PackedDataset(IterableDataset):
    def __init__(
        self,
        input_dir: str,
        max_num_tokens: int,
        buffer_size: int = 10000,
        shuffle: bool = True,
    ):
        self.stream_ds = ld.StreamingDataset(input_dir=input_dir, shuffle=shuffle)
        self.max_num_tokens = max_num_tokens
        self.buffer_size = buffer_size
        self.max_num_docs = next_multiple_of_n(self.max_num_tokens // 1000, n=128)

    def _iter_once(self):
        """Iterate through the StreamingDataset once, yielding packed batches."""
        stream_iter = iter(self.stream_ds)
        doc_buffer = []
        tokens = []
        current_len = 0
        while True:
            try:
                while len(doc_buffer) < self.buffer_size:
                    doc_buffer.append(next(stream_iter))
            except StopIteration:
                pass
            if not doc_buffer:
                break

            random.shuffle(doc_buffer)

            while doc_buffer:
                item = doc_buffer.pop()
                tokens.append(item)
                current_len += len(item)
                if current_len >= self.max_num_tokens:
                    merged = torch.cat(tokens)[: self.max_num_tokens]
                    cum_lengths = torch.nonzero(merged == BRepTokenType.BOS)[:, 0]
                    _cum_lengths = torch.full(
                        (self.max_num_docs,), self.max_num_tokens, dtype=torch.int32
                    )
                    actual_docs = len(cum_lengths)
                    safe_docs = min(actual_docs, self.max_num_docs - 1)
                    _cum_lengths[:safe_docs] = cum_lengths[:safe_docs]
                    max_seq_len = int(torch.diff(_cum_lengths).max().item())
                    yield (
                        merged,
                        _cum_lengths,
                        max_seq_len,
                    )
                    tokens = []
                    current_len = 0

    def __iter__(self):
        while True:
            yield from self._iter_once()


def custom_packed_collate_fn(batch):
    input_ids, cu_seqlens, max_seq_len = batch[0]
    return {
        "input_ids": input_ids.unsqueeze(0),
        "cu_seqlens": cu_seqlens,
        "max_seq_len": max_seq_len,
    }


class PackedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        max_num_tokens: int = 16384,
        batch_size: int = 1,
        num_workers: int = 16,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train_ds = PackedDataset(
            input_dir=self.hparams.train_dir,
            max_num_tokens=self.hparams.max_num_tokens,
            buffer_size=self.hparams.buffer_size,
            shuffle=True,
        )
        if self.hparams.val_dir:
            self.val_ds = PackedDataset(
                input_dir=self.hparams.val_dir,
                max_num_tokens=self.hparams.max_num_tokens,
                buffer_size=self.hparams.buffer_size,
                shuffle=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=custom_packed_collate_fn,
        )

    def val_dataloader(self):
        if not self.hparams.val_dir:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=max(1, self.hparams.num_workers / 2),
            pin_memory=True,
            collate_fn=custom_packed_collate_fn,
        )


if __name__ == "__main__":
    max_num_tokens = 32768
    dm = PackedDataModule(
        train_dir="data/abc-optimized-sep-train/",
        val_dir="data/abc-optimized-sep-val/",
        max_num_tokens=max_num_tokens,
        batch_size=1,
        num_workers=40,
    )
    dm.setup()

    global_min_id, global_max_id = float("inf"), float("-inf")
    all_lengths = []

    loaders = chain(dm.train_dataloader(), dm.val_dataloader())

    for batch_idx, (input_ids, cu_seqlens) in enumerate(
        tqdm(loaders, desc="Testing Datasets")
    ):
        global_min_id = min(global_min_id, input_ids.min().item())
        global_max_id = max(global_max_id, input_ids.max().item())

        if (cu_seqlens < 0).any() or (cu_seqlens > max_num_tokens).any():
            raise ValueError(f"Batch {batch_idx}: cu_seqlens 越界！\n{cu_seqlens}")

        seq_lens = torch.diff(cu_seqlens)
        if (seq_lens < 0).any():
            raise ValueError(
                f"Batch {batch_idx}: cu_seqlens 非单调递增！\n{cu_seqlens}"
            )

        valid_lens = seq_lens[seq_lens > 0]

        # 排除每个 batch 最后一个大概率被截断的序列
        valid_lens = valid_lens[:-1]

        if len(valid_lens) > 0:
            all_lengths.append(valid_lens)

    all_lengths = torch.cat(all_lengths).float() if all_lengths else torch.tensor([])

    print(f"\n--- Token ID 统计 ---")
    print(f"全局最小 token id: {int(global_min_id)}")
    print(f"全局最大 token id: {int(global_max_id)}")

    print(f"\n--- 序列长度统计 ---")
    if len(all_lengths) > 0:
        len_2_count = (all_lengths == 2).sum().item()
        lengths_without_2 = all_lengths[all_lengths != 2]
        min_without_2 = (
            int(lengths_without_2.min().item()) if len(lengths_without_2) > 0 else "无"
        )

        unique_lens = torch.unique(all_lengths)
        top_20_shortest = torch.sort(unique_lens)[0][:20].int().tolist()

        print(f"有效样本总数 (已排除截断): {len(all_lengths)}")
        print(f"最长长度: {int(all_lengths.max().item())}")
        print(f"平均长度: {all_lengths.mean().item():.2f}")
        print(f"中位数长度: {all_lengths.median().item():.2f}")
        print(f"长度为2的序列数量: {int(len_2_count)}")
        print(f"排除长度2后的最短长度: {min_without_2}")
        print(f"最短的前20个不同长度: {top_20_shortest}")
    else:
        print("未检测到有效序列长度。")
