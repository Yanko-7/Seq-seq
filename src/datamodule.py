import torch
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import random
import litdata as ld
from src.tokenizer import BRepTokenType


def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)


class PackedDataset(IterableDataset):
    def __init__(self, input_dir: str, max_num_tokens: int, buffer_size: int = 10000):
        self.stream_ds = ld.StreamingDataset(input_dir=input_dir)
        self.max_num_tokens = max_num_tokens
        self.buffer_size = buffer_size

    def __iter__(self):
        stream_iter = iter(self.stream_ds)

        doc_buffer = []
        tokens = []
        current_len = 0
        max_num_docs = next_multiple_of_n(
            self.max_num_tokens // 1000, n=128
        )  # median doc length is ~1000

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
                    _cum_lengths = torch.full((max_num_docs,), self.max_num_tokens)
                    actual_docs = len(cum_lengths)
                    safe_docs = min(actual_docs, max_num_docs - 1)
                    _cum_lengths[:safe_docs] = cum_lengths[:safe_docs]
                    yield (merged, _cum_lengths, self.max_num_tokens, actual_docs)
                    tokens = []
                    current_len = 0


def custom_packed_collate_fn(batch):
    input_ids, cu_seqlens, max_seq_len, sample_nums = batch[0]
    return (input_ids.unsqueeze(0), cu_seqlens.unsqueeze(0), max_seq_len, sample_nums)


class PackedDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        val_dir: str = None,
        max_num_tokens: int = 16384,
        batch_size: int = 1,
        num_workers: int = 16,
        buffer_size: int = 10000,
    ):
        super().__init__()
        self.save_hyperparameters()

    def setup(self, stage: str = None):
        self.train_ds = PackedDataset(
            input_dir=self.hparams.train_dir,
            max_num_tokens=self.hparams.max_num_tokens,
            buffer_size=self.hparams.buffer_size,
        )
        if self.hparams.val_dir:
            self.val_ds = PackedDataset(
                input_dir=self.hparams.val_dir,
                max_num_tokens=self.hparams.max_num_tokens,
                buffer_size=self.hparams.buffer_size // 2,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_packed_collate_fn,
        )

    def val_dataloader(self):
        if not self.hparams.val_dir:
            return None
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=max(1, self.hparams.num_workers // 2),
            pin_memory=True,
            drop_last=False,
            collate_fn=custom_packed_collate_fn,
        )
