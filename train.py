import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, WeightAveraging
from lightning.pytorch.loggers import WandbLogger
from torch.optim.swa_utils import get_ema_avg_fn

from src.datamodule import PackedDataModule

# from src.config import Config
from src.gptv3 import GPTConfig
from src.model import ForgeTrace

torch.serialization.add_safe_globals([GPTConfig])


class EMAWeightAveraging(WeightAveraging):
    def __init__(self):
        super().__init__(avg_fn=get_ema_avg_fn())

    def should_update(self, step_idx=None, epoch_idx=None):
        return (step_idx is not None) and (step_idx >= 2000)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="forge-trace-nanogpt")

    data_module = PackedDataModule(
        train_dir="data/abc-optimized-sep-train/",
        val_dir="data/abc-optimized-sep-val/",
        max_num_tokens=24576,
        batch_size=1,
        num_workers=24,
    )
    config = GPTConfig(
        sequence_len=8192,
        vocab_size=4096,
        n_layer=30,
        n_head=12,
        n_kv_head=4,
        n_embd=768,
        window_pattern="SSSL",
    )
    model = ForgeTrace(config)
    model = torch.compile(model)
    checkpoint_best = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath="checkpoint/nanochat/",
        # 文件名使用 step 变量，方便排序，比如 step005000
        filename="best-step{step:06d}-val_loss{val/loss:.4f}",
        auto_insert_metric_name=False,
        save_top_k=2,
    )
    checkpoint_latest = ModelCheckpoint(
        dirpath="checkpoint/nanochat/",
        # 注意：按步保存时，建议文件名只带 step 或 train_loss。
        # 因为如果你在第 500 步保存，但第 500 步刚好没有运行 validation，获取不到 val_loss 会报错。
        filename="latest-step{step:06d}",
        auto_insert_metric_name=False,
        every_n_train_steps=3000,  # 关键参数：每 3000 个 training step 保存一次
        save_top_k=1,  # 永远只保留最近这 1 个，旧的自动删除
    )
    trainer = L.Trainer(
        max_steps=300000,
        num_nodes=1,
        devices=3,
        val_check_interval=1500,
        check_val_every_n_epoch=None,
        limit_val_batches=650,
        accumulate_grad_batches=3,
        precision="bf16-mixed",
        model_registry="ForgeTrace",
        logger=logger,
        use_distributed_sampler=False,
        callbacks=[checkpoint_best, checkpoint_latest, EMAWeightAveraging()],
    )
    trainer.fit(model, datamodule=data_module)
