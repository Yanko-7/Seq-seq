import torch
import lightning as L
from src.model import ForgeTrace

# from src.config import Config
from src.gptv3 import GPTConfig
from src.datamodule import PackedDataModule

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="forge-trace-nanogpt")

    data_module = PackedDataModule(
        train_dir="data/abc-optimized-sep-train/",
        val_dir="data/abc-optimized-sep-val/",
        max_num_tokens=32768,
        batch_size=1,
        num_workers=16,
    )
    config = GPTConfig(
        sequence_len=4096,
        vocab_size=4096,
        n_layer=16,
        n_head=16,
        n_kv_head=16,
        n_embd=1024,
        window_pattern="SSSL",
    )
    model = ForgeTrace(config)
    model = torch.compile(model)
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoint/nanochat/",
        filename="epoch{epoch:02d}-val_loss{val/loss:.4f}",
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        max_steps=200000,
        num_nodes=1,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        limit_val_batches=500,
        # accumulate_grad_batches=1,
        precision="bf16-mixed",
        model_registry="ForgeTrace",
        callbacks=checkpoint_callback,
        logger=logger,
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=data_module)
