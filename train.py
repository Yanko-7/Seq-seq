import torch
import lightning as L
from src.model import ForgeTrace

# from src.config import Config
from src.gptv3 import GPTConfig
from src.datamodule import PackedDataModule

from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="forge-trace-nanogpt")

    data_module = PackedDataModule(
        train_dir="/cache/yanko/dataset/abc-optimized-sep-train/",
        val_dir="/cache/yanko/dataset/abc-optimized-sep-val/",
        max_num_tokens=32768,
        batch_size=1,
        num_workers=20,
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
    trainer = L.Trainer(
        max_steps=400000,
        val_check_interval=2000,
        check_val_every_n_epoch=None,
        accumulate_grad_batches=4,
        precision="bf16-mixed",
        model_registry="ForgeTrace",
        # gradient_clip_val=1.0,
        logger=logger,
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=data_module)
