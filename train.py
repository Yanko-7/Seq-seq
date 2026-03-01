import torch
import lightning as L
from src.model import ForgeTrace
from src.config import Config
from src.datamodule import PackedDataModule

from lightning.pytorch.loggers import WandbLogger

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    logger = WandbLogger(project="forge-trace")

    data_module = PackedDataModule(
        train_dir="data/abc-optimized-sep-train/",
        # val_dir="data/abc-optimized-sep-val/",
        max_num_tokens=32768,
        batch_size=1,
        num_workers=20,
    )
    yaml_config = Config.from_file("src/model_config.yaml")
    model = ForgeTrace(yaml_config)
    model = torch.compile(model)
    trainer = L.Trainer(
        max_steps=150000,
        # val_check_interval=200,
        check_val_every_n_epoch=None,
        # accumulate_grad_batches=2,
        precision="bf16-mixed",
        model_registry="ForgeTrace",
        # gradient_clip_val=1.0,
        logger=logger,
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=data_module)
