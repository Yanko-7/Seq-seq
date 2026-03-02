import math

import torch
import lightning as L
from src.gptv3 import GPT
from litgpt.utils import chunked_cross_entropy


class ForgeTrace(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPT(config)
        self.save_hyperparameters()
        self.model.init_weights()
        self.config = config

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        logits = self.model(
            input_ids, mask=None, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
        )
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log("train/loss", loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        logits = self.model(
            input_ids, mask=None, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
        )
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log("val/loss", loss, prog_bar=True, batch_size=1, sync_dist=True)
        return loss

    def configure_optimizers(self):
        self.num_iters = 200000
        optimizer = self.model.setup_optimizer(
            unembedding_lr=0.002,
            embedding_lr=0.1,
            matrix_lr=0.01,
            scalar_lr=0.1,
            weight_decay=0.1,
        )

        def lr_lambda(step):
            if step < 2000:
                return (step + 1) / 2000
            progress = min(1.0, (step - 2000) / (self.num_iters - 2000))
            return 0.1 + 0.45 * (1.0 + math.cos(math.pi * progress))

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
                "interval": "step",
            },
        }

    def on_before_optimizer_step(self, optimizer):
        step = self.global_step
        muon_momentum = 0.95 - 0.1 * max(0.0, 1.0 - step / 500.0)
        current_wd = 0.1 * max(0.0, 1.0 - step / self.num_iters)

        for group in optimizer.param_groups:
            if group.get("kind") == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = current_wd
