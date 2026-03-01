import torch
import lightning as L
from src.gptv3 import GPT
from litgpt.utils import chunked_cross_entropy
from torch.nn.attention.flex_attention import (
    create_block_mask,
)


def create_flex_packed_mask(cu_seqlens: torch.Tensor, max_seq_len: int):
    cu_seqlens = torch.atleast_2d(cu_seqlens)
    pos = torch.arange(max_seq_len, device=cu_seqlens.device)
    doc_ids = torch.stack(
        [torch.bucketize(pos, seq[1:], right=False) for seq in cu_seqlens]
    )

    def mask_mod(b, h, q, kv):
        return (q >= kv) & (doc_ids[b, q] == doc_ids[b, kv])

    return create_block_mask(
        mask_mod, B=1, H=None, Q_LEN=max_seq_len, KV_LEN=max_seq_len, _compile=True
    )


class ForgeTrace(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = GPT(config)
        self.save_hyperparameters()
        self.config = config

    def on_train_start(self):
        self.model.init_weights()

    def training_step(self, batch):
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        logits = self.model(
            input_ids, mask=None, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
        )
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch):
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        logits = self.model(
            input_ids, mask=None, cu_seqlens=cu_seqlens, max_seqlen=max_seq_len
        )
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log(
            "val_loss", loss, prog_bar=True, batch_size=1, on_step=True, on_epoch=False
        )
        return loss

    def configure_optimizers(self):
        warmup_steps = 500
        optimizer = self.model.setup_optimizer()
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(step / warmup_steps, 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
