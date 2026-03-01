import torch
from litgpt.utils import chunked_cross_entropy
from litgpt.pretrain import initialize_weights
import lightning as L
from src.gptv2 import GPT
from muon import MuonWithAuxAdam
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
        initialize_weights(
            self.trainer,
            self.model,
            n_layer=self.model.config.n_layer,
            n_embd=self.model.config.n_embd,
        )

    def training_step(self, batch):
        # input_ids, cu_seqlens = batch
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        # mask = create_flex_packed_mask(cu_seqlens[0], 32768)
        # logits = self.model(input_ids, mask=mask)
        logits = self.model(input_ids, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log("train_loss", loss, prog_bar=True, batch_size=1)
        return loss

    def validation_step(self, batch):
        # input_ids, cu_seqlens = batch
        input_ids = batch["input_ids"]
        cu_seqlens = batch["cu_seqlens"]
        max_seq_len = batch["max_seq_len"]
        # mask = create_flex_packed_mask(cu_seqlens[0], 32768)
        # logits = self.model(input_ids, mask=mask)
        logits = self.model(input_ids, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
        loss = chunked_cross_entropy(logits[..., :-1, :], input_ids[..., 1:])
        self.log("val_loss", loss, batch_size=1)
        return loss

    def configure_optimizers(self):
        warmup_steps = 500
        muon_params = []
        adam_params = []

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "wte" in name or "lm_head" in name:
                adam_params.append(p)
            elif p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)

        param_groups = [
            dict(params=muon_params, use_muon=True, lr=0.024, weight_decay=0.01),
            dict(
                params=adam_params,
                use_muon=False,
                lr=4e-4,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            ),
        ]

        optimizer = MuonWithAuxAdam(param_groups)
        # optimizer = torch.optim.AdamW(
        #     self.model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        # )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(step / warmup_steps, 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
