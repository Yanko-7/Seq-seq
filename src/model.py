import torch
import litgpt
from litgpt.pretrain import initialize_weights
import lightning as L


class ForgeTrace(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = litgpt.GPT(config)

    def on_train_start(self):
        initialize_weights(
            self.trainer,
            self.model,
            n_layer=self.model.config.n_layer,
            n_embd=self.model.config.n_embd,
        )

    def training_step(self, batch):
        input_ids = batch.long()
        logits = self.model(input_ids)
        loss = litgpt.utils.chunked_cross_entropy(
            logits[..., :-1, :], input_ids[..., 1:]
        )
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 500
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=4e-4, weight_decay=0.1, betas=(0.9, 0.95)
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: min(step / warmup_steps, 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
