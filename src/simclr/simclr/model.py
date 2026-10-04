"""Author SimCLR architecture with all-patch mean pooling and full-state resume."""

import pytorch_lightning as pl
import torch
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import SimCLRProjectionHead
from monai.networks.nets import ViT
from torch.nn import functional as F


class SimCLRModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(vars(hparams) if hasattr(hparams, "__dict__") else hparams)
        self.backbone = ViT(
            in_channels=1, img_size=(96, 96, 96), patch_size=(16, 16, 16),
            hidden_size=768, mlp_dim=3072, num_layers=12, num_heads=12,
            classification=False, save_attn=True,
        )
        self.projection_head = SimCLRProjectionHead(768, 768, 2048)
        self.criterion = NTXentLoss()  # Preserve the author's actual library default.

    def forward(self, x):
        tokens, _ = self.backbone(x)
        # classification=False provides 216 patch tokens and no CLS token.
        h = tokens.mean(dim=1)
        return F.normalize(self.projection_head(h), dim=1)

    def training_step(self, batch, batch_idx):
        x0, x1 = batch
        loss = self.criterion(self(x0["image"]), self(x1["image"]))
        self.log("train_loss_ssl", loss, on_step=True, on_epoch=True,
                 batch_size=x0["image"].shape[0])
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=5e-4,
            betas=(0.9, 0.999),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-6,
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def initialize_encoder(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        backbone = {key[len("backbone."):]: value for key, value in state.items()
                    if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone, strict=True)
