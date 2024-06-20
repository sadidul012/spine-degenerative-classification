from tqdm.auto import tqdm
import torch
import torch.nn as nn
import timm
import lightning as pl
from config import CONFIG

import cv2

cv2.setNumThreads(0)
tqdm.pandas()


class TimmModel(nn.Module):
    def __init__(self, pretrained=False):
        super(TimmModel, self).__init__()

        self.encoder = timm.create_model(
            CONFIG["backbone"],
            num_classes=CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained
        )
        hdim = 1
        if 'efficient' in CONFIG["backbone"]:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in CONFIG["backbone"]:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        # self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=CONFIG["drop_rate"], bidirectional=True, batch_first=True)
        # self.head = nn.Sequential(
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(CONFIG["drop_rate_last"]),
        #     nn.LeakyReLU(0.1),
        #     nn.Linear(256, CONFIG["out_dim"]),
        #     nn.Linear(hdim, CONFIG["out_dim"]),
        #     nn.Sigmoid()
        # )
        self.head = nn.Sequential(
            nn.Linear(hdim, CONFIG["out_dim"]),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.encoder(x)
        # feat, _ = self.lstm(feat)
        feat = self.head(feat)
        return feat

    def save_state_dict(self, path):
        torch.save(self.state_dict(), path)

    def load_state_dict_from_path(self, path):
        weights = torch.load(path)
        self.load_state_dict(weights)


class LumbarLightningModel(pl.LightningModule):
    def __init__(self, pretrained=False):
        self.save_hyperparameters()
        super().__init__()
        self.model = TimmModel(pretrained=pretrained)
        class_weights = torch.tensor([1, 2, 4], dtype=torch.float32)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def save(self, path):
        self.model.save_state_dict(path)

    def load(self, path):
        self.model.load_state_dict_from_path(path)

    def forward(self, images):
        return self.model(images)

    def shared_step(self, batch):
        images, labels = batch[0], batch[1]
        logits = self.forward(images)
        loss = self.loss_fn(logits, labels.to(torch.int64))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=CONFIG['lr'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"], eta_min=0)
        return [optimizer], [scheduler]

