import pytorch_lightning as pl

import torch
from torchmetrics import Accuracy, F1Score
import torch.nn as nn

class DummyModel(pl.LightningModule):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.linear = nn.Linear(10, 1)
        self.accuracy = Accuracy(task="binary", threshold=0.5)
        self.f1_score = F1Score(task="binary", num_classes=2, threshold=0.5)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y)
        acc = self.accuracy(y_hat, y.int())
        f1 = self.f1_score(y_hat, y.int())
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_f1', f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss, 'test_acc': acc, 'test_f1': f1}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)