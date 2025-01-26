from typing import Any

import lightning as L
import torch.nn.functional as F
import torch.optim.lr_scheduler
from lightning.pytorch.utilities.types import OptimizerLRScheduler


class ClassifierWrapper(L.LightningModule):

    def __init__(self, classifier, num_classes, image_enhancer=None):
        super().__init__()
        self.classifier = classifier
        self.num_classes = num_classes
        self.image_enhancer = image_enhancer

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.image_enhancer is not None:
            x = self.image_enhancer(y)

        output = self.classifier(x)
        _, pred = torch.max(output, 1)
        loss = F.cross_entropy(output, y)

        self.log('train_acc', torch.sum(pred == y).item() / (len(y)), reduce_fx='mean', prog_bar=True)

        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.image_enhancer is not None:
            x = self.image_enhancer(y)

        output = self.classifier(x)
        _, pred = torch.max(output, 1)
        loss = F.cross_entropy(output, y)

        self.log('val_acc', torch.sum(pred == y).item() / (len(y)), reduce_fx='mean', prog_bar=True)
        self.log('val_loss', loss, reduce_fx='mean', prog_bar=True)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]
