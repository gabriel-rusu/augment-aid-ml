import lightning as L
import torch
from torch.nn import functional as F

from src.train.utils.helpers import combine_sample


class VAEAugmentedTrainingWrapper(L.LightningModule):

    def __init__(self, classifier, vae, train_img_original, display_every_n_steps=10, p_real=0.5):
        super().__init__()
        self.classifier = classifier
        self.vae = vae.eval()
        self.p_real = p_real
        self.dataloader = train_img_original
        self.train_img_original = iter(train_img_original)
        self.valid_img_original = iter(train_img_original)
        self.display_every_n_steps = display_every_n_steps

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_orig, y_orig = next(self.train_img_original)
        x_orig = x_orig.to(self.device)
        y_orig = y_orig.to(self.device)
        x_hat, _, _ = self.vae(x_orig.view(x_orig.shape[0], -1))

        y = torch.cat([y, y_orig], 0)
        x_hat = x_hat.view(-1, 1, 28, 28)
        target_images = torch.cat([x.clone(), x_hat.clone()], 0)
        labels_hat = self.classifier(target_images.detach())

        loss = F.cross_entropy(labels_hat, y)


        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        self.log('train_acc', torch.sum(torch.argmax(labels_hat, dim=1) == y).item() / (len(y)), reduce_fx='mean',
                 prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_orig, y_orig = next(self.valid_img_original)
        x_orig = x_orig.to(self.device)
        y_orig = y_orig.to(self.device)
        x_hat, _, _ = self.vae(x_orig.view(x_orig.shape[0], -1))

        y = torch.cat([y, y_orig], 0)
        x_hat = x_hat.view(-1, 1, 28, 28)
        target_images = torch.cat([x.clone(), x_hat.clone()], 0)
        labels_hat = self.classifier(target_images.detach())

        loss = F.cross_entropy(labels_hat, y)

        self.log('val_loss', loss, reduce_fx='mean', prog_bar=True)
        self.log('val_acc', torch.sum(torch.argmax(labels_hat, dim=1) == y).item() / (len(y)), reduce_fx='mean',
                 prog_bar=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.train_img_original = iter(self.dataloader)
        self.valid_img_original = iter(self.dataloader)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]
