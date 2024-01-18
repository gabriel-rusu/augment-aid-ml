import lightning as L
import torch
import torchvision
from torch import nn
from torch.nn import functional as F

from src.train.utils.helpers import combine_sample


class GANAugmentedTrainingWrapper(L.LightningModule):

    def __init__(self, classifier, generator, z_dim, num_classes, display_every_n_steps=10, p_real=0.5):
        super().__init__()
        self.classifier = classifier
        self.generator = generator.eval()
        self.criterion = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.z_dim = z_dim
        self.p_real = p_real
        self.display_every_n_steps = display_every_n_steps

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = len(x)

        one_hot_y = F.one_hot(y, self.num_classes).float()
        noise = torch.randn(batch_size, self.z_dim, device=self.device)
        noise_and_labels = torch.cat([noise, one_hot_y], 1)

        generated_images = self.generator(noise_and_labels)

        target_images = combine_sample(x.clone(), generated_images.clone(), self.p_real)
        labels_hat = self.classifier(target_images.detach())

        # if self.current_epoch % self.display_every_n_steps == 0:
        #     print('shape of generated images: ', generated_images.shape)
        #     print('shape of real images: ', x.shape)
        #
        #     generated_images = generated_images[:32]
        #     grid_generated = torchvision.utils.make_grid(generated_images.view(-1, 1, 28, 28))
        #
        #     real_images = x[:32]
        #     grid_real = torchvision.utils.make_grid(real_images.view(-1, 1, 28, 28))
        #
        #     self.logger.experiment.add_images('generated_images', grid_generated, self.current_epoch)
        #     self.logger.experiment.add_images('real_images', grid_real, self.current_epoch)

        loss = self.criterion(labels_hat, y)

        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)
        self.log('train_acc', torch.sum(torch.argmax(labels_hat, dim=1) == y).item() / (len(y)), reduce_fx='mean',
                 prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = len(x)

        one_hot_y = F.one_hot(y, self.num_classes).float()
        noise = torch.randn(batch_size, self.z_dim, device=self.device)
        noise_and_labels = torch.cat([noise, one_hot_y], 1)

        generated_images = self.generator(noise_and_labels)

        target_images = combine_sample(x.clone(), generated_images.clone(), self.p_real)
        labels_hat = self.classifier(target_images.detach())

        loss = self.criterion(labels_hat, y)

        self.log('val_loss', loss, reduce_fx='mean', prog_bar=True)
        self.log('val_acc', torch.sum(torch.argmax(labels_hat, dim=1) == y).item() / (len(y)), reduce_fx='mean',
                 prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [lr_scheduler]
