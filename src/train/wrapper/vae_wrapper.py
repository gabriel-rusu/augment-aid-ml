import lightning as L
import torch
import torchvision
from torch.nn import functional as F


class VAEWrapper(L.LightningModule):


    def __init__(self, vae, display_every_n_steps=10):
        super().__init__()
        self.vae = vae
        self.display_every_n_steps = display_every_n_steps

    def loss_function(self, x_hat, x, mean, log_var):
        # check if all values are in range [0, 1]
        assert torch.all(x_hat >= 0.) and torch.all(x_hat <= 1.)
        x = x / 2 + 0.5 # unnormalize
        bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return bce + kld


    def training_step(self, batch, batch_idx):

        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat, mean, log_var = self.vae(x)
        loss = self.loss_function(x_hat, x, mean, log_var)

        self.log('train_loss', loss, reduce_fx='mean', prog_bar=True)

        if self.current_epoch % self.display_every_n_steps == 0:
            generated_images = x_hat.clone().detach()[0:32]
            generated_images = generated_images.view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(generated_images)
            self.logger.experiment.add_images('generated_images', torch.unsqueeze(grid,1), self.current_epoch)


        return loss


    def validation_step(self, batch, batch_idx):

        x, y = batch
        x = x.view(x.size(0), -1)
        x_hat, mean, log_var = self.vae(x)
        loss = self.loss_function(x_hat, x, mean, log_var)

        self.log('val_loss', loss, reduce_fx='mean', prog_bar=True)

        if self.current_epoch % self.display_every_n_steps == 0:
            generated_images = x_hat.clone().detach()[0:32]
            generated_images = generated_images.view(-1, 1, 28, 28)
            grid = torchvision.utils.make_grid(generated_images)
            self.logger.experiment.add_images('generated_images', torch.unsqueeze(grid,1), self.current_epoch)

        return loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.vae.parameters(), lr=1e-3)