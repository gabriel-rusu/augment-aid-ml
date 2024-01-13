import lightning as L
import torch
import torch.nn.functional as F
import torchvision
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch import nn

from src.train.utils.initialization import weights_init


class GANWrapper(L.LightningModule):

    def __init__(self, generator, discriminator, z_dim, num_classes, display_every_n_steps, init_weights=weights_init):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.criterion = nn.BCEWithLogitsLoss()
        self.num_classes = num_classes
        self.display_every_n_steps = display_every_n_steps
        self.z_dim = z_dim

        if init_weights is not None:
            self.generator.apply(init_weights)
            self.discriminator.apply(init_weights)

        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        batch_size = len(x)

        generator_opt, discriminator_opt = self.optimizers()

        one_hot_y = F.one_hot(y, self.num_classes).float()
        image_one_hot_y = one_hot_y[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        discriminator_opt.zero_grad()

        fake_noise = torch.randn(batch_size, self.z_dim, device=self.device)
        noise_and_labels = torch.cat([fake_noise, one_hot_y], 1)

        fake = self.generator(noise_and_labels)

        fake_image_and_labels = torch.cat([fake.detach(), image_one_hot_y], 1)
        real_image_and_labels = torch.cat([x, image_one_hot_y], 1)

        fake_pred = self.discriminator(fake_image_and_labels)
        real_pred = self.discriminator(real_image_and_labels)

        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

        discriminator_loss = (fake_loss + real_loss) / 2

        discriminator_loss.backward(retain_graph=True)
        discriminator_opt.step()

        generator_opt.zero_grad()
        fake_image_and_labels = torch.cat([fake, image_one_hot_y], 1)
        fake_pred = self.discriminator(fake_image_and_labels)

        generator_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))
        generator_loss.backward()
        generator_opt.step()

        self.log_dict({
            'generator_loss': generator_loss,
            'discriminator_loss': discriminator_loss,
        }, prog_bar=True, reduce_fx='mean')

        if self.current_epoch % self.display_every_n_steps == 0:
            display_fake = fake[:32]
            display_real = x[:32]
            grid_fake = torchvision.utils.make_grid(display_fake.view(-1, 1, 28, 28))
            grid_real = torchvision.utils.make_grid(display_real.view(-1, 1, 28, 28))

            self.logger.experiment.add_image('fake', grid_fake, self.current_epoch)
            self.logger.experiment.add_image('real', grid_real, self.current_epoch)


        return generator_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        batch_size = len(x)

        one_hot_y = F.one_hot(y, self.num_classes).float()
        image_one_hot_y = one_hot_y[:, :, None, None].repeat(1, 1, x.shape[2], x.shape[3])

        fake_noise = torch.randn(batch_size, self.z_dim, device=self.device)
        noise_and_labels = torch.cat([fake_noise, one_hot_y], 1)

        fake = self.generator(noise_and_labels)

        fake_image_and_labels = torch.cat([fake.detach(), image_one_hot_y], 1)
        real_image_and_labels = torch.cat([x, image_one_hot_y], 1)

        fake_pred = self.discriminator(fake_image_and_labels)
        real_pred = self.discriminator(real_image_and_labels)

        fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
        real_loss = self.criterion(real_pred, torch.ones_like(real_pred))

        discriminator_loss = (fake_loss + real_loss) / 2

        fake_image_and_labels = torch.cat([fake, image_one_hot_y], 1)
        fake_pred = self.discriminator(fake_image_and_labels)

        generator_loss = self.criterion(fake_pred, torch.ones_like(fake_pred))

        self.log_dict({
            'val_generator_loss': generator_loss,
            'val_discriminator_loss': discriminator_loss,
        }, prog_bar=True, reduce_fx='mean')

        return generator_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)

        return generator_optimizer, discriminator_optimizer
