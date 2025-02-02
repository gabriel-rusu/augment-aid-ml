# Training a GAN on Fashion MNIST

###Importing modules

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.dataset.FashionMNISTDataModule import FashionMNISTDataModule
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.train.utils.helpers import show_train_images_sample
from src.train.wrapper.gan_wrapper import GANWrapper
from src.utils.constants import Paths
from src.utils.helpers import detect_device

###Setting up the data

BATCH_SIZE = 512

###Downloading and preparing the data

datamodule = FashionMNISTDataModule(Paths.DATA_DIR, BATCH_SIZE)
datamodule.setup('fit')

print('Training set has {} instances'.format(len(datamodule.train_dataloader()) * BATCH_SIZE))
print('Validation set has {} instances'.format(len(datamodule.val_dataloader()) * BATCH_SIZE))

###Defining the model hyperparameters

z_dim = 128
generator_input_dim = z_dim + datamodule.num_classes()
input_channels = 1
discriminator_input_dim = input_channels + datamodule.num_classes()

###Visualizing the data

show_train_images_sample(datamodule)

###Defining the model

generator = Generator(generator_input_dim, input_channels)
discriminator = Discriminator(discriminator_input_dim)

gan_wrapper = GANWrapper(generator, discriminator, z_dim, datamodule.num_classes(),
                         display_every_n_steps=100)

###Adding logging and checkpointing

loggers = [
    TensorBoardLogger(Paths.LOGS_DIR, name='gan-training.logs', log_graph=True, version='version-1.0'),
    CSVLogger(Paths.LOGS_DIR, name='gan-training.logs', version='version-1.0')
]
checkpoint_callback = ModelCheckpoint(dirpath=Paths.MODEL_CHECKPOINT_DIR,
                                      filename='gan-wrapper', save_top_k=1,
                                      monitor='val_generator_loss')

###Training the model

trainer = L.Trainer(default_root_dir=Paths.MODEL_CHECKPOINT_DIR, max_epochs=10000, callbacks=[checkpoint_callback],
                    logger=loggers, accelerator=detect_device(), enable_checkpointing=True, log_every_n_steps=50)

trainer.fit(gan_wrapper, datamodule=datamodule)
