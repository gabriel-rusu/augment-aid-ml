###Importing modules

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.dataset.FashionMNISTDataModule import FashionMNISTDataModule
from src.train.utils.helpers import show_train_images_sample
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

input_dim = 28 * 28
hidden_dim = 400
latent_dim = 200

###Defining the model

from src.train.wrapper.vae_wrapper import VAEWrapper
from src.models.vae import VAE

vae = VAE(input_dim, hidden_dim, latent_dim, detect_device()).to(detect_device())
vae_wrapper = VAEWrapper(vae, display_every_n_steps=100)

###Adding logging and checkpointing

loggers = [
    TensorBoardLogger(Paths.LOGS_DIR, name='vae-training.logs', log_graph=True, version='version-1.0'),
    CSVLogger(Paths.LOGS_DIR, name='vae-training.logs', version='version-1.0')
]
checkpoint_callback = ModelCheckpoint(dirpath=Paths.MODEL_CHECKPOINT_DIR,
                                      filename='vae-wrapper', save_top_k=1,
                                      monitor='val_loss')

###Training the model

trainer = L.Trainer(default_root_dir=Paths.MODEL_CHECKPOINT_DIR, max_epochs=10000, callbacks=[checkpoint_callback],
                    logger=loggers, accelerator=detect_device(), enable_checkpointing=True, log_every_n_steps=10)

trainer.fit(vae_wrapper, datamodule=datamodule)