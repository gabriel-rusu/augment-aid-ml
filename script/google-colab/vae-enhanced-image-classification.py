
###Imports
import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import TensorBoardLogger

from src.dataset.FashionMNISTDataModule import FashionMNISTDataModule
from src.models.classifier import ResNet18Classifier
from src.models.vae import VAE
from src.train.wrapper.vae_augmented_training_wrapper import VAEAugmentedTrainingWrapper
from src.train.wrapper.vae_wrapper import VAEWrapper
from src.utils.constants import Paths
from src.utils.helpers import detect_device

###Defining batch size

BATCH_SIZE = 1024

###Create FashionMNISTDataModule

datamodule = FashionMNISTDataModule(Paths.DATA_DIR, BATCH_SIZE)
datamodule.setup('fit')

print('Training set has {} instances'.format(len(datamodule.train_dataloader()) * BATCH_SIZE))
print('Validation set has {} instances'.format(len(datamodule.val_dataloader()) * BATCH_SIZE))

###Defining VAE properties

input_dim = 28 * 28
hidden_dim = 400
latent_dim = 200

###Assembling the model

model = ResNet18Classifier(datamodule.num_classes())
vae = VAE(input_dim, hidden_dim, latent_dim, detect_device()).to(detect_device())

vae_wrapper = VAEWrapper(vae, display_every_n_steps=100)
vae_wrapper.load_state_dict(torch.load(Paths.VAE_WRAPPER_CHECKPOINT_FILE_PATH)['state_dict'])

classifier_wrapper = VAEAugmentedTrainingWrapper(model, vae_wrapper.vae,
                                                 datamodule.train_dataloader_unaltered())

###Adding logging and checkpointing

loggers = [
    TensorBoardLogger(Paths.LOGS_DIR, name='classifier-vae-training.logs', log_graph=True, version='version-1.0'),
    CSVLogger(Paths.LOGS_DIR, name='classifier-vae-training.logs', version='version-1.0')
]

logger = TensorBoardLogger(Paths.LOGS_DIR, name='classifier-vae-training.logs')
checkpoint_callback = ModelCheckpoint(dirpath=Paths.MODEL_CHECKPOINT_DIR,
                                      filename='classifier-vae-{epoch:02d}-{val_loss:.2f}', save_top_k=3,
                                      monitor='val_loss')

###Training the enhanced classifier

trainer = L.Trainer(default_root_dir=Paths.MODEL_CHECKPOINT_DIR, max_epochs=50, callbacks=[checkpoint_callback],
                    logger=logger, accelerator=detect_device(), enable_checkpointing=True, log_every_n_steps=50)

trainer.fit(classifier_wrapper, datamodule=datamodule)