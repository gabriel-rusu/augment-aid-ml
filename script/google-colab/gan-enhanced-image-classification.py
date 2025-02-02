###Imports

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.dataset.FashionMNISTDataModule import FashionMNISTDataModule
from src.models.classifier import ResNet18Classifier
from src.models.discriminator import Discriminator
from src.models.generator import Generator
from src.train.wrapper.gan_augmented_training_wrapper import GANAugmentedTrainingWrapper
from src.train.wrapper.gan_wrapper import GANWrapper
from src.utils.constants import Paths
from src.utils.helpers import detect_device

###Defining batch size

BATCH_SIZE = 512

###Creating the FashionMNISTDataModule

datamodule = FashionMNISTDataModule(Paths.DATA_DIR, BATCH_SIZE)
datamodule.setup('fit')


print('Training set has {} instances'.format(len(datamodule.train_dataloader()) * BATCH_SIZE))
print('Validation set has {} instances'.format(len(datamodule.val_dataloader()) * BATCH_SIZE))

###Defining the GAN parameters

z_dim = 128
generator_input_dim = z_dim + datamodule.num_classes()
input_channels = 1
discriminator_input_dim = input_channels + datamodule.num_classes()

###Assemble the model

model = ResNet18Classifier(datamodule.num_classes())
generator = Generator(generator_input_dim, input_channels)
discriminator = Discriminator(discriminator_input_dim)

gan_wrapper = GANWrapper(generator, discriminator, z_dim, datamodule.num_classes(), 10)
gan_wrapper.load_state_dict(torch.load(Paths.GAN_WRAPPER_CHECKPOINT_FILE_PATH)['state_dict'])

classifier_wrapper = GANAugmentedTrainingWrapper(model, gan_wrapper.generator, z_dim,
                                                 datamodule.num_classes())

###Adding logging and checkpointing

loggers = [
    TensorBoardLogger(Paths.LOGS_DIR, name='classifier-gan-training.logs', log_graph=True, version='version-1.0'),
    CSVLogger(Paths.LOGS_DIR, name='classifier-gan-training.logs', version='version-1.0')
]
checkpoint_callback = ModelCheckpoint(dirpath=Paths.MODEL_CHECKPOINT_DIR,
                                      filename='classifier-gan-{epoch:02d}-{val_loss:.2f}', save_top_k=3,
                                      monitor='val_loss')

###Training the model

trainer = L.Trainer(default_root_dir=Paths.MODEL_CHECKPOINT_DIR, max_epochs=50, callbacks=[checkpoint_callback],
                    logger=loggers, accelerator=detect_device(), enable_checkpointing=True, log_every_n_steps=50)

trainer.fit(classifier_wrapper, datamodule=datamodule)