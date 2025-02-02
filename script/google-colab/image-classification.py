
###Imports

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.dataset.FashionMNISTDataModule import FashionMNISTDataModule
from src.models.classifier import ResNet18Classifier
from src.train.wrapper.classifier_wrapper import ClassifierWrapper
from src.utils.constants import Paths
from src.utils.helpers import detect_device

###Defining batch size

BATCH_SIZE = 512

###Create FashionMNISTDataModule

datamodule = FashionMNISTDataModule(Paths.DATA_DIR, BATCH_SIZE)
datamodule.setup('fit')

print('Training set has {} instances'.format(len(datamodule.train_dataloader()) * BATCH_SIZE))
print('Validation set has {} instances'.format(len(datamodule.val_dataloader()) * BATCH_SIZE))

###Assemble the model

model = ResNet18Classifier(datamodule.num_classes())
classifier_wrapper = ClassifierWrapper(model, datamodule.num_classes())

###Adding logging and checkpointing

loggers = [
    TensorBoardLogger(Paths.LOGS_DIR, name='classifier-training.logs', log_graph=True, version='version-1.0'),
    CSVLogger(Paths.LOGS_DIR, name='classifier-training.logs', version='version-1.0')
]
checkpoint_callback = ModelCheckpoint(dirpath=Paths.MODEL_CHECKPOINT_DIR,
                                      filename='classifier-{epoch:02d}-{val_loss:.2f}', save_top_k=3,
                                      monitor='val_loss')

###Training the model

trainer = L.Trainer(default_root_dir=Paths.MODEL_CHECKPOINT_DIR, max_epochs=50, callbacks=[checkpoint_callback],
                    logger=loggers, accelerator=detect_device(), enable_checkpointing=True, log_every_n_steps=50)

trainer.fit(classifier_wrapper, datamodule=datamodule)