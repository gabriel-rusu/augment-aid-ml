import lightning
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms.v2 import Compose, RandomVerticalFlip, RandomHorizontalFlip, ToTensor, Normalize, ToImage, \
    ToDtype


class FashionMNISTDataModule(lightning.LightningDataModule):

    def __init__(self, dataset_dir, batch_size):
        super().__init__()
        self._classes = FashionMNIST.classes
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.mnist_train = FashionMNIST(self.dataset_dir, train=True, transform=self.data_transforms(True),
                                        download=True)
        self.mnist_validation = FashionMNIST(self.dataset_dir, train=False, transform=self.data_transforms(False),
                                                       download=True)

    def data_transforms(self, train=False):
        transform = [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize((0.5,), (0.5,))
        ]

        if train:
            transform = [
                RandomVerticalFlip(),
                RandomHorizontalFlip()
            ] + transform

        return Compose(transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=4,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.mnist_validation, batch_size=self.batch_size, num_workers=4,
                          persistent_workers=True)

    def dataset_classes(self):
        return self._classes

    def num_classes(self):
        return len(self._classes)