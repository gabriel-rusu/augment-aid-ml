import torch
import torchvision

from src.utils.helpers import matplotlib_imshow


def combine_sample(real, fake, p_real):
    '''
    Function to take a set of real and fake images of the same length (x)
    and produce a combined tensor with length (x) and sampled at the target probability
    Parameters:
        real: a tensor of real images, length (x)
        fake: a tensor of fake images, length (x)
        p_real: the probability the images are sampled from the real set
    '''
    target_images = real.clone()
    fake_idx = torch.rand(fake.size()[0]) >= p_real
    target_images[fake_idx] = fake[fake_idx]
    return target_images


def show_train_images_sample(datamodule, col_image_number=4):
    dataiter = iter(datamodule.train_dataloader())
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(datamodule.dataset_classes()[labels[j]] for j in range(col_image_number)))
