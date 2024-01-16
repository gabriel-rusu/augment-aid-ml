import torch


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