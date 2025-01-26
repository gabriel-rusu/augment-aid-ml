import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def detect_device(debug=False):
    if debug:
        return "cpu"
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def matplotlib_imshow(img, one_channel=False):
    print(img.shape)
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def determine_state_dict(dir_path: str, model_name: str):
    latest_weights = None
    latest_weights_date = None
    for checkpoint in filter(lambda dir: model_name in dir, os.listdir(dir_path)):
        checkpoint_path = os.path.join(dir_path, checkpoint)
        checkpoint_date = os.path.getmtime(checkpoint_path)
        if latest_weights is None:
            latest_weights = checkpoint
            latest_weights_date = checkpoint_date
        elif checkpoint_date > latest_weights_date:
            latest_weights = checkpoint
            latest_weights_date = checkpoint_date

    return torch.load(os.path.join(dir_path, latest_weights))['state_dict']