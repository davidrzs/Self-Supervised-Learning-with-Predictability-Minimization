
import math
import os
import random
import string
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import umap
import wandb
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from solo.utils.misc import gather, omegaconf_select
from tqdm import tqdm


def calculate_correlation(
        self,
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        csv_path: str,
):
    """Produces a UMAP visualization by forwarding all data of the
    first validation dataloader through the model.
    **Note: the model should produce features for the forward() function.

    Args:
        device (str): gpu/cpu device.
        model (nn.Module): current model.
        dataloader (torch.utils.data.Dataloader): current dataloader containing data.
        csv_path (str): path to save the csv.
    """

    data = []
    Y = []

    # set module to eval model and collect all feature representations
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Collecting features"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feats = model(x)
            data.append(feats.cpu())
            Y.append(y.cpu())
    model.train()

    nr_of_elements = len(Y)

    data = torch.cat(data, dim=0).numpy()
    Y = torch.cat(Y, dim=0)
    num_classes = len(torch.unique(Y))
    Y = Y.numpy()

    corr = (np.transpose(data) @ data) / nr_of_elements

    np.savetxt(csv_path, corr, delimiter=",")



