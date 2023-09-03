
import math
import os
import pathlib
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
        device: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        path: str,
):

    data = []
    Y = []

    path = pathlib.Path(path)
    path.mkdir(exist_ok=True,parents=True)


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

    data = torch.cat(data, dim=0).numpy()

    data_standardized = (data - data.mean(0)) / data.std(0)

    Y = torch.cat(Y, dim=0)
    Y = Y.numpy()
    corr = (np.transpose(data_standardized) @ data_standardized) / data_standardized.shape[0]

    np.savetxt(path / Path('labels.csv'), Y, delimiter=",")
    np.savetxt(path / Path('data_standardized.csv'), data_standardized, delimiter=",")
    np.savetxt(path / Path('correlation.csv'), corr, delimiter=",")



