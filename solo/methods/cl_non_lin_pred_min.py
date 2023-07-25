# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
import torch.distributed as dist


class CLNonLinPredMin(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
        """

        super().__init__(cfg)

        self.lamb: float = cfg.method_kwargs.lamb
        self.mask_fraction : float = cfg.method_kwargs.mask_fraction
        self.ridge_lambd : float = cfg.method_kwargs.ridge_lambd

        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim


        # projector

        if cfg.method_kwargs.proj_size == 1:
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, self.proj_output_dim),
            )
        if cfg.method_kwargs.proj_size == 2:
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            )
        if cfg.method_kwargs.proj_size == 3:
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_hidden_dim),
                nn.BatchNorm1d(self.proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.proj_hidden_dim, self.proj_output_dim),
            )

        # predictor
        self.predictor = Predictor(self.proj_output_dim)
        self.predictor_optimizer = torch.optim.AdamW(self.predictor.parameters(), weight_decay=1e-4)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CLNonLinPredMin, CLNonLinPredMin).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_size")


        cfg.method_kwargs.lamb = omegaconf_select(cfg, "method_kwargs.lamb", 0.1)

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X):
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])

        out.update({"z": z})
        return out



    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)


        batch_size, _ = z1_norm.size()

        out_1_norm = (z1_norm - z1_norm.mean(dim=0)) / z1_norm.std(dim=0)
        out_2_norm = (z2_norm - z2_norm.mean(dim=0)) / z2_norm.std(dim=0)

        # embeddings = torch.cat((out_1_norm, out_2_norm), 0)

        # here we split the embeddings into train and test (first half of out_1_norm & out_2_norm is train, second half is test)
        embeddings_train = torch.concat((out_1_norm[:batch_size//2], out_2_norm[:batch_size//2]), 0).clone().detach()
        embeddings_eval = torch.concat((out_1_norm[batch_size//2:], out_2_norm[batch_size//2:]), 0)
        


        number_to_mask = int(self.proj_output_dim * self.mask_fraction)

        batch_size, embedding_dimension = out_1_norm.shape

        masked_indices_eval = (torch.rand(batch_size, embedding_dimension, device=embeddings_eval.device) < (
                    number_to_mask / embedding_dimension))

        masked_embeddings_eval = (~masked_indices_eval) * embeddings_eval

        
        
        # get a first guess at how good the predictor is
        self.predictor.eval() 
        predictions = self.predictor(masked_embeddings_eval)
        self.predictor.train()
        prediction_loss = average_predictor_mse_loss(predictions, embeddings_eval, masked_indices_eval)


        
        # here we train the predictor while the validation loss is still going down 
        self.log("train_cl_pred_first_prediction_loss", prediction_loss, on_epoch=True, sync_dist=True)

        
        # this emulates a do-while loop
        counter = 0
        while True:
            counter += 1
            
            masked_indices_train = (torch.rand(batch_size, embedding_dimension, device=embeddings_train.device) < (
                    number_to_mask / embedding_dimension))
            masked_embeddings_train = (~masked_indices_train) * embeddings_train

            # train for one round:
            self.predictor_optimizer.zero_grad()
            predictions = self.predictor(masked_embeddings_train)
            prediction_loss_train = average_predictor_mse_loss(predictions, embeddings_train, masked_indices_train)
            prediction_loss_train.backward()
            self.predictor_optimizer.step()
            self.predictor_optimizer.zero_grad()
            self.predictor.eval() 
            predictions = self.predictor(masked_embeddings_eval)
            prediction_loss_new = average_predictor_mse_loss(predictions, embeddings_eval, masked_indices_eval)
            self.predictor.train()
            if (prediction_loss_new >= prediction_loss).item() or counter > 50:
                prediction_loss = prediction_loss_new
                break
            prediction_loss = prediction_loss_new

            
        self.log("train_cl_pred_last_prediction_loss", prediction_loss, on_epoch=True, sync_dist=True)
        
        self.log("train_cl_pred_last_counter", counter, on_epoch=True, sync_dist=True)

        self.log("eval_cl_pred_min_predictor_loss_log", torch.log(prediction_loss), on_epoch=True, sync_dist=True)


        del masked_indices_train, masked_indices_eval, masked_embeddings_train, masked_embeddings_eval

        last_prediction_loss = torch.log(prediction_loss)

        # cross-correlation matrix
        c = (out_1_norm.T @ out_2_norm) / self.batch_size

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c)
            world_size = dist.get_world_size()
            c /= world_size

        on_diag = torch.diagonal(c).add(-1).pow(2).sum()

        # note the minus as we try to maximize the prediction loss
        loss = on_diag - self.lamb * (self.proj_output_dim * last_prediction_loss)

        self.log("train_cl_pred_min_on_diag_loss", on_diag, on_epoch=True, sync_dist=True)
        self.log("train_cl_pred_min_total_loss", loss, on_epoch=True, sync_dist=True)

        return loss + class_loss




def average_predictor_mse_loss(
    predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor
):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)

    only_prediction = torch.masked_select(diff_square, index_mask)
    prediction_error = torch.mean(only_prediction)

    return prediction_error



def individual_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)
    reconstruction = index_mask * diff_square

    # case that we divide by zero extremely unlikely
    prediction_error = reconstruction.sum(dim=0) / reconstruction.count_nonzero(dim=0)

    return prediction_error


class Predictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # self.l1 = nn.Linear(feature_dim, feature_dim)
        # self.bn1 = nn.BatchNorm1d(feature_dim)
        # self.r1 = nn.LeakyReLU()
        # self.l2 = nn.Linear(feature_dim, feature_dim)
        # self.bn2 = nn.BatchNorm1d(feature_dim)
        # self.r2 = nn.LeakyReLU()
        self.l3 = nn.Linear(feature_dim, feature_dim)


    def forward(self, x):
        # w1 = self.bn1(self.r1(self.l1(x)))
        # w2 = self.bn2(self.r2(self.l2(w1)))
        return self.l3(x)


