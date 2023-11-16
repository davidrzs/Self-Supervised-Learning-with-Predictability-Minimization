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


class CLNonLinPredMinv3(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements Barlow Twins (https://arxiv.org/abs/2103.03230)

        Extra cfg settings:
            method_kwargs:
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                proj_output_dim (int): number of dimensions of projected features.
                mask_fraction (float): fraction of the embeddings to mask
                lamb (float): off-diagonal scaling factor for the cross-covariance matrix.
                proj_size (int): Layers in projector
                pred_type (str): Type of predictor, currently supported: "mlp"
                norm_type (str): Type of normalization, currently supported: "l2", "standardize", "both"
                pred_loss_transform (str): Type of transform for the predictor loss, currently supported: "log", "sqrt", "identity"
                pred_lamb (float): Scaling factor for the predictor loss
                predictor_kwargs (dict): kwargs for predictor
"""

        super().__init__(cfg)

        self.lamb: float = cfg.method_kwargs.lamb
        self.pred_lamb: float = cfg.method_kwargs.pred_lamb
        self.mask_fraction : float = cfg.method_kwargs.mask_fraction

        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim

        self.pred_loss_transform = cfg.method_kwargs.pred_loss_transform
        self.norm_type = cfg.method_kwargs.norm_type
        # projector
        self.pred_train_embed = None
        proj_layers = nn.ModuleList()
        cur_in_dim = self.features_dim

        #TODO: Support 0 layers
        for _ in range(cfg.method_kwargs.proj_size - 1):
            proj_layers.append(nn.Linear(cur_in_dim, self.proj_hidden_dim),)
            proj_layers.append(nn.BatchNorm1d(self.proj_hidden_dim))
            proj_layers.append(nn.ReLU())
            cur_in_dim = self.proj_hidden_dim

        proj_layers.append(nn.Linear(cur_in_dim, self.proj_output_dim))

        self.projector = nn.Sequential(*proj_layers)

        # predictor
        # TODO: Add kwargs to predictor and oprimizer, Maybe improve opt
        if cfg.method_kwargs.pred_type == "mlp":
            self.predictor = MLPPredictor(feature_dim = self.proj_output_dim, **cfg.method_kwargs.pred_kwargs)
        else:
            raise ValueError(f"Predictor {cfg.method_kwargs.pred_type} not implemented")

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        #TODO: Add warnings for missing parameters
        cfg = super(CLNonLinPredMinv3, CLNonLinPredMinv3).add_and_assert_specific_cfg(cfg)
        
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.lamb")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_kwargs")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.norm_type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_loss_transform")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_lamb")

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.mask_fraction")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_size")

        assert cfg.method_kwargs.pred_type in ["mlp"]
        assert cfg.method_kwargs.norm_type in ["l2", "standardize", "both"]
        assert cfg.method_kwargs.pred_loss_transform in ["log", "sqrt", "identity"]

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        #TODO: possibly add custom params for predictor
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()},
                                  {"name": "predictor", "params": self.predictor.parameters()}]
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
        z1, z2 = out["z"]

        #Normalize each feature vector
        if self.norm_type == "l2":
            z1_norm = F.normalize(z1, dim=-1)
            z2_norm = F.normalize(z2, dim=-1)
        elif self.norm_type == "standardize":
            norm_layer = nn.BatchNorm1d(self.proj_output_dim, affine=False, device=z1.device)
            z1_norm = norm_layer(z1)
            z2_norm = norm_layer(z2)
        elif self.norm_type == "both":
            z1_norm_ = F.normalize(z1, dim=-1)
            z2_norm_ = F.normalize(z2, dim=-1)
            z1_norm = (z1_norm_ - z1_norm_.mean(dim=0)) / z1_norm_.std(dim=0)
            z2_norm = (z2_norm_ - z2_norm_.mean(dim=0)) / z2_norm_.std(dim=0)
        else:
            raise ValueError(f"Norm type {self.norm_type} not implemented")
        N, D = z1.shape
        corr = torch.einsum("bi, bj -> ij", z1, z2) / N
        on_diag = torch.diagonal(corr).add(-1).pow(2).sum()

        embeddings_eval = torch.cat((z1_norm, z2_norm), dim=0)
        N, D = z1_norm.shape
        corr = torch.einsum("bi, bj -> ij", z1_norm, z2_norm) / N
        on_diag = torch.diagonal(corr).add(-1).pow(2).sum()

        mask_eval, eval_input =   to_dataset(embeddings_eval, self.mask_fraction)
        prediction_eval = self.predictor(eval_input)
        predictability_loss_raw = average_predictor_mse_loss(prediction_eval, embeddings_eval, mask_eval).mean()
        predictability_loss = predictability_loss_raw * self.proj_output_dim

        if self.pred_loss_transform == "log":   
            predictability_loss = torch.log(predictability_loss_raw)
        elif self.pred_loss_transform == "sqrt":
            predictability_loss = torch.sqrt(predictability_loss_raw)
        elif self.pred_loss_transform == "identity":
            predictability_loss = predictability_loss_raw
        else:
            raise ValueError(f"Transform {self.pred_loss_transform} not implemented")
        
        embeddings_train = self.pred_train_embed
        if embeddings_train is not None:
            mask_train, train_input = to_dataset(embeddings_train, self.mask_fraction)
            prediction_train = self.predictor(train_input)
            predictor_loss = average_predictor_mse_loss(prediction_train, embeddings_train, mask_train).mean()
            predictor_loss = predictor_loss * self.proj_output_dim
            if self.pred_loss_transform == "log":   
                predictor_loss = torch.log(predictor_loss)
            elif self.pred_loss_transform == "sqrt":
                predictor_loss = torch.sqrt(predictor_loss)
            elif self.pred_loss_transform == "identity":
                predictor_loss = predictor_loss
            else:
                raise ValueError(f"Transform {self.pred_loss_transform} not implemented")
        else:
            predictor_loss = torch.tensor(0, device=predictability_loss.device)

        self.pred_train_embed = embeddings_eval.detach()
        

        
        loss_encoder = on_diag - self.lamb * predictability_loss
        loss_predictor = self.pred_lamb * predictor_loss
        loss = loss_encoder + loss_predictor

        self.log("cl_scaled_predictor_loss", loss_predictor, on_epoch=True, sync_dist=True)
        self.log("cl_scaled_predictability_loss", predictability_loss, on_epoch=True, sync_dist=True)
        self.log("cl_predictor_loss", predictor_loss, on_epoch=True, sync_dist=True)
        self.log("cl_predictability_loss", predictability_loss_raw, on_epoch=True, sync_dist=True)
        self.log("cl_on_diag_loss", on_diag, on_epoch=True, sync_dist=True)
        self.log("train_cl_pred_min_total_loss", loss, on_epoch=True, sync_dist=True)
        # #This is just here for backwards compatibility
        # self.log("eval_cl_pred_min_predictor_loss_log", torch.log(predictability_loss_raw), on_epoch=True, sync_dist=True)
        # self.log("eval_cl_pred_min_predictor_loss_mse   ", predictability_loss_raw, on_epoch=True, sync_dist=True)
        
        return loss + out["loss"]




def average_predictor_mse_loss(
    predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor
):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)

    only_prediction = torch.masked_select(diff_square, index_mask)
    prediction_error = torch.mean(only_prediction)

    return prediction_error



# def individual_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
#     assert predictions.shape == labels.shape

#     diff_square = torch.square(predictions - labels)
#     reconstruction = index_mask * diff_square

#     # case that we divide by zero extremely unlikely
#     prediction_error = reconstruction.sum(dim=0) / reconstruction.count_nonzero(dim=0)

#     return prediction_error

class MLPPredictor(nn.Module):
    def __init__(self, feature_dim , hidden_dim=512, layers=3, activation="relu"):
        super().__init__()
        self.pred_layers = nn.ModuleList()
        cur_in_dim = feature_dim*2
        for _ in range(layers - 1):
            self.pred_layers.append(nn.Linear(cur_in_dim, hidden_dim))
            self.pred_layers.append(nn.BatchNorm1d(hidden_dim))
            if activation == "relu":
                self.pred_layers.append(nn.ReLU())
            elif activation == "tanh":
                self.pred_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Activation {activation} not implemented")

            cur_in_dim = hidden_dim

        self.pred_layers.append(nn.Linear(cur_in_dim, feature_dim))

    def forward(self, x):
        for layer in self.pred_layers:
            x = layer(x)
        return x



# def validate_predictor(predictability_net, args, input, true_output, masked_indices):
#     # predictability_net.eval()
#     val_outputs = predictability_net(input.cuda(non_blocking=True))
#     prediction_loss = average_predictor_mse_loss(val_outputs.cuda(non_blocking=True), true_output.cuda(non_blocking=True), masked_indices.cuda(non_blocking=True)).mean()
#     # del val_outputs
#     return prediction_loss



# class WidePredictor(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.l1 = nn.Linear(feature_dim*2, feature_dim*2*10)
#         self.bn1 = nn.BatchNorm1d(feature_dim*2*10)
#         self.r1 = nn.ReLU()
#         self.l2 = nn.Linear(feature_dim*2*10, feature_dim)

#     def forward(self, x):
#         w1 = self.bn1(self.r1(self.l1(x)))
#         return self.l2(w1)


def to_dataset(embeddings, masking_fraction):
    batch_size, embedding_dimension = embeddings.shape
    ret_arr = torch.zeros(batch_size, 2*embedding_dimension,device=embeddings.device)
    masked_indices = torch.rand(batch_size, embedding_dimension,device=embeddings.device) < (masking_fraction)

    # copy over first half of input
    ret_arr[:, :embedding_dimension] = embeddings
    # mask the inputs we are predicting
    ret_arr[:, :embedding_dimension].masked_fill_(masked_indices, 0)
    # tell the neural network which ones we are predicting
    ret_arr[:, embedding_dimension:].masked_fill_(masked_indices, 1)

    return masked_indices, ret_arr


# def individual_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
#     assert predictions.shape == labels.shape

#     diff_square = torch.square(predictions - labels)
#     reconstruction = index_mask * diff_square

#     # case that we divide by zero extremely unlikely
#     prediction_error = reconstruction.sum(dim=0) / reconstruction.count_nonzero(dim=0)

#     return prediction_error


# def average_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
#     assert predictions.shape == labels.shape

#     diff_square = torch.square(predictions - labels)

#     only_prediction = torch.masked_select(diff_square, index_mask)
#     prediction_error = torch.mean(only_prediction)

#     return prediction_error


# def predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor, gamma: float,
#                        theta: float):
#     """
#     :param logger: if we pass a logger we will log the losses
#     :param predictions: the LOO predictions of our network
#     :param labels: the original embedding
#     :param index_mask: the indices we are masking
#     :param gamma: weight weighting error of masked error in sum with non-masked reconstruction error
#     :return: the loss as a tensor
#     """
#     # print(f' predictions.shape {predictions.shape}, labels.shape {labels.shape}')
#     assert predictions.shape == labels.shape
#     # print(f'{index_mask}')
#     # number_of_batches = predictions.shape[0]
#     # number_of_elements_in_batch = predictions.shape[1]

#     # get the squared difference of our neural networks output and the original labels
#     diff_square = torch.square(predictions - labels)
#     # print(diff_square.mean())
#     # now we must calculate the two parts - the reconstruction error and the prediction error
#     only_reconstruction = torch.masked_select(diff_square, torch.logical_not(index_mask))

#     # recall that mask returns a one dimensional flattened tensor
#     reconstruction_error = torch.mean(only_reconstruction)

#     only_prediction = torch.masked_select(diff_square,index_mask)
#     prediction_error = torch.mean(only_prediction)

#     # in the unlikely case that all were selected or non were selected
#     # TODO assumes we are running on a nvidia card
#     if prediction_error.isnan():
#         prediction_error = torch.tensor(0).cuda()

#     if reconstruction_error.isnan():
#         reconstruction_error = torch.tensor(0).cuda()

#     total_loss = theta*prediction_error + gamma*reconstruction_error

#     del prediction_error, reconstruction_error, only_prediction, only_reconstruction

#     return total_loss

