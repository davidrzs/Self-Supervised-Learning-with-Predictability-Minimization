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

import copy
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from solo.losses.barlow import barlow_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select, remove_bias_and_norm_from_weight_decay
import torch.distributed as dist
from solo.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from pytorch_lightning.profilers import SimpleProfiler, PassThroughProfiler
from pytorch_lightning.core.module import MODULE_OPTIMIZERS
from pytorch_lightning.utilities.types import LRSchedulerPLType

def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs

class CLNonLinPredMinv6(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig, profiler: SimpleProfiler = None):
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
        self.clip_pred_loss: float = cfg.method_kwargs.clip_pred_loss
        self.mask_fraction : float = cfg.method_kwargs.mask_fraction
        self.embed_train = None
        self.proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        self.proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.profiler = profiler or PassThroughProfiler()
        self.pred_loss_transform = cfg.method_kwargs.pred_loss_transform
        self.pred_eval_steps = cfg.method_kwargs.pred_eval_steps
        self.norm_type = cfg.method_kwargs.norm_type
        # projector
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
        # TODO: Maybe improve opt
        self.pred_train_embed = None
        self.max_pred_steps = cfg.method_kwargs.max_pred_steps
        self.pred_train_type = cfg.method_kwargs.pred_train_type
        self.pred_clip_grad = cfg.method_kwargs.pred_clip_grad
        self.pred_lr = cfg.method_kwargs.pred_lr_init
        self.pred_weight_decay = cfg.method_kwargs.pred_weight_decay
        self.pred_lr_update = 1.005
        self.patience = cfg.method_kwargs.patience
        self.pred_steps_target = cfg.method_kwargs.pred_steps_target
        if cfg.method_kwargs.pred_type == "mlp":
            self.k = cfg.method_kwargs.pred_kwargs.k
            self.group_size = self.proj_output_dim // self.k
            assert self.proj_output_dim % self.k == 0
            self.predictors = nn.ModuleList(MLPPredictor(feature_dim = self.proj_output_dim, **cfg.method_kwargs.pred_kwargs) for _ in range(cfg.method_kwargs.pred_kwargs.k))
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
        cfg = super(CLNonLinPredMinv6, CLNonLinPredMinv6).add_and_assert_specific_cfg(cfg)
        
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.lamb")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_kwargs")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.norm_type")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_loss_transform")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_lamb")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.clip_pred_loss")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_lr")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.max_pred_steps")

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
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        params = super().learnable_params + extra_learnable_params
        return params

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
 
    def configure_optimizers(self) -> Tuple[List, List]:
        self.predictors.to(self.device)
        self.opts_pred = [torch.optim.SGD(self.predictors[i].parameters(), lr = self.pred_lr, weight_decay=self.pred_weight_decay) for i in range(self.k)]
        return super().configure_optimizers()
        # pred_opt, pred_sched = self.configure_optimizers_base([{"name": "predictor", "params": self.predictor.parameters()}])
       
    def optimize_predictor(self, embeddings_train: torch.Tensor, embeddings_eval: torch.Tensor, 
                           optimizer: MODULE_OPTIMIZERS, predictor: nn.Module, idx: int):
        """Optimizes the predictor.

        Args:
            embeddings_train (torch.Tensor): embeddings to train the predictor on. (Ignored if pred_train_type == 'same')
            embeddings_eval (torch.Tensor): embeddings to evaluate the predictor on.
            optimizer (MODULE_OPTIMIZERS): optimizer to use for the predictor.
            predictor (nn.Module): predictor to optimize.
            idx (int): index of the predictor to optimize.
        Returns:
            None
        """
        eval_X = torch.cat((embeddings_eval[:,:idx*self.group_size],embeddings_eval[:,(idx+1)*self.group_size:]), dim=1)
        eval_y = embeddings_eval[:,idx*self.group_size:(idx+1)*self.group_size]
        train_X = torch.cat((embeddings_train[:,:idx*self.group_size],embeddings_train[:,(idx+1)*self.group_size:]), dim=1)
        train_y = embeddings_train[:,idx*self.group_size:(idx+1)*self.group_size]
        #Do a first eval step
        with torch.no_grad():
            predictor.eval()
            pred = predictor(eval_X)
            loss_eval_initial = (pred - eval_y).pow(2).mean()
                
        #Train the predictor
        count = 0
        last_improved = 0
        loss_eval_old = loss_eval_initial

        while count < self.max_pred_steps and last_improved <= self.patience:
            count += 1
            predictor.train()
            prediction_train = predictor(train_X)
            predictor_loss =  (prediction_train - train_y).pow(2).mean()
            
            #Optimize
            optimizer.zero_grad()
            predictor_loss.backward()
            if self.pred_clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), self.pred_clip_grad)
            # self.clip_gradients(optimizer, 0.5, gradient_clip_algorithm="norm")
            optimizer.step()
            
            #TODO: Optimize in case of same data
            predictor.eval()
            prediction_eval_new = predictor(eval_X)
            loss_eval_new = (prediction_eval_new - eval_y).pow(2).mean()
            
            # Update values to track patience
            if loss_eval_new > loss_eval_old:
                last_improved += 1
            else:
                loss_eval_old = loss_eval_new 
                last_improved = 0

        #  Update learning rate of predictor optimizer
        pred_lr = optimizer.param_groups[0]['lr']
        if self.pred_steps_target > 0:
            if count > self.pred_steps_target:
                pred_lr = pred_lr * self.pred_lr_update
            else:
                pred_lr = pred_lr / self.pred_lr_update
        
        pred_lr = min(max(pred_lr, 1e-5), 1e-1)
        optimizer.param_groups[0]['lr'] = pred_lr
        
        # return dict with all metrics
        return {"eval_old": loss_eval_initial.detach(),
                "eval_new": loss_eval_new.detach(), 
                "eval_diff": (loss_eval_initial - loss_eval_new).detach(), 
                "cl_predictor_loss": predictor_loss.detach(), 
                "opt_pred_steps": count,
                "pred_lr": pred_lr}

    def optimize_encoder(self, embeddings_eval: torch.Tensor, z1: torch.Tensor, z2: torch.Tensor):
        """Optimizes the encoder.
        
        Args:
            embeddings_eval (torch.Tensor): embeddings to evaluate the predictor on.
            z1 (torch.Tensor): embeddings from transformation 1
            z2 (torch.Tensor): embeddings from transformation 2
        Returns:
            loss_encoder (torch.Tensor): total loss composed of Barlow loss and classification loss.
        """

        #Calculate loss
        predictability_loss_raw = torch.tensor(0.0, device=embeddings_eval.device, requires_grad=True, dtype=embeddings_eval.dtype)
        for idx in range(self.k):
            predictor = self.predictors[idx]
            predictor.eval()
            prediction_eval = predictor(torch.cat((embeddings_eval[:,:idx*self.group_size],embeddings_eval[:,(idx+1)*self.group_size:]), dim=1))
            predictability_loss_raw_step = (prediction_eval - embeddings_eval[:,idx*self.group_size:(idx+1)*self.group_size]).pow(2).mean()
            predictability_loss_raw = predictability_loss_raw + predictability_loss_raw_step / self.k

        if self.clip_pred_loss>0:
            predictability_loss_raw = torch.clamp(predictability_loss_raw, max=self.clip_pred_loss)

        if self.pred_loss_transform == "log":   
            predictability_loss = torch.log(predictability_loss_raw)
        elif self.pred_loss_transform == "sqrt":
            predictability_loss = torch.sqrt(predictability_loss_raw)
        elif self.pred_loss_transform == "identity":
            predictability_loss = predictability_loss_raw
        else:
            raise ValueError(f"Transform {self.pred_loss_transform} not implemented")
        #Scale
        loss_encoder_pred =  -self.pred_lamb * predictability_loss

        N, D = z1.shape
        corr = torch.einsum("bi, bj -> ij", z1, z2) / N
        on_diag = torch.diagonal(corr).add(-1).pow(2).sum()
        loss_encoder = on_diag + loss_encoder_pred


        #Log
        # self.log("cl_predictor_loss", predictor_loss, on_epoch=True, sync_dist=True)
        self.log("cl_predictability_loss", predictability_loss_raw, on_epoch=True, on_step=False)
        self.log("cl_scaled_transformed_predictability_loss", loss_encoder_pred, on_epoch=True, on_step=False)
        self.log("cl_on_diag_loss", on_diag, on_epoch=True, on_step=False)
        self.log("train_cl_pred_min_total_loss", loss_encoder, on_epoch=True, on_step=False)
        self.log("cl_pred_over_diag", abs(loss_encoder_pred.detach()/on_diag.detach()), on_epoch=True, on_step=False)

        return loss_encoder

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Barlow Twins reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Barlow loss and classification loss.
        """
        with self.profiler.profile("forward_backbone1"):
            #===== Encoder forward pass =====
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

            embeddings_eval = torch.cat((z1_norm, z2_norm), dim=0)

        with self.profiler.profile("train_predictor"):
            metric_summary = {"eval_old":0,
                    "eval_new":0,
                    "eval_diff":0,
                    "cl_predictor_loss":0,
                    "opt_pred_steps":0,
                    "pred_lr":0}
            detached_embeddings_eval = embeddings_eval.detach()
            if self.pred_train_type == 'split':
                pred_embed_train = detached_embeddings_eval[:detached_embeddings_eval.shape[0]//2]
                pred_embed_eval = detached_embeddings_eval[detached_embeddings_eval.shape[0]//2:] 
            elif self.pred_train_type == 'overfit':
                pred_embed_train = detached_embeddings_eval
                pred_embed_eval = detached_embeddings_eval
            else:
                raise ValueError(f"Predictor train type {self.pred_train_type} not implemented")
            #Train all k predictors separately
            for idx in range(self.k):
                metrics = self.optimize_predictor(pred_embed_train, pred_embed_eval, self.opts_pred[idx], self.predictors[idx], idx)
                for key, value in metrics.items():
                    #Aggregate metrics
                    metric_summary[key] += value
            #Log values
            for key, value in metric_summary.items():
                self.log(f"{key}", value/self.k, on_epoch=True, on_step=False, sync_dist=True)

        with self.profiler.profile("forward_backbone2"):
            loss_encoder = self.optimize_encoder(embeddings_eval, z1_norm, z2_norm)
        return loss_encoder + out["loss"]

        
def average_predictor_mse_loss(
    predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor
):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)

    only_prediction = torch.masked_select(diff_square, index_mask)
    prediction_error = torch.mean(only_prediction)

    return prediction_error


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
        # print("Input:",x.requires_grad, x)
        for layer in self.pred_layers:
            x = layer(x)
            # print("layer:",layer.weight)
            # print("output:",x.requires_grad, x)
            
        return x


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
