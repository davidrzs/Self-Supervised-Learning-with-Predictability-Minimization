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


class CLNonLinPredMinv2(BaseMethod):
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
        self.proj_train_data = None
        self.proj_train_embed = None
        proj_layers = nn.ModuleList()
        cur_in_dim = self.features_dim

        for _ in range(cfg.method_kwargs.proj_size - 1):
            proj_layers.append(nn.Linear(cur_in_dim, self.proj_hidden_dim),)
            proj_layers.append(nn.BatchNorm1d(self.proj_hidden_dim))
            proj_layers.append(nn.ReLU())
            cur_in_dim = self.proj_hidden_dim

        proj_layers.append(nn.Linear(cur_in_dim, self.proj_output_dim))
        
        self.projector = nn.Sequential(*proj_layers)

        # predictor
        # TODO: Add kwargs to predictor and oprimizer, Maybe improve opt
        self.predictor = Predictor(self.proj_output_dim)
        self.predictor_optimizer = torch.optim.AdamW(self.predictor.parameters(), weight_decay=1e-3)

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(CLNonLinPredMinv2, CLNonLinPredMinv2).add_and_assert_specific_cfg(cfg)

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

        #Normalize each feature vector
        z1_norm = F.normalize(z1, dim=-1)
        z2_norm = F.normalize(z2, dim=-1)


        batch_size, _ = z1_norm.size()
        
        #TODO: Why normalize and standardize?
        #Standardize over each feature across batch
        # out_1_norm = (z1_norm - z1_norm.mean(dim=0)) / z1_norm.std(dim=0)
        # out_2_norm = (z2_norm - z2_norm.mean(dim=0)) / z2_norm.std(dim=0)
        embeddings_eval = torch.cat((z1_norm, z2_norm), dim=0)

        if self.proj_train_data is None:
            number_to_mask = int(self.proj_output_dim * self.mask_fraction)      
            
            detached_embeddings_eval = embeddings_eval.clone().detach()
            mask_eval, eval_input = to_dataset(detached_embeddings_eval, number_to_mask)
            self.proj_train_embed = detached_embeddings_eval
            self.proj_train_data = mask_eval, eval_input

            self.predictor.eval() 
            predictions = self.predictor(eval_input)
            eval_loss = average_predictor_mse_loss(predictions, embeddings_eval, mask_eval)

        else:
            number_to_mask = int(self.proj_output_dim * self.mask_fraction)      

            detached_embeddings_eval = embeddings_eval.clone().detach()
            mask_eval, eval_input = to_dataset(detached_embeddings_eval, number_to_mask)
            mask_train, train_input = self.proj_train_data
            embeddings_train = self.proj_train_embed

            self.proj_train_data = mask_eval.clone().detach(), eval_input.clone().detach()
            self.proj_train_embed = embeddings_eval.clone().detach()
                                                                                
            # get a first guess at how good the predictor is
            self.predictor.eval() 
            predictions = self.predictor(eval_input)
            eval_loss = average_predictor_mse_loss(predictions, embeddings_eval, mask_eval)

            # here we train the predictor while the validation loss is still going down 
            self.log("train_cl_pred_first_prediction_loss", eval_loss, on_epoch=True, sync_dist=True)

            opt_steps = 0
            eval_loss_old = eval_loss + 1
            while eval_loss_old > eval_loss:
                eval_loss_old = eval_loss
                self.predictor.train()
                outputs = self.predictor(train_input)
                #TODO: Why the mean?
                predictability_loss = average_predictor_mse_loss(outputs,embeddings_train,mask_train).mean()
                self.predictor_optimizer.zero_grad()
                predictability_loss.backward()
                self.predictor_optimizer.step()
                opt_steps += 1

                self.predictor.eval()
                eval_outputs = self.predictor(eval_input)
                eval_loss = average_predictor_mse_loss(eval_outputs, embeddings_eval, mask_eval).mean()
        
            
            self.log("train_cl_pred_last_prediction_loss", eval_loss, on_epoch=True, sync_dist=True)
            
            self.log("train_cl_pred_last_counter", opt_steps, on_epoch=True, sync_dist=True)

            self.log("eval_cl_pred_min_predictor_loss_log", torch.log(eval_loss), on_epoch=True, sync_dist=True)


        last_prediction_loss = torch.log(eval_loss)

        # cross-correlation matrix
        c = (z1_norm.T @ z2_norm) / self.batch_size

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c)
            world_size = dist.get_world_size()
            c /= world_size

        #TODO: Calculating whole CC matrix just for diagonal?
        on_diag = torch.diagonal(c).add(-1).pow(2).sum()

        # note the minus as we try to maximize the prediction loss
        # TODO: Why multiply by output dim?
        # loss = on_diag - self.lamb * (self.proj_output_dim * last_prediction_loss)
        loss = on_diag - self.lamb * (last_prediction_loss)

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
        self.l2 = nn.Linear(feature_dim*2, feature_dim)
        self.bn2 = nn.BatchNorm1d(feature_dim)
        self.r2 = nn.LeakyReLU()
        self.l3 = nn.Linear(feature_dim, feature_dim)


    def forward(self, x):
        # w1 = self.bn1(self.r1(self.l1(x)))
        w2 = self.bn2(self.r2(self.l2(x)))
        return self.l3(w2)



def validate_predictor(predictability_net, args, input, true_output, masked_indices):
    # predictability_net.eval()
    val_outputs = predictability_net(input.cuda(non_blocking=True))
    prediction_loss = average_predictor_mse_loss(val_outputs.cuda(non_blocking=True), true_output.cuda(non_blocking=True), masked_indices.cuda(non_blocking=True)).mean()
    # del val_outputs
    return prediction_loss



class WidePredictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.l1 = nn.Linear(feature_dim*2, feature_dim*2*10)
        self.bn1 = nn.BatchNorm1d(feature_dim*2*10)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(feature_dim*2*10, feature_dim)

    def forward(self, x):
        w1 = self.bn1(self.r1(self.l1(x)))
        return self.l2(w1)


def to_dataset(embeddings, masking_target):
    batch_size, embedding_dimension = embeddings.shape
    ret_arr = torch.zeros(batch_size, 2*embedding_dimension,device=embeddings.device)
    masked_indices = torch.rand(batch_size, embedding_dimension,device=embeddings.device) < (masking_target / embedding_dimension)

    # copy over first half of input
    ret_arr[:, :embedding_dimension] = embeddings
    # mask the inputs we are predicting
    ret_arr[:, :embedding_dimension].masked_fill_(masked_indices, 0)
    # tell the neural network which ones we are predicting
    ret_arr[:, embedding_dimension:].masked_fill_(masked_indices, 1)

    return masked_indices, ret_arr


def individual_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)
    reconstruction = index_mask * diff_square

    # case that we divide by zero extremely unlikely
    prediction_error = reconstruction.sum(dim=0) / reconstruction.count_nonzero(dim=0)

    return prediction_error


def average_predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)

    only_prediction = torch.masked_select(diff_square, index_mask)
    prediction_error = torch.mean(only_prediction)

    return prediction_error


def predictor_mse_loss(predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor, gamma: float,
                       theta: float):
    """
    :param logger: if we pass a logger we will log the losses
    :param predictions: the LOO predictions of our network
    :param labels: the original embedding
    :param index_mask: the indices we are masking
    :param gamma: weight weighting error of masked error in sum with non-masked reconstruction error
    :return: the loss as a tensor
    """
    # print(f' predictions.shape {predictions.shape}, labels.shape {labels.shape}')
    assert predictions.shape == labels.shape
    # print(f'{index_mask}')
    # number_of_batches = predictions.shape[0]
    # number_of_elements_in_batch = predictions.shape[1]

    # get the squared difference of our neural networks output and the original labels
    diff_square = torch.square(predictions - labels)
    # print(diff_square.mean())
    # now we must calculate the two parts - the reconstruction error and the prediction error
    only_reconstruction = torch.masked_select(diff_square, torch.logical_not(index_mask))

    # recall that mask returns a one dimensional flattened tensor
    reconstruction_error = torch.mean(only_reconstruction)

    only_prediction = torch.masked_select(diff_square,index_mask)
    prediction_error = torch.mean(only_prediction)

    # in the unlikely case that all were selected or non were selected
    # TODO assumes we are running on a nvidia card
    if prediction_error.isnan():
        prediction_error = torch.tensor(0).cuda()

    if reconstruction_error.isnan():
        reconstruction_error = torch.tensor(0).cuda()

    total_loss = theta*prediction_error + gamma*reconstruction_error

    del prediction_error, reconstruction_error, only_prediction, only_reconstruction

    return total_loss

