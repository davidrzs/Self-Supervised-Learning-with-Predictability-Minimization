# Copyright 2023 solo-learn development team.
from typing import Tuple, Any

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

import torch
import torch.nn.functional as F

import torch.distributed as dist
from torch import Tensor


def cl_lin_pred_min_loss_func(
        z1: torch.Tensor, z2: torch.Tensor, lamb: float = 5e-3, mask_fraction: float = 0.5, ridge_lambd: float = 0.0
) -> tuple[Tensor, Any, Tensor]:
    """
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        lamb (float, optional): off-diagonal scaling factor for the cross-covariance matrix.
            Defaults to 5e-3.

    Returns:
        torch.Tensor: Barlow Twins' loss.
    """
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)

    batch_size, proj_output_dim = z1_norm.size()

    out_1_norm = (z1_norm - z1_norm.mean(dim=0)) / z1_norm.std(dim=0)
    out_2_norm = (z2_norm - z2_norm.mean(dim=0)) / z2_norm.std(dim=0)

    embeddings = torch.cat((out_1_norm, out_2_norm), 0)

    # sync all embeddings:
    if dist.is_available() and dist.is_initialized():
        embeddings_list = [torch.zeros_like(embeddings) for _ in range(dist.get_world_size())]
        dist.all_gather(embeddings_list, embeddings)
        embeddings = torch.cat(embeddings_list, dim=0)

    number_to_mask = int(proj_output_dim * mask_fraction)

    embedding_batch_size, embedding_dimension = embeddings.shape
    masked_indices = (torch.rand(embedding_batch_size, embedding_dimension, device=embeddings.device) < (
            number_to_mask / embedding_dimension))
    masked_embeddings = (~masked_indices) * embeddings
    X = torch.transpose(masked_embeddings, 0, 1) @ masked_embeddings
    X = X + ridge_lambd * torch.eye(proj_output_dim, device=embeddings.device)
    B = torch.transpose(masked_embeddings, 0, 1) @ (masked_indices * embeddings)
    W = torch.linalg.solve(X.float(), B.float())
    prediction_loss = average_predictor_mse_loss(masked_embeddings @ W, embeddings, masked_indices)

    del masked_embeddings, masked_indices

    # cross-correlation matrix
    corr = (out_1_norm.T @ out_2_norm) / batch_size

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(corr)
        world_size = dist.get_world_size()
        corr /= world_size

    diag_loss = torch.diagonal(corr).add(-1).pow(2).sum()

    # note the minus as we try to maximize the prediction loss
    total_loss = diag_loss - lamb * (proj_output_dim * prediction_loss)

    return total_loss, diag_loss, prediction_loss


def average_predictor_mse_loss(
        predictions: torch.Tensor, labels: torch.Tensor, index_mask: torch.Tensor
):
    assert predictions.shape == labels.shape

    diff_square = torch.square(predictions - labels)

    only_prediction = torch.masked_select(diff_square, index_mask)
    prediction_error = torch.mean(only_prediction)

    return prediction_error
