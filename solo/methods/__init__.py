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

from solo.methods.barlow_twins import BarlowTwins
from solo.methods.barlow_twins_manual import BarlowTwinsManual
from solo.methods.cl_lin_pred_min import CLLinPredMin
from solo.methods.cl_non_lin_pred_min import CLNonLinPredMin
from solo.methods.cl_non_lin_pred_minv2 import CLNonLinPredMinv2
from solo.methods.cl_non_lin_pred_minv3 import CLNonLinPredMinv3
from solo.methods.cl_non_lin_pred_minv4 import CLNonLinPredMinv4
from solo.methods.cl_non_lin_pred_minv5 import CLNonLinPredMinv5
from solo.methods.cl_non_lin_pred_minv6 import CLNonLinPredMinv6
from solo.methods.cl_non_lin_pred_min_gan import CLNonLinPredMinGAN
from solo.methods.cl_non_lin_pred_min_gan_full import CLNonLinPredMinGANFull
from solo.methods.cl_non_lin_pred_minv5_man_opt import CLNonLinPredMinv5Man
from solo.methods.cl_non_lin_pred_min_single_step import CLNonLinPredMinSingleStep
from solo.methods.cl_lin_pred_min_sgd import CLLinPredMinSGD
from solo.methods.barlow_cl_lin_pred_min import BarlowCLLinPredMin
from solo.methods.base import BaseMethod
from solo.methods.byol import BYOL
from solo.methods.deepclusterv2 import DeepClusterV2
from solo.methods.dino import DINO
from solo.methods.linear import LinearModel
from solo.methods.mae import MAE
from solo.methods.mocov2plus import MoCoV2Plus
from solo.methods.mocov3 import MoCoV3
from solo.methods.nnbyol import NNBYOL
from solo.methods.nnclr import NNCLR
from solo.methods.nnsiam import NNSiam
from solo.methods.ressl import ReSSL
from solo.methods.simclr import SimCLR
from solo.methods.simsiam import SimSiam
from solo.methods.supcon import SupCon
from solo.methods.swav import SwAV
from solo.methods.vibcreg import VIbCReg
from solo.methods.vicreg import VICReg
from solo.methods.wmse import WMSE

METHODS = {
    # base classes
    "base": BaseMethod,
    "linear": LinearModel,
    # methods
    "cl_lin_pred_min": CLLinPredMin,
    "cl_non_lin_pred_min": CLNonLinPredMin,
    "cl_non_lin_pred_minv2": CLNonLinPredMinv2,
    "cl_non_lin_pred_minv3": CLNonLinPredMinv3,
    "cl_non_lin_pred_minv4": CLNonLinPredMinv4,
    "cl_non_lin_pred_minv5": CLNonLinPredMinv5,
    "cl_non_lin_pred_minv6": CLNonLinPredMinv6,
    "cl_non_lin_pred_min_gan": CLNonLinPredMinGAN,
    "cl_non_lin_pred_min_gan_full": CLNonLinPredMinGANFull,
    "cl_non_lin_pred_minv5_man_opt": CLNonLinPredMinv5Man,
    "cl_non_lin_pred_min_single_step": CLNonLinPredMinSingleStep,
    "cl_lin_pred_min_sgd": CLLinPredMinSGD,
    "barlow_cl_lin_pred_min": BarlowCLLinPredMin,  
    "barlow_twins": BarlowTwins,
    "barlow_twins_manual": BarlowTwinsManual,
    "byol": BYOL,
    "deepclusterv2": DeepClusterV2,
    "dino": DINO,
    "mae": MAE,
    "mocov2plus": MoCoV2Plus,
    "mocov3": MoCoV3,
    "nnbyol": NNBYOL,
    "nnclr": NNCLR,
    "nnsiam": NNSiam,
    "ressl": ReSSL,
    "simclr": SimCLR,
    "simsiam": SimSiam,
    "supcon": SupCon,
    "swav": SwAV,
    "vibcreg": VIbCReg,
    "vicreg": VICReg,
    "wmse": WMSE,
}
__all__ = [
    "BarlowTwins",
    "BarlowCLLinPredMin",
    "CLLinPredMin",
    "CLNonLinPredMin",
    "CLNonLinPredMinv2",
    "CLNonLinPredMinv3",
    "CLNonLinPredMinv4",
    "BYOL",
    "BaseMethod",
    "DeepClusterV2",
    "DINO",
    "MAE",
    "LinearModel",
    "MoCoV2Plus",
    "MoCoV3",
    "NNBYOL",
    "NNCLR",
    "NNSiam",
    "ReSSL",
    "SimCLR",
    "SimSiam",
    "SupCon",
    "SwAV",
    "VIbCReg",
    "VICReg",
    "WMSE",
]
