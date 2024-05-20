#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/maotth/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/maotth/log/%j.err  # where to store error messages
#CommentSBATCH --exclude=tikgpu08,tikgpu10
#SBATCH --constraint='geforce_rtx_3090|rtx_a6000'
#CommentSBATCH --nodelist=tikgpu09
#SBATCH --dependency=afterany:818382

#a100_80gb geforce_gtx_titan_x geforce_rtx_2080_ti geforce_rtx_3090 rtx_a6000 tesla_v100 titan_rtx titan_xp
# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi

# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow_cl_lin_pred_min.yaml
python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_non_lin_pred_minv6.yaml $@
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_sgd.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml  data.dataset=cifar100  name="barlow_twins-cifar100" +method_kwargs.lamb=0.05


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml +method_kwargs.lamb=0.013


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml method_kwargs.proj_size=0 method_kwargs.ridge_lambd=1000 method_kwargs.proj_output_dim=512 name="cl_lin_pred_min-cifar100" data.dataset=cifar100 


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml method_kwargs.proj_size=1 name="cl_lin_pred_min-cifar100" data.dataset=cifar100 


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml method_kwargs.proj_size=2 name="cl_lin_pred_min-cifar100" data.dataset=cifar100 



# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml method_kwargs.proj_size=0 method_kwargs.ridge_lambd=10000 method_kwargs.proj_output_dim=512



# HERE WE 


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml method_kwargs.proj_size=0  method_kwargs.proj_output_dim=512 data.dataset=cifar100  name="barlow_twins-cifar100"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml method_kwargs.proj_size=1 data.dataset=cifar100  name="barlow_twins-cifar100"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml method_kwargs.proj_size=2 data.dataset=cifar100  name="barlow_twins-cifar100"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml method_kwargs.proj_size=3 data.dataset=cifar100  name="barlow_twins-cifar100"


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml method_kwargs.proj_size=3 




# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/2dybvyfy/barlow_twins-cifar10-2dybvyfy-ep\=999.ckpt"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml optimizer.lr=0.05 +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml optimizer.lr=0.05 +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt" optimizer.lr=0.05


# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml optimizer.lr=0.01 +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml optimizer.lr=0.005 +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt"

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml optimizer.lr=0.001 +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep\=999.ckpt"

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xv8lebu2/byol-cifar10-xv8lebu2-ep\=999.ckpt"

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/bb78kfkq/nnclr-cifar10-bb78kfkq-ep\=999.ckpt"

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/5moi8bma/mocov3-cifar10-5moi8bma-ep\=999.ckpt"

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/va9eh12u/vicreg-cifar10-va9eh12u-ep\=999.ckpt"

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain_cl_lin_pred_min.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_finetune.yaml +pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/rg5jdr51/vibcreg-cifar10-rg5jdr51-ep\=999.ckpt"


# barlowtwins	2dybvyfy
# simclr	w4efdan1
# byol	xv8lebu2
# nnclr	bb78kfkq
# mocov3	5moi8bma
# vicreg	va9eh12u
# vibcreg	rg5jdr51

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

