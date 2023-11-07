#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8
##SBATCH --constraint='geforce_rtx_3090|titan_rtx|rtx_a6000'


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi

python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow_cl_lin_pred_min.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name barlow.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml



# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

