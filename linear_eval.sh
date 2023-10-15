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
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name barlow.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name byol.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name mocov3.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name nnclr.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name simclr.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vibcreg.yaml 
# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vicreg.yaml 
python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml 


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

