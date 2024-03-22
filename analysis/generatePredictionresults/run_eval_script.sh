#!/bin/bash

#SBATCH -n 1
#SBATCH --cpus-per-task=20
#SBATCH --time=4:00:00
#SBATCH --job-name="eval"
#SBATCH --mem-per-cpu=5120


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"



# module load gcc/8.2.0 python_gpu/3.10.4

# lll.sh



python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'correlation_analysis/cifar10/*'
# python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'correlation_analysis/cifar100/*'
# python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'correlation_analysis/imagenet_100/*'


# python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'correlation_analysis/cifar10/*'
# python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'correlation_analysis/cifar100/*'
# python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'correlation_analysis/imagenet_100/*'


# python -m pipenv run python analysis/predict_all.py --model_type mlp --folder 'correlation_analysis/cifar10/*'
# python -m pipenv run python analysis/predict_all.py --model_type mlp --folder 'correlation_analysis/cifar100/*'
# python -m pipenv run python analysis/predict_all.py --model_type mlp --folder 'correlation_analysis/imagenet_100/*'


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

