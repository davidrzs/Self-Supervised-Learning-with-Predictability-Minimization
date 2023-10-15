#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --exclude=tikgpu[01-10],artongpu01
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=15

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'analysis/correlation_analysis_cifar10/*'
# python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'analysis/correlation_analysis_cifar100/*'
python -m pipenv run python analysis/predict_all.py --model_type xgb --folder 'analysis/correlation_analysis_imagenet100/*'

# python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'analysis/correlation_analysis_cifar10/*'
# python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'analysis/correlation_analysis_cifar100/*'
# python -m pipenv run python analysis/predict_all.py --model_type linear --folder 'analysis/correlation_analysis_imagenet100/*'

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

