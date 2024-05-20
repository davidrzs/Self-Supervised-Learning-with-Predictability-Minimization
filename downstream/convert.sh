#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=4
#SBATCH --constraint='geforce_rtx_3090|titan_rtx|rtx_a6000'
##SBATCH --nodelist=tikgpu06 #tikgpu05 # Specify that it should run on this particular node

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi




python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/downstream/object_detection/convert_model_to_detectron2.py --pretrained_feature_extractor PATH_TO_CKPT --output_detectron_model ./detectron_model.pkl



# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

