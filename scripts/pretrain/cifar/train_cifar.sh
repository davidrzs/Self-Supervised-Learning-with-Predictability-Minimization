#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/maotth/log/%j.out     # where to store the output (%j is the JOBID), subdirectory "log" must exist
#SBATCH --error=/home/maotth/log/%j.err  # where to store error messages
#CommentSBATCH --exclude=tikgpu05,tikgpu07,tikgpu06,tikgpu10,tikgpu08
#CommentSBATCH --constraint='geforce_rtx_3090|rtx_a6000|titan_rtx'
#SBATCH --nodelist=tikgpu05
#CommentSBATCH --dependency=afterany:843069

#a100_80gb geforce_gtx_titan_x geforce_rtx_2080_ti geforce_rtx_3090 rtx_a6000 tesla_v100 titan_rtx titan_xp
# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi

# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow_cl_lin_pred_min.yaml
python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_non_lin_pred_minv6_1layer.yaml $@
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min_sgd.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml


#cp -r /itet-stor/maotth/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
#cp -r /itet-stor/maotth/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml $@


# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

