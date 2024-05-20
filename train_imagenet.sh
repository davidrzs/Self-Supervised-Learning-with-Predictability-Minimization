#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:2
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=6
#SBATCH --nodelist='tikgpu05' #tikgpu05 # Specify that it should run on this particular node artongpu01
##SBATCH --constraint='geforce_rtx_3090|titan_rtx|rtx_a6000'
##SBATCH --constraint='titan_rtx|rtx_a6000'

##SBATCH --nodelist=tikgpu08|tikgpu07 # Specify that it should run on this particular node

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi

TEMP_DIR=$(mktemp -d /scratch/imagenet_XXXXXXXXXX)

echo Starting copying

echo Copying to $TEMP_DIR


cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/


echo Copying done


# different lambdas

# python3  /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=100 method_kwargs.lamb=0.7 optimizer.batch_size=256 data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/



# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=100 method_kwargs.lamb=0.125 optimizer.batch_size=256 data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +seed=2

#  +seed=2

python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml  data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +method_kwargs.lamb=0.01



# here we do the projector layer ablation

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=1000 method_kwargs.proj_size=0 method_kwargs.proj_output_dim=512 data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=100 method_kwargs.proj_size=1 data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=100 method_kwargs.proj_size=2 data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/


# BARLOW PROJECTOR LAYER ABLATION

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml  data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +method_kwargs.proj_size=0

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml  data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +method_kwargs.proj_size=1

# python3 /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml  data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +method_kwargs.proj_size=2




rm -r $TEMP_DIR

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0



# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow_cl_lin_pred_min.yaml



# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/cifar/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=0 optimizer.batch_size=256  method_kwargs.mask_fraction=0.5  method_kwargs.lamb=2.0

# # # +seed=42

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name cl_lin_pred_min.yaml method_kwargs.ridge_lambd=500.0 method_kwargs.lamb=0.125

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_pretrain.py --config-path scripts/pretrain/imagenet-100/ --config-name barlow.yaml  data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/ +seed=2
