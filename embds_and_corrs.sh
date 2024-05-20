#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=4
#SBATCH --constraint='geforce_rtx_3090|titan_rtx|rtx_a6000'


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi

# TEMP_DIR=$(mktemp -d /scratch/imagenet_XXXXXXXXXX)

# echo Starting copying

# echo Copying to $TEMP_DIR


# cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
# cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/


# echo Copying done



#####################
#    CIFAR-10
#####################

# THE CLPM-REG ONES

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/39z5c522 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/39z5c522/cl_lin_pred_min-cifar10-39z5c522-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/u88oucxk \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/u88oucxk/cl_lin_pred_min-cifar10-u88oucxk-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/pp1r80bl \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/pp1r80bl/cl_lin_pred_min-cifar10-pp1r80bl-ep=999.ckpt


# THE CLPM-REG ONES LAMBDA = 0

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1zdb4m23 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1zdb4m23/cl_lin_pred_min-cifar10-1zdb4m23-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8p3h88c1 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8p3h88c1/cl_lin_pred_min-cifar10-8p3h88c1-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ff7mu4is \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ff7mu4is/cl_lin_pred_min-cifar10-ff7mu4is-ep=999.ckpt


# THE THREE BARLOW ONES


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/2dybvyfy \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/2dybvyfy/barlow_twins-cifar10-2dybvyfy-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/51hzzbv7 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/51hzzbv7/barlow_twins-cifar10-51hzzbv7-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16  --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/8y16abfy \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/8y16abfy/barlow_twins-cifar10-8y16abfy-ep=999.ckpt

# THE OTHER ONES

# #SIMCLR
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/w4efdan1/simclr-cifar10-w4efdan1-ep=999.ckpt

# #BYOL
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xv8lebu2 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xv8lebu2/byol-cifar10-xv8lebu2-ep=999.ckpt

# #NNCLR
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/bb78kfkq \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/bb78kfkq/nnclr-cifar10-bb78kfkq-ep=999.ckpt

# #MOCOV3
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/5moi8bma \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/5moi8bma/mocov3-cifar10-5moi8bma-ep=999.ckpt

# #VICREG
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir  /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/va9eh12u \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/va9eh12u/vicreg-cifar10-va9eh12u-ep=999.ckpt


# #VIBCREG
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16  --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/rg5jdr51 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/rg5jdr51/vibcreg-cifar10-rg5jdr51-ep=999.ckpt


#####################
#    CIFAR-100
#####################


# THE CLPM-REG ONES

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/v14qlj0j \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/v14qlj0j/cl_lin_pred_min-cifar100-v14qlj0j-ep=999.ckpt



# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/fnydjain \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/fnydjain/cl_lin_pred_min-cifar100-fnydjain-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/te4j1ahd \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/te4j1ahd/cl_lin_pred_min-cifar100-te4j1ahd-ep=999.ckpt


# THE CLPM-REG ONES LAMBDA = 0

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/wwgtuu05 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/wwgtuu05/cl_lin_pred_min-cifar100-wwgtuu05-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/g8xjcnw5 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/g8xjcnw5/cl_lin_pred_min-cifar100-g8xjcnw5-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8r804qws \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8r804qws/cl_lin_pred_min-cifar100-8r804qws-ep=999.ckpt


# THE THREE BARLOW ONES


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/wyupong2 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/wyupong2/barlow_twins-cifar100-wyupong2-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/fo47ne50 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/fo47ne50/barlow_twins-cifar100-fo47ne50-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16  --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/sddrrq44 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/sddrrq44/barlow_twins-cifar100-sddrrq44-ep=999.ckpt

# THE OTHER ONES

# #SIMCLR
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/whfaafhs \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/whfaafhs/simclr-cifar100-whfaafhs-ep=999.ckpt

# #BYOL
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/komqiy40 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/komqiy40/byol-cifar100-komqiy40-ep=999.ckpt

# #NNCLR
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/figr6vi2 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/figr6vi2/nnclr-cifar100-figr6vi2-ep=999.ckpt

# #MOCOV3
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16   --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/4hkmtdpi \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/4hkmtdpi/mocov3-cifar100-4hkmtdpi-ep=999.ckpt

# #VICREG
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir  /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/8wav6tjf \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/8wav6tjf/vicreg-cifar100-8wav6tjf-ep=999.ckpt


# #VIBCREG
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16  --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/raeiejc5 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/raeiejc5/vibcreg-cifar100-raeiejc5-ep=999.ckpt



#####################
#    IMAGENET-100
#####################

# THE CLPM-REG ONES

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/96797f2p" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/96797f2p/cl_lin_pred_min-imagenet100-96797f2p-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/xq4q2wcs" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/xq4q2wcs/cl_lin_pred_min-imagenet100-xq4q2wcs-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1tvbr9z9" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1tvbr9z9/cl_lin_pred_min-imagenet100-1tvbr9z9-ep=399.ckpt"

# THE CLPM-REG ONES LAMBDA = 0

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/050e07dt" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/050e07dt/cl_lin_pred_min-imagenet100-050e07dt-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/d3a5u4lj" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/d3a5u4lj/cl_lin_pred_min-imagenet100-d3a5u4lj-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/4zspgu31" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/4zspgu31/cl_lin_pred_min-imagenet100-4zspgu31-ep=399.ckpt"


# THE THREE BARLOW ONES

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/l2heft8h" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/l2heft8h/barlow_twins-imagenet100-l2heft8h-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/426ke3f4" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/426ke3f4/barlow_twins-imagenet100-426ke3f4-ep=399.ckpt"

# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/g6y9vmei" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/g6y9vmei/barlow_twins-imagenet100-g6y9vmei-ep=399.ckpt"


# OTHERS

# # VICREG
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/6tfibs0s/" --pretrained_checkpoint  "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/6tfibs0s/vicreg-imagenet100-6tfibs0s-ep=399.ckpt"

# # VIBCREG
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/xunwz815/" --pretrained_checkpoint "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/xunwz815/vibcreg-imagenet100-xunwz815-ep=399.ckpt"
# # SIMCLR
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/98pa242t/" --pretrained_checkpoint "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/98pa242t/simclr-imagenet100-98pa242t-ep=399.ckpt"

# ## NNCLR
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/dsviitnf/" --pretrained_checkpoint "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/dsviitnf/nnclr-imagenet100-dsviitnf-ep=399.ckpt"

# # MOCOV3
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/o62f501b/" --pretrained_checkpoint "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/o62f501b/mocov3-imagenet100-o62f501b-ep=399.ckpt"

# # BYOL
# python -m pipenv run python main_corr.py  --dataset imagenet100    --train_data_path $TEMP_DIR/ILSVRC2012_img_train_100/   --val_data_path $TEMP_DIR/ILSVRC2012_img_val_100/  --batch_size 16  --num_workers 10     --pretrained_checkpoint_dir "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xcxvnzak/" --pretrained_checkpoint "/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xcxvnzak/byol-imagenet100-xcxvnzak-ep=399.ckpt"



####################################
#    DIFFERENT LAMBDAS OF CLPM-REG
####################################

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/slbcwy55 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/slbcwy55/cl_lin_pred_min-cifar10-slbcwy55-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/psfjoxre \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/psfjoxre/cl_lin_pred_min-cifar10-psfjoxre-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ln8089up \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ln8089up/cl_lin_pred_min-cifar10-ln8089up-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/u88oucxk \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/u88oucxk/cl_lin_pred_min-cifar10-u88oucxk-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/oxf7yrsz \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/oxf7yrsz/cl_lin_pred_min-cifar10-oxf7yrsz-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/wqqn8xme \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/wqqn8xme/cl_lin_pred_min-cifar10-wqqn8xme-ep=999.ckpt


# ####################################
# # DIFFERENT MASKING FRACTIONS
# ####################################


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/xhsvxva3 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/xhsvxva3/cl_lin_pred_min-cifar10-xhsvxva3-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/tzmqz9sy \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/tzmqz9sy/cl_lin_pred_min-cifar10-tzmqz9sy-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8qombpuc \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8qombpuc/cl_lin_pred_min-cifar10-8qombpuc-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/874vs5p6 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/874vs5p6/cl_lin_pred_min-cifar10-874vs5p6-ep=999.ckpt



# ####################################
# # DIFFERENT RIDGE PENALTIES
# ####################################


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1mesi5r8 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1mesi5r8/cl_lin_pred_min-cifar10-1mesi5r8-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/19ih9ab2 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/19ih9ab2/cl_lin_pred_min-cifar10-19ih9ab2-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/d5lzo1q1 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/d5lzo1q1/cl_lin_pred_min-cifar10-d5lzo1q1-ep=999.ckpt

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/a4rpbnje \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/a4rpbnje/cl_lin_pred_min-cifar10-a4rpbnje-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8chdi18b \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/8chdi18b/cl_lin_pred_min-cifar10-8chdi18b-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/3xruot5n \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/3xruot5n/cl_lin_pred_min-cifar10-3xruot5n-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ljhtku68 \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/ljhtku68/cl_lin_pred_min-cifar10-ljhtku68-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/o0quw92r \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/o0quw92r/cl_lin_pred_min-cifar10-o0quw92r-ep=999.ckpt




##########################
#    OTHER EXPERIMENTS
##########################

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/highpred \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/highpred/cl_non_lin_pred_minv6-cifar-bk4u6k9o-ep=999.ckpt


python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
--pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/2hkog81r \
--pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/2hkog81r/barlow_twins-cifar10-2hkog81r-ep=999.ckpt


# rm -r $TEMP_DIR

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0



# missing ones
# 2hyow7rf
# g1q2jsd7
# qrgdjbkb
# e6x3jlk3
# ay8azlvc
# fe4cjgd2
# 050e07dt
# 050e07dt
# d3a5u4lj
# 4zspgu31
# 96797f2p
# xq4q2wcs
# 1tvbr9z9
# l2heft8h
# 426ke3f4
# g6y9vmei
# hm6vpsgl
# hwjywyfj
# v7c9ciuu
# 66w0odub
# w4efdan1
# bb78kfkq
# 5moi8bma
# va9eh12u
# rg5jdr51
# whfaafhs
# figr6vi2
# 4hkmtdpi
# 8wav6tjf
# raeiejc5
# 98pa242t
# xcxvnzak
# dsviitnf
# o62f501b
# 6tfibs0s
# xunwz815
# dljb716z
# zt39a0pc
# t74rnyf4
# qwz6tivf
# 7qhzqnlo
# y06h7d8t
# pqsd7ifg
# 1i4f3xo5
# 00unwqu2
# kkxhz10p
# oe4ull10
# 7g8gcupz
# 7wj33ivg
# m80opzkh
# fg80k861
# gq8h0brz
# oz435lp2
# 6d1drs8y