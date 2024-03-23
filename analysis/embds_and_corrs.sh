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

TEMP_DIR=$(mktemp -d /scratch/imagenet_XXXXXXXXXX)

echo Starting copying

echo Copying to $TEMP_DIR


# cp -r /itet-stor/maotth/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
# cp -r /itet-stor/maotth/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/


echo Copying done



#####################
#    CIFAR-10
#####################

# THE CLPM-GAN LAMBDA
python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
--pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/v5nusgni \
--pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/v5nusgni/cl_non_lin_pred_min_gan-cifar-v5nusgni-ep=999.ckpt


python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
--pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/34mfceo1 \
--pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/34mfceo1/cl_non_lin_pred_min_gan-cifar-34mfceo1-ep=999.ckpt


python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
--pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/lypojtew \
--pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/lypojtew/cl_non_lin_pred_min_gan-cifar-lypojtew-ep=999.ckpt



# # The CLPM-Fold ONES
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/5q928iom \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/5q928iom/cl_non_lin_pred_minv4-cifar-5q928iom-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/lof0xo67 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/lof0xo67/cl_non_lin_pred_minv4-cifar-lof0xo67-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/16zy824w \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv4/16zy824w/cl_non_lin_pred_minv4-cifar-16zy824w-ep=999.ckpt

# # THE CLPM-GAN ONES
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/07fx76z3 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/07fx76z3/cl_non_lin_pred_min_gan-cifar-07fx76z3-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/2a282f09 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/2a282f09/cl_non_lin_pred_min_gan-cifar-2a282f09-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/d8k8ptah \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/d8k8ptah/cl_non_lin_pred_min_gan-cifar-d8k8ptah-ep=999.ckpt


# # THE CLPM-Opt ONES
# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/xtq8vx5x \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/xtq8vx5x/cl_non_lin_pred_minv6-cifar-xtq8vx5x-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/szaxictv \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/szaxictv/cl_non_lin_pred_minv6-cifar-szaxictv-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/a03i1bo8 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/a03i1bo8/cl_non_lin_pred_minv6-cifar-a03i1bo8-ep=999.ckpt


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

# # THE CLPM-FOLD ONES
# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/qkqxeyrg \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/qkqxeyrg/cl_non_lin_pred_minv6-cifar100-qkqxeyrg-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/6mh52nl1 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/6mh52nl1/cl_non_lin_pred_minv6-cifar100-6mh52nl1-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/3j581xwa \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/3j581xwa/cl_non_lin_pred_minv6-cifar100-3j581xwa-ep=999.ckpt

# # THE CLPM-GAN ONES
# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/b861omg1 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/b861omg1/cl_non_lin_pred_min_gan-cifar100-b861omg1-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/oiolks2e \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/oiolks2e/cl_non_lin_pred_min_gan-cifar100-oiolks2e-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/tv8myt6j \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_min_gan/tv8myt6j/cl_non_lin_pred_min_gan-cifar100-tv8myt6j-ep=999.ckpt


# # THE CLPM-Opt ONES
# python -m pipenv run python main_corr.py  --dataset cifar100   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/0k74mj23 \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/0k74mj23/cl_non_lin_pred_minv6-cifar100-0k74mj23-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/7ao96clq \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/7ao96clq/cl_non_lin_pred_minv6-cifar100-7ao96clq-ep=999.ckpt


# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/vsdqcduo \
# --pretrained_checkpoint /itet-stor/maotth/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_non_lin_pred_minv6/vsdqcduo/cl_non_lin_pred_minv6-cifar100-vsdqcduo-ep=999.ckpt


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








##########################
#    OTHER EXPERIMENTS
##########################

# python -m pipenv run python main_corr.py  --dataset cifar10   --batch_size 16    --num_workers 10  --train_data_path "./datasets" --val_data_path "./datasets" \
# --pretrained_checkpoint_dir /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/highpred \
# --pretrained_checkpoint /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/highpred/cl_non_lin_pred_minv6-cifar-bk4u6k9o-ep=999.ckpt





rm -r $TEMP_DIR

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

