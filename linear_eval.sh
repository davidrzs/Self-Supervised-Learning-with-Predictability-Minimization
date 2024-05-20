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
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name barlow.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name byol.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name mocov3.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name nnclr.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name simclr.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vibcreg.yaml 
# python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vicreg.yaml 
python -m pipenv run python /itet-stor/$USER/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml 

TEMP_DIR=$(mktemp -d /scratch/imagenet_XXXXXXXXXX)

echo Starting copying

echo Copying to $TEMP_DIR


cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/


echo Copying done


# THE THREE CLPM-REG ONES

# TODO

# THE THREE CLPM-REG ONES WITH LAMBDA = 0

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/050e07dt/cl_lin_pred_min-imagenet100-050e07dt-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/d3a5u4lj/cl_lin_pred_min-imagenet100-d3a5u4lj-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/4zspgu31/cl_lin_pred_min-imagenet100-4zspgu31-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/




# THE THREE CLPM-REG ONES

python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/96797f2p/cl_lin_pred_min-imagenet100-96797f2p-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/xq4q2wcs/cl_lin_pred_min-imagenet100-xq4q2wcs-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/cl_lin_pred_min/1tvbr9z9/cl_lin_pred_min-imagenet100-1tvbr9z9-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/


# THE THREE BARLOW ONES

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name barlow.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/l2heft8h/barlow_twins-imagenet100-l2heft8h-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name barlow.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/426ke3f4/barlow_twins-imagenet100-426ke3f4-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name barlow.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/barlow_twins/g6y9vmei/barlow_twins-imagenet100-g6y9vmei-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/


# THE OTHERS

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name byol.yaml  pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/byol/xcxvnzak/byol-imagenet100-xcxvnzak-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name mocov3.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/mocov3/o62f501b/mocov3-imagenet100-o62f501b-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name nnclr.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/nnclr/dsviitnf/nnclr-imagenet100-dsviitnf-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name simclr.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/simclr/98pa242t/simclr-imagenet100-98pa242t-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vibcreg.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vibcreg/xunwz815/vibcreg-imagenet100-xunwz815-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/

# python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name vicreg.yaml pretrained_feature_extractor="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models/vicreg/6tfibs0s/vicreg-imagenet100-6tfibs0s-ep\=399.ckpt" data.train_path=$TEMP_DIR/ILSVRC2012_img_train_100/ data.val_path=$TEMP_DIR/ILSVRC2012_img_val_100/



# # python -m pipenv run python /itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/main_linear.py --config-path scripts/linear/imagenet-100/ --config-name cl_lin_pred_min.yaml 


rm -r $TEMP_DIR

# Send more noteworthy information to the output log
echo "Finished at:     $(date)"

# End the script with exit code 0
exit 0

