#!/bin/bash
#SBATCH --job-name=BT_paper
#SBATCH --gres=gpu:1
#SBATCH --mem=25GB
#SBATCH --cpus-per-task=4
#SBATCH --constraint='titan_rtx|rtx_a6000'
####   sssBATCH --constraint='geforce_rtx_3090|titan_rtx|rtx_a6000'
#SssBATCH --nodelist=tikgpu[04-08]


# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

nvidia-smi


ids=(
wuvntua7
tf5twl2m
pyy0vdxo
)

base_path="/itet-stor/zdavid/net_scratch/Self-Supervised-Learning-with-Predictability-Minimization/trained_models"



# # START IMAGENET STUFF
TEMP_DIR=$(mktemp -d /scratch/imagenet_XXXXXXXXXX)

echo Starting copying


cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_train_100/ $TEMP_DIR/ILSVRC2012_img_train_100/
cp -r /itet-stor/zdavid/net_scratch/data/ILSVRC2012_img_val_100/ $TEMP_DIR/ILSVRC2012_img_val_100/

echo Copying done

# END IMAGENET STUFF


# Iterate over each ID
for id in "${ids[@]}"; do
    # Find the folder that contains the ID
    checkpoint_dir=$(find $base_path -type d -name $id)

    # Find all checkpoint files within this directory and choose the one with the highest epoch number
    checkpoint_files=$(find $checkpoint_dir -name '*.ckpt')
    highest_epoch_file=""
    highest_epoch_number=-1

    for file in $checkpoint_files; do
        epoch_number=$(echo $file | sed -n 's/.*-ep=\([0-9]*\)\.ckpt/\1/p')
        if [[ $epoch_number -gt $highest_epoch_number ]]; then
            highest_epoch_number=$epoch_number
            highest_epoch_file=$file
        fi
    done

    # If no checkpoint file was found, continue to the next ID
    if [[ -z $highest_epoch_file ]]; then
        echo "No checkpoint files found for ID $id"
        continue
    fi

    # Extract dataset name from the checkpoint file name
    dataset=$(basename $highest_epoch_file | cut -d'-' -f2)

    echo "starting extracting checkpointfile $highest_epoch_file"


    # Process only if dataset is imagenet100
    if [[ $dataset == "imagenet100" ]]; then

        # Construct the command and execute it
        python3 main_corr.py --dataset $dataset --batch_size 16 --num_workers 10 \
        --train_data_path "$TEMP_DIR/ILSVRC2012_img_train_100/" --val_data_path "$TEMP_DIR/ILSVRC2012_img_val_100/" \
        --pretrained_checkpoint_dir $checkpoint_dir \
        --pretrained_checkpoint $highest_epoch_file

    else
        python3 main_corr.py --dataset $dataset --batch_size 16 --num_workers 10 \
        --train_data_path "./datasets" --val_data_path "./datasets" \
        --pretrained_checkpoint_dir $checkpoint_dir \
        --pretrained_checkpoint $highest_epoch_file

    fi
    echo "finished extracting checkpointfile $highest_epoch_file"

done

# remove the imagenet directory
rm -r $TEMP_DIR


