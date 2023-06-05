python3 main_corr.py \
    --dataset imagenet-100 \
    --train_data_path ./datasets/imagenet-100/train \
    --val_data_path ./datasets/imagenet-100/val \
    --batch_size 16 \
    --num_workers 10 \
    --pretrained_checkpoint_dir $1
