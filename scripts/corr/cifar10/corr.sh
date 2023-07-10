python3 main_corr.py \
    --dataset cifar10 \
    --batch_size 16 \
    --num_workers 10 \
    --train_data_path "./datasets" \
    --val_data_path "./datasets" \
    --pretrained_checkpoint_dir $1
