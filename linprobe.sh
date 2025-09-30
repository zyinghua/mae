#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 main_linprobe.py \
    --model vit_tiny_patch16 \
    --finetune /users/yzhou392/scratch/mae-output_dir/checkpoint-350.pth \
    --data_path /users/yzhou392/scratch/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 64 \
    --epochs 150 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --output_dir /users/yzhou392/scratch/mae-linprobe-out