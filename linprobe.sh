#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=6 main_linprobe.py \
    --model vit_tiny_patch16 \
    --finetune /root/scratch/mae-pretrain-output-dir/checkpoint-799.pth \
    --data_path /root/autodl-tmp/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 64 \
    --epochs 120 \
    --blr 1e-3 \
    --weight_decay 0.0 \
    --output_dir /root/scratch/mae-linprobe-output-dir