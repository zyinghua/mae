#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --model vit_tiny_patch16 \
    --finetune /root/scratch/vit-tiny-scratch-out/checkpoint-199.pth \
    --data_path /root/autodl-tmp/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 64 \
    --epochs 90 \
    --blr 0.005 \
    --weight_decay 0.0 \
    --output_dir /root/scratch/vit-tiny-linprobe-out