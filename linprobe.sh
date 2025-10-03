#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /root/scratch/mae-base-pretrain-output-dir/checkpoint-199.pth \
    --data_path /root/autodl-tmp/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 64 \
    --epochs 120 \
    --blr 2e-3 \
    --weight_decay 0.0 \
    --output_dir /root/scratch/mae-base-linprobe-output