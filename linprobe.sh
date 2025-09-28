#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 python main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /users/yzhou392/scratch/mae-output_dir/checkpoint-350.pth \
    --data_path /users/yzhou392/scratch/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 32 \
    --epochs 50 \
    --blr 0.003 \
    --weight_decay 0.0 \
    --output_dir /users/yzhou392/scratch/linprobe_out