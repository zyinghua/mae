#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --batch_size 64 \
  --model mae_vit_base_patch16_dec256d4b \
  --mask_ratio 0.75 \
  --epochs 200 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --input_size 224 \
  --data_path /root/autodl-tmp/galaxy10-dataset \
  --output_dir /root/scratch/mae-base-pretrain-output-dec256d4b \
  --log_dir /root/scratch/mae-base-pretrain-output-dec256d4b

python -m torch.distributed.launch --nproc_per_node=4 main_linprobe.py \
    --model vit_base_patch16 \
    --finetune /root/scratch/mae-base-pretrain-output-dec256d4b/checkpoint-199.pth \
    --data_path /root/autodl-tmp/galaxy10-dataset \
    --nb_classes 10 \
    --batch_size 64 \
    --epochs 90 \
    --blr 2e-3 \
    --weight_decay 0.0 \
    --output_dir /root/scratch/mae-base-linprobe-output-dec256d4b