#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --batch_size 64 \
  --model mae_vit_base_patch16 \
  --mask_ratio 0.6 \
  --epochs 800 \
  --warmup_epochs 40 \
  --blr 1.5e-3 \
  --weight_decay 0.05 \
  --input_size 224 \
  --output_dir /root/scratch/mae-pretrain-output-dir \
  --log_dir /root/scratch/mae-pretrain-output-dir \
  --data_path /root/autodl-tmp/galaxy10-dataset