#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py \
  --batch_size 128 \
  --model mae_vit_tiny_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.5 \
  --epochs 800 \
  --warmup_epochs 40 \
  --blr 2e-4 \
  --weight_decay 0.05 \
  --input_size 224 \
  --output_dir ../scratch/mae-pretrain-output-dir \
  --log_dir ../scratch/mae-pretrain-output-dir \
  --data_path "../scratch/galaxy10-dataset/"