#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --batch_size 64 \
  --model mae_vit_large_patch16 \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs 600 \
  --warmup_epochs 30 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --input_size 256 \
  --output_dir ./mae-output_dir \
  --log_dir ./mae-output_dir \
  --data_path "../autodl-tmp/galaxy10-dataset/"