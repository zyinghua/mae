#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_pretrain.py \
  --batch_size 64 \
  --model mmae_vit_base_patch16_dec512d8b \
  --mask_ratio 0.75 \
  --epochs 200 \
  --warmup_epochs 40 \
  --blr 1.5e-4 \
  --weight_decay 0.05 \
  --input_size 224 \
  --data_path ./galaxy10-dataset \
  --output_dir /root/scratch/mae-base-pretrain-output-dec512d8b \
  --log_dir /root/scratch/mae-base-pretrain-output-dec512d8b
