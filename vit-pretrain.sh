#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
  --model vit_tiny_patch16 \
  --data_path /root/autodl-tmp/galaxy10-dataset \
  --nb_classes 10 \
  --input_size 256 \
  --batch_size 64 \
  --epochs 150 \
  --blr 1e-3 \
  --weight_decay 0.1 \
  --drop_path 0.1 \
  --warmup_epochs 10 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 1.0 \
  --output_dir /root/scratch/vit-tiny-scratch-out