#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=4 main_finetune.py \
  --model vit_base_patch16 \
  --data_path ./galaxy10-dataset \
  --finetune /root/scratch/mae-base-pretrain-output-dec256d4b/checkpoint-199.pth \
  --nb_classes 10 \
  --input_size 224 \
  --batch_size 64 \
  --epochs 130 \
  --blr 1e-3 \
  --cls_token \
  --weight_decay 0.1 \
  --drop_path 0.1 \
  --warmup_epochs 10 \
  --smoothing 0.1 \
  --mixup 0.5 \
  --cutmix 1.0 \
  --output_dir /root/scratch/mae-finetune-output-dir