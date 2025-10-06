#!/bin/bash

# download dataset (parquet files) from huggingface with git lfs
# Please choose the correct package to download depending on your OS (use brew with MacOS)
apt-get update
apt-get install git-lfs # brew install git-lfs
git lfs install
git clone https://huggingface.co/datasets/matthieulel/galaxy10_decals

# Extract parquet files to actual images
python extract_parquet_to_folder.py --src ./galaxy10_decals/data --dst ./galaxy10-dataset