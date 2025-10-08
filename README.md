## Masked Autoencoders for the [Galaxy10_DECALs dataset](https://huggingface.co/datasets/matthieulel/galaxy10_decals) in Pytorch, extended from the original [MAE repo](https://github.com/facebookresearch/mae).

## This is a brief introduction of the scripts for reproducing the results and ablations. Please firstly ensure you have the right environmental setup for the original [MAE repo](https://github.com/facebookresearch/mae).


#### Download Dataset:
Please run download_data.sh to download the galaxy dataset. Please note, the script by default is for Linux system. For MacOS and Windows, please try adjust the commands accordingly in the sh script. This script automatically downloads the data with git lfs, and extract them to actual images. Please also specify the desired place to store the actual images in the script (must align with the later sh scripts for actual training / lin probing).

```bash
chmod +x download_data.sh
./download_data.sh
```

#### MAE Training and Linear Probing Scripts:
You can find several sh scripts in this repo. "pretrain_" means pretraining the MAE **followed by** linear probing using the specific settings after the prefix, for example, `pretrain_dec256d4` means pretraining and linear probing the MAE with a decoder embedding dimension of 256 and a depth (number of ViT blocks) of 4, which uses ViT-B for the encoder by default, unless otherwise indicated in the file name. To reproduce the linear probing accuracy of our default MAE setting, run:

```bash
chmod +x pretrain_dec256d4.sh
./pretrain_dec256d4.sh
```

Please Note:
- The default gpu used is 4, you may change this in the corresponding sh script.
- You may also change the output_dir and log_dir in the scripts, as well as the data_path accordingly. But please ensure the paths match for pretrain and linear probing runs.
- in the script names, suffix `npl` means with pixel norm loss, by default we do not include the pixel norm loss. suffix `mr04` stands for masking ratio of '0.4'.


#### Supervised ViT Training Script:
```bash
chmod +x vit_pretrain.sh
./vit_pretrain.sh
```
Please adjust the parameters like paths accordingly.

#### MAE Finetuning Script:
```bash
chmod +x finetune.sh
./finetune.sh
```
Please adjust the parameters like paths accordingly. Especially the `--finetune` parameter, which indicates the path to which pretrained MAE backbone you'd like to finetune with.

The results will either be printed during the runs, or saved in the logs that you can inspect the accuracies.