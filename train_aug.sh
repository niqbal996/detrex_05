#!/usr/bin/env bash
aug_index=$1
ulimit -s unlimited
./source.sh
echo $aug_index
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_pheno_augmented_full.py --aug-index $aug_index