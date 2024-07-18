#!/usr/bin/env bash
subset_index=$1
ulimit -s unlimited
./source.sh
export PYTHONPATH=/home/iqbal/detrex_05:/home/iqbal/detrex_05/detectron2
echo $subset_index
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_pheno_augmented.py --subset-index $subset_index