#!/usr/bin/env bash
ulimit -s unlimited
export CUDA_HOME=/usr/local/cuda-11.8/
export PYTHONPATH=$PYTHONPATH:/workspace/detrex/detectron2
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_10_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_20_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_30_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_40_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_50_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_60_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_70_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_80_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_90_source.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_100_source.py