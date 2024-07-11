#!/usr/bin/env bash
ulimit -s unlimited
export CUDA_HOME=/usr/local/cuda-11.8/
export PYTHONPATH=$PYTHONPATH:/workspace/detrex:workspace/detrex/detectron2
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_10.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_20.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_30.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_40.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_50.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_60.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_70.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_80.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_90.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_100.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_maize.py
# python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_pheno.py
python3 tools/train_net.py --config-file projects/deta/configs/deta_swin_large_finetune_24ep_pheno.py