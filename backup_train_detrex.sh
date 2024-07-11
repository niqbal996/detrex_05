#!/usr/bin/env bash
ulimit -s unlimited
export CUDA_HOME=/usr/local/cuda-11.8/
export PYTHONPATH=$PYTHONPATH:/home/niqbal/detrex/detectron2
# python3 tools/train_net.py --config-file projects/dino/configs/dino_r50_4scale_12ep_maize.py
# python3 tools/train_net.py --config-file projects/deta/configs/deta_r50_5scale_12ep_phenobench.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_r50_5scale_12ep_pheno_syn_baseline.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_r50_5scale_12ep_pheno_syn_augmented.py
python3 projects/deta/train_net.py --config-file projects/deta/configs/deta_r50_5scale_12ep_pheno_syn_secogan.py
# python3 tools/visualize_predictions.py --config-file projects/deta/configs/deta_r50_5scale_12ep_phenobench.py --eval-only
# python3 tools/lazyconfig_train_net.py --config-file configs/COCO-Detection/retinanet_R_50_FPN_1x.py
