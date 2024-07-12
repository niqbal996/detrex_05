#!/usr/bin/env bash
cfg=$1
ulimit -s unlimited
./source.sh
python3 projects/deformable_detr/train_net.py --config-file projects/deformable_detr/configs/deformable_detr_r50_50ep_syn_v6_augmented.py --aug-index $cfg