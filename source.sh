#!/usr/bin/env bash
cd /home/iqbal/detrex_orig
python3 -m pip install -e detectron2 
python3 -m pip install -e .
export PYTHONPATH=/home/iqbal/detrex_orig:/home/iqbal/detrex_orig/detectron2