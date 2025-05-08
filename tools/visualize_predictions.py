#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.file_io import PathManager
from detectron2.checkpoint import DetectionCheckpointer
from detrex.utils import WandbWriter
from detrex.modeling import ema
from glob import glob
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
      
def generate_submission(model, predictor, image_root_dir=None):

    def toYOLO(boxes):
        pass

    
def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret
    
    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    
    target_layer = [model.backbone.layers[-1].blocks[-1].norm2]
    cam = GradCAM(model=model, target_layers=target_layer)
    
    # using ema for evaluation
    # ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    # if True:
    #     predictor = DefaultPredictor(cfg)
    #     generate_submission(model, None, '/mnt/d/datasets/phenobench/test/images')
    # elif visualize:
    #     generate_submission(model, predictor)
    # print(do_test(cfg, model, eval_only=True))
    model.eval()
    # images = glob(os.path.join('/mnt/e/datasets/MOT20_raps/test/sequence_2/img1/', '*.png'))
    images = glob(os.path.join('/mnt/e/datasets/phenobench/test/images', '*.png'))
    cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
    for img, c in zip(images, range(len(images))):
        img_data = []  # Reset img_data for each image
        img_dict = {}
        img_dict['filename'] = img
        img = cv2.imread(img)
        # img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_NEAREST)
        orig = img.copy()
        img = np.transpose(img, (2, 0, 1))
        img_dict['height'] = img.shape[1]
        img_dict['width'] = img.shape[2]
        img_dict['image_id'] = c
        img_dict['image'] = torch.tensor(img, device='cpu', dtype=torch.uint8)
        img_data.append(img_dict)
        # outputs = model(img_data)
        sample_cam = cam(input_tensor=img_dict['image'], targets=None)
        visualization = show_cam_on_image(orig, sample_cam)
        cv2.imshow('cam', visualization)
        # v = Visualizer(orig,
        #        scale=1.0,
        #        instance_mode=ColorMode.IMAGE
        #    )
        # out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        # img = out.get_image()
        # cv2.imshow('prediction', img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC key to exit
            break
    cv2.destroyAllWindows()  # Clean up the window at the end
            


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
