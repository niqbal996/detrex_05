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
    
    # using ema for evaluation
    ema.may_build_model_ema(cfg, model)
    DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
    # if True:
    #     # predictor = DefaultPredictor(cfg)
    #     generate_submission(model, None, '/mnt/d/datasets/phenobench/test/images')
    # elif visualize:
    #     generate_submission(model, predictor)
    # print(do_test(cfg, model, eval_only=True))
    model.eval()
    pred = {}
    images = glob(os.path.join('/netscratch/naeem/phenobench/test/images', '*.png'))
    img_data = []
    for img, c in zip(images, range(len(images))):
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
        outputs = model(img_data)
        # v = Visualizer(orig,              #im[:, :, ::-1]
        #        scale=1.0,
        #        instance_mode=ColorMode.IMAGE
        #    )
        # outputs = predictor(img_data)
        # out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
        # img = out.get_image()
        
        predictions = []
        for output in outputs:
            boxes = output['instances']._fields['pred_boxes'].tensor.cpu().detach().numpy()
            scores = output['instances']._fields['scores'].cpu().detach().numpy()
            labels = output['instances']._fields['pred_classes'].cpu().detach().numpy()
            indices = np.where(scores > 0.5)

            boxes = boxes[indices]
            scores = scores[indices]
            labels = labels[indices]

            filename = os.path.basename(img_dict['filename'])
            boxes_xywh = boxes.copy()
            # Convert XYXY to XYWH
            boxes_xywh[:, 0] = (boxes[:, 2] - boxes[:, 0])/2        # x_center
            boxes_xywh[:, 1] = (boxes[:, 3] - boxes[:, 1])/2        # y_center
            boxes_xywh[:, 2] = boxes[:, 2] - boxes[:, 0]            # width
            boxes_xywh[:, 3] = boxes[:, 3] - boxes[:, 1]            # height
            for idx in range(boxes_xywh.shape[0]):
                predictions.append([
                                    labels[idx], 
                                    boxes_xywh[idx, 0]/img_dict['width'], 
                                    boxes_xywh[idx, 1]/img_dict['height'], 
                                    boxes_xywh[idx, 2]/img_dict['width'],
                                    boxes_xywh[idx, 3]/img_dict['height'], 
                                    scores[idx]
                                    ])
        with open(os.path.join('/netscratch/naeem/phenobench/DETA_submission', filename[:-4]+'.txt'), 'w') as f:
            for line in predictions:
                string = " ".join(str(item) for item in line)
                f.write(string + '\n')
            # for box in boxes[indices]:
            #     cv2.rectangle(orig, 
            #                   (int(box[0]),int(box[1])),
            #                   (int(box[2]),int(box[3])),
            #                   color=(0, 255, 0),
            #                   thickness=2
            #     )
        # cv2.imshow('prediction', orig)
        # cv2.waitKey()
            


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
